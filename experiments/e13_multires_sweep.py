"""
E13 — Multi-Resolution Temporal Sweep (thí nghiệm "đinh" cho bài Q1/Q2)
========================================================================
Câu hỏi khoa học:
  Giá trị của bộ nhớ thời gian (LSTM/Hybrid) là hàm của sampling interval Δt
  so với hằng số thời gian nhiệt của module (τ ≈ 7–12 min).
  - JST paper:   Δt = 15 min (nowcast, DuraMAT)  -> LSTM vô dụng
  - MEAFD paper: Δt = 5 min  (forecast, PVDAQ)   -> Hybrid thắng ở GHI cao
  -> Sweep Δt ∈ {5, 10, 15, 30, 60} min × {nowcast, forecast} để tìm ngưỡng.

Nguồn dữ liệu: processed/pvdaq_7333_v2_2022_2023.csv (5-min, đã QC).
  Bản 10-giây gốc KHÔNG có trên đĩa -> Δt < 5 min nằm ngoài phạm vi (ghi rõ
  trong bài; có thể bổ sung sau bằng cách tải lại từ NREL OpenEI).
  Gộp lên Δt thô hơn bằng mean, chỉ nhận bin đủ số mẫu con (count == Δt/5).

Khác biệt phương pháp so với 2 bài trước (ghi chú trong manuscript):
  1. Gap-aware windowing: cửa sổ chỉ hợp lệ nếu timestamp liên tục đúng Δt
     (code cũ xếp hàng liên tiếp bất chấp khoảng trống dữ liệu).
  2. Validation: 10% cuối của 2022 theo thứ tự thời gian cho MỌI model
     (JST dùng random 85/15 cho MLP — thống nhất lại để so sánh công bằng).

Protocol:
  - Train 2022 / Test 2023 (temporal split, không leakage; scaler fit trên train).
  - Adam 1e-3, MSE, EarlyStopping(patience=20, restore best), ReduceLR(0.5, 10),
    batch 256, max 200 epochs (theo protocol MEAFD).
  - Seeds: range(N_SEEDS), mặc định 10 (prefix của range(20) — mở rộng được).

Models:
  - Faiman (reference, nowcast only)
  - PG-ResMLP [32,16], 4 physics features, point-wise (705 params)
  - LSTM-64  (6 raw features, window 6 steps — spec MEAFD)
  - Hybrid Transformer-LSTM (spec MEAFD, ~47K params)

Modes:
  - nowcast : dự đoán T_module(t) từ thông tin đến t (window kết thúc tại t)
  - forecast: dự đoán T_module(t+1) từ thông tin đến t (protocol MEAFD)

Chạy:
  python e13_multires_sweep.py --smoke            # kiểm tra end-to-end ~2 phút
  python e13_multires_sweep.py                    # full sweep (nhiều giờ)
  python e13_multires_sweep.py --seeds 20         # mở rộng lên n=20
  python e13_multires_sweep.py --window-sweep     # phụ: sensitivity window @5min

Output (ghi tăng dần sau mỗi run — an toàn khi ngắt giữa chừng):
  results/e13_multires_sweep.csv
  results/e13_predictions.npz  (y_true, ghi, pred từng run — cho figure + UQ)
"""
import io, os, sys, json, time, argparse, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

DATA = r'c:\Users\admin\NCKH\processed\pvdaq_7333_v2_2022_2023.csv'
RESULT_DIR = r'c:\Users\admin\NCKH\results'
CSV_OUT = os.path.join(RESULT_DIR, 'e13_multires_sweep.csv')
NPZ_OUT = os.path.join(RESULT_DIR, 'e13_predictions.npz')

RAW_FEATURES = ['T_ambient_mean', 'GHI_mean', 'POA_mean', 'Wind_speed_mean',
                'hour', 'day_of_year']            # cho sequence models (MEAFD)
PHYS_FEATURES = ['T_Faiman', 'T_ambient_mean', 'Wind_speed_mean', 'GHI_lag']
TARGET = 'T_module_mean'
WINDOW = 6                                         # 6 bước (spec MEAFD)
RESOLUTIONS = [60, 30, 15, 10, 5]                  # phút (thô->mịn: rẻ ra trước)
LAG_TARGET_MIN = 30                                # GHI_lag ~ 30 phút (JST)
U0, U1 = 25.0, 6.84                                # Faiman Negev defaults
EPOCHS, BATCH = 200, 256
D_MODEL, N_HEADS, FF_DIM = 64, 4, 128


def met(yt, yp):
    return (float(mean_absolute_error(yt, yp)),
            float(np.sqrt(mean_squared_error(yt, yp))),
            float(r2_score(yt, yp)))


# ────────────────────────── data per resolution ──────────────────────────────
def load_base():
    df = pd.read_csv(DATA, usecols=['timestamp', 'year', TARGET,
                                    'T_ambient_mean', 'GHI_mean', 'POA_mean',
                                    'Wind_speed_mean'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna(subset=[TARGET, 'T_ambient_mean', 'GHI_mean', 'POA_mean',
                           'Wind_speed_mean']).sort_values('timestamp')
    return df.set_index('timestamp')


def resample_to(df5, dt_min):
    """Gộp 5-min -> dt_min bằng mean; chỉ nhận bin đủ count = dt/5 mẫu con."""
    if dt_min == 5:
        out = df5.copy()
    else:
        need = dt_min // 5
        g = df5.resample(f'{dt_min}min', closed='right', label='right')
        out = g.mean()
        cnt = g[TARGET].count()
        out = out[cnt == need]
    out = out.dropna()
    # Faiman tính lại từ input đã gộp (nhất quán với cách file 5-min được tạo)
    out['T_Faiman'] = out['T_ambient_mean'] + out['POA_mean'] / (
        U0 + U1 * out['Wind_speed_mean'].clip(lower=0.1))
    out['hour'] = out.index.hour + out.index.minute / 60.0
    out['day_of_year'] = out.index.dayofyear
    out['year'] = out.index.year
    return out.reset_index()


def add_ghi_lag(df, dt_min):
    """GHI_lag ≈ 30 phút; gap-aware: chỉ nhận lag khi timestamp cách đúng k·Δt."""
    k = max(1, round(LAG_TARGET_MIN / dt_min))
    lag_ok = (df['timestamp'].diff(k).dt.total_seconds() == k * dt_min * 60)
    df = df.copy()
    df['GHI_lag'] = df['GHI_mean'].shift(k).where(lag_ok)
    return df.dropna(subset=['GHI_lag']), k * dt_min


def make_windows(X, y, ts, dt_min, window, horizon):
    """Cửa sổ gap-aware: `window` bước liên tục đúng Δt, target cách horizon bước.
    horizon=0 -> nowcast (target = bước cuối cửa sổ); 1 -> forecast t+1."""
    step = np.timedelta64(dt_min * 60, 's')
    n = len(X)
    span = window - 1 + horizon
    if n <= span:
        return (np.empty((0, window, X.shape[1]), np.float32),
                np.empty(0, np.float32), np.empty(0, 'datetime64[ns]'))
    # hợp lệ nếu mọi bước trong [i, i+span] cách nhau đúng Δt
    diffs = np.diff(ts) == step
    ok = np.array([diffs[i:i + span].all() for i in range(n - span)])
    idx = np.flatnonzero(ok)
    if len(idx) == 0:      # đủ dài nhưng mọi cửa sổ đều vắt gap -> rỗng an toàn
        return (np.empty((0, window, X.shape[1]), np.float32),
                np.empty(0, np.float32), np.empty(0, 'datetime64[ns]'))
    Xw = np.stack([X[i:i + window] for i in idx]).astype(np.float32)
    yw = y[idx + span].astype(np.float32)
    return Xw, yw, ts[idx + span]


# ─────────────────────────────── models ──────────────────────────────────────
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='relu'),
                                     layers.Dense(d_model)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        a = self.drop1(self.att(inputs, inputs), training=training)
        o1 = self.norm1(inputs + a)
        f = self.drop2(self.ffn(o1), training=training)
        return self.norm2(o1 + f)


def build_pg_resmlp(n_feat):
    m = keras.Sequential([layers.Input(shape=(n_feat,)),
                          layers.Dense(32, activation='relu'),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return m


def build_lstm(w, n):
    m = keras.Sequential([layers.Input(shape=(w, n)),
                          layers.LSTM(64, dropout=0.1, recurrent_dropout=0.1),
                          layers.Dense(32, activation='relu'),
                          layers.Dropout(0.1), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return m


def build_hybrid(w, n):
    inp = layers.Input(shape=(w, n))
    x = layers.Dense(D_MODEL)(inp)
    pos = tf.range(start=0, limit=w, delta=1)
    x = x + layers.Embedding(input_dim=w, output_dim=D_MODEL)(pos)
    x = TransformerBlock(D_MODEL, N_HEADS, FF_DIM, 0.1)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    m = keras.Model(inp, layers.Dense(1)(x))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return m


def get_callbacks():
    return [callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=10, min_lr=1e-6)]


# ─────────────────────────────── runner ──────────────────────────────────────
SMOKE = False   # set trong main(); smoke KHÔNG được ghi vào CSV chính thức


def append_row(row):
    if SMOKE:
        return
    hdr = not os.path.exists(CSV_OUT)
    pd.DataFrame([row]).to_csv(CSV_OUT, mode='a', header=hdr, index=False)


def adaptive_batch(n_fit):
    """Batch scale theo cỡ dữ liệu: đảm bảo >=60 bước cập nhật/epoch (floor 32).
    Ở 5/10/15-min giữ nguyên 256 (protocol MEAFD); ở 30/60-min giảm xuống
    để mọi model có ngân sách tối ưu tương đương — tránh nhầm 'thiếu thông tin
    thời gian' với 'đói gradient update' khi so sánh giữa các độ phân giải."""
    return int(min(BATCH, max(32, n_fit // 60)))


def train_eval(build_fn, Xtr, ytr, Xte, yte, seed, epochs):
    """build_fn: callable không tham số trả về model MỚI.
    QUAN TRỌNG (audit 2026-08-15): model PHẢI được build SAU khi seed —
    keras.utils.set_random_seed reset cả Python/NumPy/Keras seed generator,
    đảm bảo init tất định theo seed, độc lập lịch sử RNG (resume == fresh).
    Bug cũ: build trước seed -> init lấy RNG trôi nổi, vi phạm reproducibility."""
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_fn()
    # validation = 10% cuối của train theo thứ tự thời gian
    n_val = max(1, int(0.1 * len(Xtr)))
    batch = adaptive_batch(len(Xtr) - n_val)
    t0 = time.time()
    h = model.fit(Xtr[:-n_val], ytr[:-n_val],
                  validation_data=(Xtr[-n_val:], ytr[-n_val:]),
                  epochs=epochs, batch_size=batch,
                  callbacks=get_callbacks(), verbose=0)
    p = model.predict(Xte, batch_size=4096, verbose=0).flatten()
    return p, time.time() - t0, len(h.history['loss']), batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--window-sweep', action='store_true',
                    help='phụ: sensitivity window LSTM @5min forecast')
    args = ap.parse_args()

    global SMOKE
    SMOKE = args.smoke
    resolutions = [30] if args.smoke else RESOLUTIONS
    seeds = [0] if args.smoke else list(range(args.seeds))
    epochs = 3 if args.smoke else EPOCHS
    print(f'TF {tf.__version__} | GPU: {tf.config.list_physical_devices("GPU")}')
    print(f'resolutions={resolutions} seeds={seeds[:3]}...n={len(seeds)} '
          f'epochs={epochs}', flush=True)

    base = load_base()
    preds = {}
    if os.path.exists(NPZ_OUT) and not args.smoke:
        preds = dict(np.load(NPZ_OUT, allow_pickle=True))
    done = set()
    if os.path.exists(CSV_OUT) and not args.smoke:
        old = pd.read_csv(CSV_OUT)
        done = {(r.resolution, r.mode, r.model, r.seed)
                for r in old.itertuples()}
        print(f'resume: {len(done)} runs đã có trong CSV', flush=True)

    window_grid = [(WINDOW, res, mode) for res in resolutions
                   for mode in ('nowcast', 'forecast')]
    if args.window_sweep:
        window_grid = [(w, 5, 'forecast') for w in (3, 12, 24)]

    for window, res, mode in window_grid:
        df = resample_to(base, res)
        df, lag_actual = add_ghi_lag(df, res)
        tr = df[df['year'] == 2022]
        te = df[df['year'] == 2023]
        ts_tr = tr['timestamp'].values
        ts_te = te['timestamp'].values
        horizon = 0 if mode == 'nowcast' else 1

        # ---- point-wise (Faiman + PG-ResMLP): X(t) -> y(t+horizon), gap-aware
        step = np.timedelta64(res * 60, 's')
        if horizon == 0:
            tr_idx = np.arange(len(tr)); te_idx = np.arange(len(te))
        else:
            tr_idx = np.flatnonzero(np.diff(ts_tr) == step)
            te_idx = np.flatnonzero(np.diff(ts_te) == step)
        sc_p = StandardScaler()
        Xp_tr = sc_p.fit_transform(tr[PHYS_FEATURES].values)[tr_idx]
        Xp_te = sc_p.transform(te[PHYS_FEATURES].values)[te_idx]
        yp_tr = tr[TARGET].values[tr_idx + horizon].astype(np.float32)
        yp_te = te[TARGET].values[te_idx + horizon].astype(np.float32)
        ghi_p = te['GHI_mean'].values[te_idx + horizon]
        # PG-ResMLP là mô hình RESIDUAL (JST Eq. 1): train trên Δ = y − T_Faiman
        # với T_Faiman tại thời điểm feature (t) — không rò rỉ thời tiết t+1
        fai_tr = tr['T_Faiman'].values[tr_idx].astype(np.float32)
        fai_te = te['T_Faiman'].values[te_idx].astype(np.float32)
        rp_tr = yp_tr - fai_tr
        rp_te = yp_te - fai_te

        # ---- sequence: window 6 bước, gap-aware
        sc_s = StandardScaler()
        Xs_tr_flat = sc_s.fit_transform(tr[RAW_FEATURES].values)
        Xs_te_flat = sc_s.transform(te[RAW_FEATURES].values)
        Xw_tr, yw_tr, _ = make_windows(Xs_tr_flat, tr[TARGET].values,
                                       ts_tr, res, window, horizon)
        Xw_te, yw_te, tsw = make_windows(Xs_te_flat, te[TARGET].values,
                                         ts_te, res, window, horizon)
        ghi_s = te.set_index('timestamp')['GHI_mean'].reindex(tsw).values

        tag = f'{res}min_{mode}' + (f'_w{window}' if window != WINDOW else '')
        print(f'\n=== Δt={res}min {mode} window={window} '
              f'(lag={lag_actual}min) | point {len(Xp_tr)}/{len(Xp_te)} '
              f'| seq {len(Xw_tr)}/{len(Xw_te)} ===', flush=True)

        # Faiman reference (nowcast only, deterministic)
        if mode == 'nowcast' and window == WINDOW:
            fmae, frmse, fr2 = met(yp_te, te['T_Faiman'].values[te_idx])
            if (res, mode, 'Faiman', -1) not in done:
                append_row(dict(resolution=res, mode=mode, model='Faiman',
                                seed=-1, window=0, lag_min=0,
                                n_train=len(yp_tr), n_test=len(yp_te),
                                MAE=fmae, RMSE=frmse, R2=fr2,
                                epochs=0, batch=0, train_s=0.0))
            print(f'  Faiman: MAE={fmae:.3f}', flush=True)

        # Hybrid chỉ chạy forecast mode (phạm vi MEAFD); LSTM trả lời câu hỏi
        # memory ở cả 2 mode; PG-ResMLP point-wise ở cả 2 mode
        jobs = []
        if window == WINDOW:
            jobs.append(('PG-ResMLP', 'point'))
        jobs.append(('LSTM', 'seq'))
        if mode == 'forecast':
            jobs.append(('Hybrid', 'seq'))
        for name, kind in jobs:
            for seed in seeds:
                if (res, mode, name, seed) in done:
                    continue
                if kind == 'point':
                    p_res, dt, ep, bs = train_eval(
                        lambda: build_pg_resmlp(len(PHYS_FEATURES)),
                        Xp_tr, rp_tr, Xp_te, rp_te, seed, epochs)
                    p = fai_te + p_res          # T_pred = T_Faiman + Δ_neural
                    yt, ghi = yp_te, ghi_p
                else:
                    builder = build_lstm if name == 'LSTM' else build_hybrid
                    p, dt, ep, bs = train_eval(
                        lambda: builder(window, len(RAW_FEATURES)),
                        Xw_tr, yw_tr, Xw_te, yw_te, seed, epochs)
                    yt, ghi = yw_te, ghi_s
                mae, rmse, r2 = met(yt, p)
                append_row(dict(resolution=res, mode=mode, model=name,
                                seed=seed, window=(0 if kind == 'point'
                                                   else window),
                                lag_min=(lag_actual if kind == 'point' else 0),
                                n_train=(len(yp_tr) if kind == 'point'
                                         else len(yw_tr)),
                                n_test=len(yt), MAE=mae, RMSE=rmse, R2=r2,
                                epochs=ep, batch=bs, train_s=round(dt, 1)))
                preds[f'{tag}_{name}_s{seed}'] = p.astype(np.float32)
                print(f'  {name:10} seed {seed:2d}: MAE={mae:.4f} '
                      f'R2={r2:.4f} ({dt:.0f}s, {ep} ep)', flush=True)
            # ground truth + GHI lưu một lần cho mỗi (tag, kind)
            preds[f'{tag}_ytrue_{kind}'] = (yp_te if kind == 'point'
                                            else yw_te).astype(np.float32)
            preds[f'{tag}_ghi_{kind}'] = np.asarray(
                ghi_p if kind == 'point' else ghi_s, np.float32)
            if not args.smoke:
                np.savez_compressed(NPZ_OUT, **preds)

    if args.smoke:
        print('\nSMOKE OK — pipeline end-to-end chạy được.')
    else:
        np.savez_compressed(NPZ_OUT, **preds)
        print(f'\nDONE. Kết quả: {CSV_OUT}\nPredictions: {NPZ_OUT}')


if __name__ == '__main__':
    main()
