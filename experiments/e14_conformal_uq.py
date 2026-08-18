"""
E14 — Split Conformal Prediction Intervals cho PG-ResMLP (PVDAQ 7333)
======================================================================
Mục tiêu (bài Q1/Q2, mục UQ):
  Khoảng dự báo có bảo đảm thống kê (distribution-free, finite-sample) cho
  T_module, tính được on-device: q̂ per-band chỉ là bảng tra 3 số float.

Phương pháp: Split Conformal + Mondrian theo dải GHI (cùng band với JST Table 7:
  [0,200), [200,500), [500,∞) W/m²).
  - 2022 chia theo thứ tự thời gian: 70% train / 10% val (early stopping) /
    20% calibration (tách biệt hoàn toàn khỏi mọi bước fit/model-selection).
  - Nonconformity score: |y − ŷ|; q̂_b = quantile bậc ⌈(n_b+1)(1−α)⌉/n_b
    (method='higher') của score trong band b trên calibration set.
  - Đánh giá coverage thực nghiệm + độ rộng interval trên test 2023.
  - So sánh Mondrian vs Global (một q̂ chung) để cho thấy tính thích nghi.

Model: PG-ResMLP [32,16] residual (T_pred = T_Faiman + Δ_neural), 4 features,
5-min nowcast (chế độ triển khai), protocol train như JST/E13.

Chạy:  python e14_conformal_uq.py [--seeds 5] [--smoke]
Output: results/e14_conformal_uq.csv + summary in ra console
"""
import io, os, sys, time, argparse, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

DATA = r'c:\Users\admin\NCKH\processed\pvdaq_7333_v2_2022_2023.csv'
CSV_OUT = r'c:\Users\admin\NCKH\results\e14_conformal_uq.csv'

FEATURES = ['T_Faiman', 'T_ambient_mean', 'Wind_speed_mean', 'GHI_lag_30min']
TARGET = 'T_module_mean'
BANDS = [(0, 200), (200, 500), (500, np.inf)]      # GHI W/m² (JST Table 7)
ALPHAS = [0.10, 0.05]                              # 90% và 95%
EPOCHS, BATCH = 200, 256


def build_mlp(n_feat):
    m = keras.Sequential([layers.Input(shape=(n_feat,)),
                          layers.Dense(32, activation='relu'),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return m


def conformal_qhat(scores, alpha):
    """q̂ = phần tử nhỏ thứ k, k = ⌈(n+1)(1−α)⌉ (canonical split conformal).
    Audit 2026-08-15: bản cũ chuyển rank->level rồi np.quantile(method='higher')
    bị overshoot ~1 bậc (bảo thủ hơn canonical); nay lấy đúng k-th smallest.
    k > n (calibration quá nhỏ) -> +inf theo định nghĩa."""
    n = len(scores)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def band_of(ghi):
    idx = np.zeros(len(ghi), dtype=int)
    for i, (lo, hi) in enumerate(BANDS):
        idx[(ghi >= lo) & (ghi < hi)] = i
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    seeds = [0] if args.smoke else list(range(args.seeds))
    epochs = 3 if args.smoke else EPOCHS

    df = pd.read_csv(DATA)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna(subset=FEATURES + [TARGET, 'GHI_mean']).sort_values('timestamp')
    tr = df[df['year'] == 2022].reset_index(drop=True)
    te = df[df['year'] == 2023].reset_index(drop=True)

    n = len(tr)
    i_tr, i_val = int(0.70 * n), int(0.80 * n)
    fit_df, val_df, cal_df = tr.iloc[:i_tr], tr.iloc[i_tr:i_val], tr.iloc[i_val:]
    print(f'2022: fit {len(fit_df)} | val {len(val_df)} | cal {len(cal_df)} '
          f'|| test 2023: {len(te)}', flush=True)

    sc = StandardScaler().fit(fit_df[FEATURES].values)
    X = {k: sc.transform(d[FEATURES].values)
         for k, d in [('fit', fit_df), ('val', val_df),
                      ('cal', cal_df), ('te', te)]}
    fai = {k: d['T_Faiman'].values.astype(np.float32)
           for k, d in [('fit', fit_df), ('val', val_df),
                        ('cal', cal_df), ('te', te)]}
    y = {k: d[TARGET].values.astype(np.float32)
         for k, d in [('fit', fit_df), ('val', val_df),
                      ('cal', cal_df), ('te', te)]}
    band_cal = band_of(cal_df['GHI_mean'].values)
    band_te = band_of(te['GHI_mean'].values)

    rows = []
    for seed in seeds:
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(seed)   # audit fix: full deterministic init
        model = build_mlp(len(FEATURES))
        t0 = time.time()
        model.fit(X['fit'], y['fit'] - fai['fit'],
                  validation_data=(X['val'], y['val'] - fai['val']),
                  epochs=epochs, batch_size=BATCH, verbose=0,
                  callbacks=[callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=20,
                                                     restore_best_weights=True),
                             callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.5, patience=10,
                                                         min_lr=1e-6)])
        p_cal = fai['cal'] + model.predict(X['cal'], batch_size=4096,
                                           verbose=0).flatten()
        p_te = fai['te'] + model.predict(X['te'], batch_size=4096,
                                         verbose=0).flatten()
        mae_te = mean_absolute_error(y['te'], p_te)
        s_cal = np.abs(y['cal'] - p_cal)
        s_te = np.abs(y['te'] - p_te)
        print(f'seed {seed}: test MAE={mae_te:.3f} ({time.time()-t0:.0f}s)',
              flush=True)

        for alpha in ALPHAS:
            # Global conformal
            qg = conformal_qhat(s_cal, alpha)
            rows.append(dict(seed=seed, alpha=alpha, method='global',
                             band='all', n_cal=len(s_cal), n_test=len(s_te),
                             qhat=qg, coverage=float((s_te <= qg).mean()),
                             mean_width=2 * qg, test_MAE=float(mae_te)))
            # Mondrian per-band
            cov_all, w_all = [], []
            for b, (lo, hi) in enumerate(BANDS):
                sb_cal = s_cal[band_cal == b]
                sb_te = s_te[band_te == b]
                qb = conformal_qhat(sb_cal, alpha)
                cov = float((sb_te <= qb).mean()) if len(sb_te) else np.nan
                rows.append(dict(seed=seed, alpha=alpha, method='mondrian',
                                 band=f'[{lo},{hi})', n_cal=len(sb_cal),
                                 n_test=len(sb_te), qhat=qb, coverage=cov,
                                 mean_width=2 * qb, test_MAE=float(mae_te)))
                cov_all.append((s_te[band_te == b] <= qb))
                w_all.append(np.full(len(sb_te), 2 * qb))
            rows.append(dict(seed=seed, alpha=alpha, method='mondrian',
                             band='all', n_cal=len(s_cal), n_test=len(s_te),
                             qhat=np.nan,
                             coverage=float(np.concatenate(cov_all).mean()),
                             mean_width=float(np.concatenate(w_all).mean()),
                             test_MAE=float(mae_te)))

    out = pd.DataFrame(rows)
    if not args.smoke:
        out.to_csv(CSV_OUT, index=False)
    print('\n=== SUMMARY (mean over seeds) ===')
    g = (out.groupby(['alpha', 'method', 'band'])
            [['coverage', 'mean_width', 'qhat']].mean().round(4))
    print(g.to_string())
    if not args.smoke:
        print(f'\nsaved {CSV_OUT}')
    print('\nGhi chú on-device: bảng tra q̂ Mondrian = 3 float/α — chi phí RAM'
          ' không đáng kể, interval = ŷ ± q̂[band(GHI)].')


if __name__ == '__main__':
    main()
