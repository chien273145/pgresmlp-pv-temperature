"""
E13c — Wind-Availability Disambiguation Ablation (PVDAQ)
=========================================================
Bối cảnh: E13b phát hiện SIGN FLIP — PG-ResMLP thắng LSTM ở mọi ô DuraMAT,
ngược với PVDAQ. Hai giải thích cạnh tranh:
  (H1) LSTM mất input gió trên DuraMAT (không đo) -> thua vì thiếu feature
  (H2) DuraMAT nhỏ/nhiễu -> physics-anchor sample efficiency thắng

Thí nghiệm: trên PVDAQ (dữ liệu LỚN, có gió đo), chạy lại 2 cấu hình handicap:
  - LSTM-noWS      : raw features BỎ Wind_speed (5f) — mô phỏng handicap DuraMAT
  - PG-ResMLP-ws2  : T_Faiman tính với WS=2.0 hằng số, features
                     [T_Faiman_ws2, T_ambient, GHI_lag] (3f, 673 params)
                     — mô phỏng MLP wind-blind của DuraMAT (không có RH thay thế
                     vì PVDAQ 7333 không đo RH; ghi rõ trong Methods)

Suy luận:
  - LSTM-noWS vẫn thắng PG-ResMLP(full, từ e13) ở 5/15-min -> H1 bị loại phần lớn
  - LSTM-noWS thắng PG-ResMLP-ws2 (cấu hình DuraMAT, dữ liệu lớn) nhưng thua
    trên DuraMAT (e13b) -> flip do data volume/quality -> H2 được củng cố

Grid: Δt ∈ {5, 15} × {nowcast, forecast} × 2 models × 20 seeds = 160 runs.
Đối chứng full-feature: đọc từ results/e13_multires_sweep.csv (cùng protocol,
cùng seeds, cùng splits -> paired được từng seed).

Chạy:  python e13c_ws_ablation.py [--smoke] [--seeds 20]
Output: results/e13c_ws_ablation.csv
"""
import io, os, sys, argparse, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from e13_multires_sweep import (met, make_windows, build_lstm, train_eval,
                                load_base, resample_to, add_ghi_lag, U0, U1)

RESULT_DIR = r'c:\Users\admin\NCKH\results'
CSV_OUT = os.path.join(RESULT_DIR, 'e13c_ws_ablation.csv')

RAW_NOWS = ['T_ambient_mean', 'GHI_mean', 'POA_mean', 'hour', 'day_of_year']
PHYS_WS2 = ['T_Faiman_ws2', 'T_ambient_mean', 'GHI_lag']
TARGET = 'T_module_mean'
WINDOW = 6
RESOLUTIONS = [5, 15]          # nơi LSTM từng thắng có ý nghĩa trên PVDAQ
WS_CONST = 2.0


def build_mlp_3f():
    m = keras.Sequential([layers.Input(shape=(3,)),
                          layers.Dense(32, activation='relu'),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return m


def append_row(row, smoke):
    if smoke:
        return
    hdr = not os.path.exists(CSV_OUT)
    pd.DataFrame([row]).to_csv(CSV_OUT, mode='a', header=hdr, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seeds', type=int, default=20)
    args = ap.parse_args()
    resolutions = [15] if args.smoke else RESOLUTIONS
    seeds = [0] if args.smoke else list(range(args.seeds))
    epochs = 3 if args.smoke else 200

    import tensorflow as tf
    print(f'TF {tf.__version__} | res={resolutions} n_seeds={len(seeds)}',
          flush=True)

    done = set()
    if os.path.exists(CSV_OUT) and not args.smoke:
        old = pd.read_csv(CSV_OUT)
        done = {(r.resolution, r.mode, r.model, r.seed)
                for r in old.itertuples()}
        print(f'resume: {len(done)} runs đã có', flush=True)

    base = load_base()
    for res in resolutions:
        df = resample_to(base, res)
        df, lag_actual = add_ghi_lag(df, res)
        # T_Faiman wind-blind (WS=2.0 hằng số) — mô phỏng cấu hình DuraMAT
        df['T_Faiman_ws2'] = (df['T_ambient_mean']
                              + df['POA_mean'] / (U0 + U1 * WS_CONST))
        tr = df[df['year'] == 2022]
        te = df[df['year'] == 2023]
        ts_tr = tr['timestamp'].values
        ts_te = te['timestamp'].values
        for mode in ('nowcast', 'forecast'):
            horizon = 0 if mode == 'nowcast' else 1
            step = np.timedelta64(res * 60, 's')
            if horizon == 0:
                tr_idx = np.arange(len(tr)); te_idx = np.arange(len(te))
            else:
                tr_idx = np.flatnonzero(np.diff(ts_tr) == step)
                te_idx = np.flatnonzero(np.diff(ts_te) == step)

            # ---- PG-ResMLP wind-blind (3f, residual vs T_Faiman_ws2)
            sc_p = StandardScaler()
            Xp_tr = sc_p.fit_transform(tr[PHYS_WS2].values)[tr_idx]
            Xp_te = sc_p.transform(te[PHYS_WS2].values)[te_idx]
            yp_tr = tr[TARGET].values[tr_idx + horizon].astype(np.float32)
            yp_te = te[TARGET].values[te_idx + horizon].astype(np.float32)
            fai_tr = tr['T_Faiman_ws2'].values[tr_idx].astype(np.float32)
            fai_te = te['T_Faiman_ws2'].values[te_idx].astype(np.float32)
            rp_tr, rp_te = yp_tr - fai_tr, yp_te - fai_te

            # ---- LSTM-noWS (5 raw features)
            sc_s = StandardScaler()
            Xs_tr = sc_s.fit_transform(tr[RAW_NOWS].values)
            Xs_te = sc_s.transform(te[RAW_NOWS].values)
            Xw_tr, yw_tr, _ = make_windows(Xs_tr, tr[TARGET].values,
                                           ts_tr, res, WINDOW, horizon)
            Xw_te, yw_te, _ = make_windows(Xs_te, te[TARGET].values,
                                           ts_te, res, WINDOW, horizon)

            print(f'\n=== {res}min {mode} (lag={lag_actual}min) | point '
                  f'{len(Xp_tr)}/{len(Xp_te)} | seq {len(Xw_tr)}/{len(Xw_te)}'
                  f' ===', flush=True)
            faimae = met(yp_te, fai_te)[0]
            print(f'  Faiman(WS=2.0) ref: MAE={faimae:.3f}', flush=True)

            for seed in seeds:
                if (res, mode, 'PG-ResMLP-ws2', seed) not in done:
                    p_res, dt, ep, bs = train_eval(build_mlp_3f, Xp_tr, rp_tr,
                                                   Xp_te, rp_te, seed, epochs)
                    mae, rmse, r2 = met(yp_te, fai_te + p_res)
                    append_row(dict(resolution=res, mode=mode,
                                    model='PG-ResMLP-ws2', seed=seed,
                                    n_train=len(yp_tr), n_test=len(yp_te),
                                    MAE=mae, RMSE=rmse, R2=r2, epochs=ep,
                                    batch=bs, train_s=round(dt, 1)), args.smoke)
                    print(f'  PG-ResMLP-ws2 seed {seed:2d}: MAE={mae:.4f} '
                          f'({dt:.0f}s)', flush=True)
                if (res, mode, 'LSTM-noWS', seed) not in done:
                    p, dt, ep, bs = train_eval(
                        lambda: build_lstm(WINDOW, len(RAW_NOWS)),
                        Xw_tr, yw_tr, Xw_te, yw_te, seed, epochs)
                    mae, rmse, r2 = met(yw_te, p)
                    append_row(dict(resolution=res, mode=mode,
                                    model='LSTM-noWS', seed=seed,
                                    n_train=len(yw_tr), n_test=len(yw_te),
                                    MAE=mae, RMSE=rmse, R2=r2, epochs=ep,
                                    batch=bs, train_s=round(dt, 1)), args.smoke)
                    print(f'  LSTM-noWS     seed {seed:2d}: MAE={mae:.4f} '
                          f'({dt:.0f}s)', flush=True)

    print('\nDONE.' if not args.smoke else '\nSMOKE OK.')


if __name__ == '__main__':
    main()
