"""
E13b — Coarse-end Replication trên DuraMAT (3 site, Δt ∈ {15, 30, 60} min)
===========================================================================
Mục đích: kiểm tra ranh giới "giá trị của temporal memory" tìm thấy trên PVDAQ
(e13: LSTM thắng ≤15 min, hoà 30, đảo chiều 60 — nowcast) có tái hiện trên
3 khí hậu khác (CO semi-arid, FL subtropical, OR oceanic) hay không.
DuraMAT/PERT gốc 15-min → chỉ lặp được đầu thô của sweep (15/30/60).

Protocol: THỐNG NHẤT với e13 (import trực tiếp models + windowing + adaptive
batch từ e13_multires_sweep). Khác biệt do dữ liệu, ghi rõ trong Methods:
  - WS không đo → Faiman dùng WS = 2.0 hằng số (JST §4); WS bị LOẠI khỏi
    feature (hằng số vô nghĩa), thay bằng RH:
      point: [T_Faiman, T_ambient, RH, GHI_lag]            (705 params)
      seq  : [T_ambient, GHI, POA, RH, hour, day_of_year]  (6 features)
  - GHI_lag theo quy tắc e13: ~30 min = max(1, round(30/Δt)) bước
    (KHÁC JST DuraMAT dùng shift=6 = 90 min — đây là protocol e13, không phải
    tái tạo bảng JST).
  - Split: chronological 80/20 mỗi site (CLAUDE.md), val = 10% cuối của train.
  - Module: đúng file JST dùng — Golden_mSi0247, Cocoa_mSi0166, Eugene_mSi0166.
  - Timestamp floor về phút (PERT lệch giây :31) trước gap-check.

Chạy:
  python e13b_duramat_coarse.py --smoke      # CO, 30-min, 1 seed, 3 epochs
  python e13b_duramat_coarse.py --seeds 20   # full (~4h CPU), resume được

Output: results/e13b_duramat_coarse.csv (+ e13b_predictions.npz)
"""
import io, os, sys, time, argparse, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from e13_multires_sweep import (met, make_windows, build_pg_resmlp,
                                build_lstm, build_hybrid, train_eval)

BASE = r'c:\Users\admin\NCKH\data_duramat\pert\Data For Validating Models'
RESULT_DIR = r'c:\Users\admin\NCKH\results'
CSV_OUT = os.path.join(RESULT_DIR, 'e13b_duramat_coarse.csv')
NPZ_OUT = os.path.join(RESULT_DIR, 'e13b_predictions.npz')

SITES = {'CO': ('Golden', 'Golden_mSi0247.csv'),
         'FL': ('Cocoa', 'Cocoa_mSi0166.csv'),
         'OR': ('Eugene', 'Eugene_mSi0166.csv')}
COL_MAP = {0: 'timestamp', 1: 'POA', 3: 'T_module', 20: 'T_ambient',
           22: 'RH', 30: 'GHI'}
RESOLUTIONS = [60, 30, 15]                 # thô -> mịn (rẻ ra trước)
FULL_RESOLUTIONS = [60, 30, 15, 10, 5]     # --full: site 5-min gốc chạy cả dải
# Cadence gốc KHÁC NHAU giữa site (phát hiện 2026-08-15, xác minh từ timestamp):
#   Golden/CO = 15-min; Cocoa/FL và Eugene/OR = 5-min.
# -> native cadence tự phát hiện từ median diff; bin coarse cần đủ dt/native mẫu.
MIN_POINT_TRAIN, MIN_POINT_TEST = 500, 50
MIN_SEQ_TRAIN, MIN_SEQ_TEST = 200, 50
LAG_TARGET_MIN = 30
U0, U1, WS_CONST = 25.0, 6.84, 2.0
WINDOW = 6

POINT_FEATURES = ['T_Faiman', 'T_ambient', 'RH', 'GHI_lag']
SEQ_FEATURES = ['T_ambient', 'GHI', 'POA', 'RH', 'hour', 'day_of_year']
TARGET = 'T_module'

SMOKE = False


def read_pert(fp):
    rows = []
    with open(fp, 'r') as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 42:
                rows.append({n: p[i] for i, n in COL_MAP.items()})
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for c in df.columns:
        if c != 'timestamp':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace(-9999.0, np.nan).dropna(subset=['timestamp'])
    # PERT timestamp lệch giây (vd :31) -> floor về phút cho gap-check chính xác
    df['timestamp'] = df['timestamp'].dt.floor('min')
    df = df.dropna(subset=['POA', 'T_module', 'T_ambient', 'RH', 'GHI'])
    df = df[df['POA'] > 50]                 # daytime (quy ước DuraMAT của dự án)
    return df.sort_values('timestamp').set_index('timestamp')


def detect_native_min(df):
    d = pd.Series(df.index).diff().dt.total_seconds().dropna()
    return int(d.mode().iloc[0] // 60)


def resample_to(df_native, dt_min, native_min):
    if dt_min == native_min:
        out = df_native.copy()
    else:
        assert dt_min % native_min == 0, f'{dt_min} not multiple of {native_min}'
        need = dt_min // native_min
        g = df_native.resample(f'{dt_min}min', closed='right', label='right')
        out = g.mean()
        out = out[g[TARGET].count() == need]
    out = out.dropna()
    out['T_Faiman'] = out['T_ambient'] + out['POA'] / (U0 + U1 * WS_CONST)
    out['hour'] = out.index.hour + out.index.minute / 60.0
    out['day_of_year'] = out.index.dayofyear
    return out.reset_index()


def add_ghi_lag(df, dt_min):
    k = max(1, round(LAG_TARGET_MIN / dt_min))
    lag_ok = (df['timestamp'].diff(k).dt.total_seconds() == k * dt_min * 60)
    df = df.copy()
    df['GHI_lag'] = df['GHI'].shift(k).where(lag_ok)
    return df.dropna(subset=['GHI_lag']), k * dt_min


def append_row(row):
    if SMOKE:
        return
    hdr = not os.path.exists(CSV_OUT)
    pd.DataFrame([row]).to_csv(CSV_OUT, mode='a', header=hdr, index=False)


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--full', action='store_true',
                    help='site 5-min gốc (FL/OR) chạy cả dải 5/10/15/30/60')
    args = ap.parse_args()
    SMOKE = args.smoke

    sites = ['CO'] if args.smoke else list(SITES)
    resolutions = ([30] if args.smoke else
                   (FULL_RESOLUTIONS if args.full else RESOLUTIONS))
    seeds = [0] if args.smoke else list(range(args.seeds))
    epochs = 3 if args.smoke else 200

    import tensorflow as tf
    print(f'TF {tf.__version__} | sites={sites} res={resolutions} '
          f'n_seeds={len(seeds)} epochs={epochs}', flush=True)

    preds = {}
    if os.path.exists(NPZ_OUT) and not args.smoke:
        preds = dict(np.load(NPZ_OUT, allow_pickle=True))
    done = set()
    if os.path.exists(CSV_OUT) and not args.smoke:
        old = pd.read_csv(CSV_OUT)
        done = {(r.site, r.resolution, r.mode, r.model, r.seed)
                for r in old.itertuples()}
        print(f'resume: {len(done)} runs đã có', flush=True)

    for site in sites:
        sub, fname = SITES[site]
        base = read_pert(os.path.join(BASE, sub, fname))
        native = detect_native_min(base)
        print(f'\n##### SITE {site} ({fname}): {len(base)} rows, native='
              f'{native}-min [{base.index.min()} .. {base.index.max()}]',
              flush=True)
        for res in resolutions:
            if res < native or res % native != 0:
                print(f'  [skip] {res}min: không tương thích native '
                      f'{native}min', flush=True)
                continue
            df = resample_to(base, res, native)
            df, lag_actual = add_ghi_lag(df, res)
            n = len(df)
            i80 = int(0.8 * n)
            tr, te = df.iloc[:i80], df.iloc[i80:]
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
                sc_p = StandardScaler()
                Xp_tr = sc_p.fit_transform(tr[POINT_FEATURES].values)[tr_idx]
                Xp_te = sc_p.transform(te[POINT_FEATURES].values)[te_idx]
                yp_tr = tr[TARGET].values[tr_idx + horizon].astype(np.float32)
                yp_te = te[TARGET].values[te_idx + horizon].astype(np.float32)
                fai_tr = tr['T_Faiman'].values[tr_idx].astype(np.float32)
                fai_te = te['T_Faiman'].values[te_idx].astype(np.float32)
                rp_tr, rp_te = yp_tr - fai_tr, yp_te - fai_te

                sc_s = StandardScaler()
                Xs_tr = sc_s.fit_transform(tr[SEQ_FEATURES].values)
                Xs_te = sc_s.transform(te[SEQ_FEATURES].values)
                Xw_tr, yw_tr, _ = make_windows(Xs_tr, tr[TARGET].values,
                                               ts_tr, res, WINDOW, horizon)
                Xw_te, yw_te, tsw = make_windows(Xs_te, te[TARGET].values,
                                                 ts_te, res, WINDOW, horizon)

                tag = f'{site}_{res}min_{mode}'
                print(f'\n=== {tag} (lag={lag_actual}min) | point '
                      f'{len(Xp_tr)}/{len(Xp_te)} | seq {len(Xw_tr)}/'
                      f'{len(Xw_te)} ===', flush=True)
                # Guard: cell quá mỏng -> skip (tránh crash predict rỗng
                # và tránh ghi kết quả vô nghĩa vào CSV)
                if len(Xp_tr) < MIN_POINT_TRAIN or len(Xp_te) < MIN_POINT_TEST:
                    print(f'  [SKIP CELL] point data quá mỏng '
                          f'(<{MIN_POINT_TRAIN}/{MIN_POINT_TEST})', flush=True)
                    continue
                seq_ok = (len(Xw_tr) >= MIN_SEQ_TRAIN
                          and len(Xw_te) >= MIN_SEQ_TEST)
                if not seq_ok:
                    print(f'  [SKIP SEQ] sequence data quá mỏng '
                          f'(<{MIN_SEQ_TRAIN}/{MIN_SEQ_TEST}) — chỉ chạy '
                          f'point models', flush=True)

                if mode == 'nowcast':
                    fmae, frmse, fr2 = met(yp_te, fai_te)
                    if (site, res, mode, 'Faiman', -1) not in done:
                        append_row(dict(site=site, resolution=res, mode=mode,
                                        model='Faiman', seed=-1,
                                        n_train=len(yp_tr), n_test=len(yp_te),
                                        MAE=fmae, RMSE=frmse, R2=fr2,
                                        epochs=0, batch=0, train_s=0.0))
                    print(f'  Faiman(WS=2.0): MAE={fmae:.3f}', flush=True)

                jobs = [('PG-ResMLP', 'point')]
                if seq_ok:
                    jobs.append(('LSTM', 'seq'))
                    if mode == 'forecast':
                        jobs.append(('Hybrid', 'seq'))
                for name, kind in jobs:
                    for seed in seeds:
                        if (site, res, mode, name, seed) in done:
                            continue
                        if kind == 'point':
                            p_res, dt, ep, bs = train_eval(
                                lambda: build_pg_resmlp(len(POINT_FEATURES)),
                                Xp_tr, rp_tr, Xp_te, rp_te, seed, epochs)
                            p, yt = fai_te + p_res, yp_te
                        else:
                            builder = (build_lstm if name == 'LSTM'
                                       else build_hybrid)
                            p, dt, ep, bs = train_eval(
                                lambda: builder(WINDOW, len(SEQ_FEATURES)),
                                Xw_tr, yw_tr, Xw_te, yw_te, seed, epochs)
                            yt = yw_te
                        mae, rmse, r2 = met(yt, p)
                        append_row(dict(site=site, resolution=res, mode=mode,
                                        model=name, seed=seed,
                                        n_train=(len(yp_tr) if kind == 'point'
                                                 else len(yw_tr)),
                                        n_test=len(yt), MAE=mae, RMSE=rmse,
                                        R2=r2, epochs=ep, batch=bs,
                                        train_s=round(dt, 1)))
                        preds[f'{tag}_{name}_s{seed}'] = p.astype(np.float32)
                        print(f'  {name:10} seed {seed:2d}: MAE={mae:.4f} '
                              f'R2={r2:.4f} ({dt:.0f}s, {ep} ep)', flush=True)
                    preds[f'{tag}_ytrue_{kind}'] = (
                        yp_te if kind == 'point' else yw_te).astype(np.float32)
                if not args.smoke:
                    np.savez_compressed(NPZ_OUT, **preds)

    if args.smoke:
        print('\nSMOKE OK.')
    else:
        np.savez_compressed(NPZ_OUT, **preds)
        print(f'\nDONE. {CSV_OUT}')


if __name__ == '__main__':
    main()
