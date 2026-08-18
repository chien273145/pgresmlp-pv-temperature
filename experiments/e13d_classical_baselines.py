"""
E13d — Classical baselines dưới UNIFIED PROTOCOL (Table 3, bài Q1)
===================================================================
LR / Ridge / RF / XGBoost trên đúng pipeline e13 (5-min nowcast + forecast,
4 physics features, temporal split, gap-aware lag) để Table 3 tự chứa —
không mượn số từ protocol JST cũ (random-val).

Residual formulation thống nhất với PG-ResMLP: fit trên (y − T_Faiman),
prediction = T_Faiman + Δ̂. Deterministic models (LR/Ridge) chạy 1 lần;
RF/XGBoost chạy 20 seeds.

Chạy:  python e13d_classical_baselines.py
Output: results/e13d_classical_baselines.csv
"""
import io, os, sys, time, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, r'c:\Users\admin\NCKH')
from e13_multires_sweep import load_base, resample_to, add_ghi_lag, PHYS_FEATURES

CSV_OUT = r'c:\Users\admin\NCKH\results\e13d_classical_baselines.csv'
TARGET = 'T_module_mean'
SEEDS = list(range(20))


def met(yt, yp):
    return (float(mean_absolute_error(yt, yp)),
            float(np.sqrt(mean_squared_error(yt, yp))),
            float(r2_score(yt, yp)))


def main():
    base = load_base()
    df = resample_to(base, 5)
    df, _ = add_ghi_lag(df, 5)
    tr = df[df['year'] == 2022]
    te = df[df['year'] == 2023]
    rows = []
    for mode, horizon in [('nowcast', 0), ('forecast', 1)]:
        ts_tr = tr['timestamp'].values
        ts_te = te['timestamp'].values
        step = np.timedelta64(300, 's')
        if horizon == 0:
            tr_idx = np.arange(len(tr)); te_idx = np.arange(len(te))
        else:
            tr_idx = np.flatnonzero(np.diff(ts_tr) == step)
            te_idx = np.flatnonzero(np.diff(ts_te) == step)
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[PHYS_FEATURES].values)[tr_idx]
        Xte = sc.transform(te[PHYS_FEATURES].values)[te_idx]
        ytr = tr[TARGET].values[tr_idx + horizon]
        yte = te[TARGET].values[te_idx + horizon]
        fai_tr = tr['T_Faiman'].values[tr_idx]
        fai_te = te['T_Faiman'].values[te_idx]
        rtr = ytr - fai_tr

        print(f'=== 5-min {mode}: {len(Xtr)}/{len(Xte)} ===', flush=True)
        fmae, frmse, fr2 = met(yte, fai_te)
        rows.append(dict(mode=mode, model='Faiman', seed=-1,
                         MAE=fmae, RMSE=frmse, R2=fr2, train_s=0.0))
        print(f'  Faiman: {fmae:.4f}', flush=True)

        for name, mk in [('LinearReg', lambda s: LinearRegression()),
                         ('Ridge', lambda s: Ridge(alpha=1.0))]:
            t0 = time.time()
            m = mk(0).fit(Xtr, rtr)
            mae, rmse, r2 = met(yte, fai_te + m.predict(Xte))
            rows.append(dict(mode=mode, model=name, seed=-1, MAE=mae,
                             RMSE=rmse, R2=r2, train_s=round(time.time()-t0, 1)))
            print(f'  {name}: {mae:.4f}', flush=True)

        makers = [('RandomForest', lambda s: RandomForestRegressor(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=s))]
        try:
            from xgboost import XGBRegressor
            makers.append(('XGBoost', lambda s: XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=s, verbosity=0)))
        except ImportError:
            print('  [skip] xgboost không có', flush=True)
        for name, mk in makers:
            for seed in SEEDS:
                t0 = time.time()
                m = mk(seed).fit(Xtr, rtr)
                mae, rmse, r2 = met(yte, fai_te + m.predict(Xte))
                rows.append(dict(mode=mode, model=name, seed=seed, MAE=mae,
                                 RMSE=rmse, R2=r2,
                                 train_s=round(time.time()-t0, 1)))
            sub = [r for r in rows if r['model'] == name and r['mode'] == mode]
            mm = np.mean([r['MAE'] for r in sub])
            ss = np.std([r['MAE'] for r in sub], ddof=1)
            print(f'  {name}: {mm:.4f} ± {ss:.4f} (n={len(sub)})', flush=True)

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f'\nsaved {CSV_OUT}')


if __name__ == '__main__':
    main()
