"""
Phân tích E13c — phán quyết H1 (thiếu gió) vs H2 (data volume/quality)
=======================================================================
Input : results/e13c_ws_ablation.csv  (LSTM-noWS, PG-ResMLP-ws2; PVDAQ 5/15-min)
        results/e13_multires_sweep.csv (đối chứng full-feature, cùng seeds)
Output: results/e13c_stats.csv + verdict in console

4 so sánh paired theo seed, Holm trong từng family (4 cells res×mode):
  C1: LSTM-noWS  vs LSTM-full        — chi phí mất gió của LSTM
  C2: MLP-ws2    vs MLP-full         — chi phí mất gió của PG-ResMLP
  C3: LSTM-noWS  vs MLP-full         — Arm 1: LSTM mất gió còn thắng MLP đủ gió?
  C4: LSTM-noWS  vs MLP-ws2          — Arm 2: cấu hình DuraMAT trên data lớn
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats as st

R = r'c:\Users\admin\NCKH\results'
CELLS = [(5, 'nowcast'), (5, 'forecast'), (15, 'nowcast'), (15, 'forecast')]


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def series(df, res, mode, model):
    s = df[(df.resolution == res) & (df['mode'] == mode) & (df.model == model)]
    return s.set_index('seed')['MAE']


def main():
    abl = pd.read_csv(os.path.join(R, 'e13c_ws_ablation.csv'))
    full = pd.read_csv(os.path.join(R, 'e13_multires_sweep.csv'))

    comps = [('C1_LSTM_wind_cost', 'LSTM-noWS', abl, 'LSTM', full),
             ('C2_MLP_wind_cost', 'PG-ResMLP-ws2', abl, 'PG-ResMLP', full),
             ('C3_arm1_noWS_vs_fullMLP', 'LSTM-noWS', abl, 'PG-ResMLP', full),
             ('C4_arm2_duramat_config', 'LSTM-noWS', abl, 'PG-ResMLP-ws2', abl)]
    rows = []
    for fam, m1, d1, m2, d2 in comps:
        fam_rows = []
        for res, mode in CELLS:
            a = series(d1, res, mode, m1)
            b = series(d2, res, mode, m2)
            common = a.index.intersection(b.index)
            if len(common) < 5:
                print(f'  [skip] {fam} {res}min {mode}: {len(common)} seeds')
                continue
            d = a[common].values - b[common].values
            t, p = st.ttest_rel(a[common].values, b[common].values)
            try:
                _, pw = st.wilcoxon(a[common].values, b[common].values)
            except ValueError:
                pw = np.nan
            fam_rows.append(dict(family=fam, resolution=res, mode=mode,
                                 n=len(common), mean_1=a[common].mean(),
                                 mean_2=b[common].mean(), diff=d.mean(),
                                 t=t, p_raw=p, p_wilcoxon=pw,
                                 cohen_d=d.mean() / d.std(ddof=1)))
        if fam_rows:
            adj = holm(np.array([r['p_raw'] for r in fam_rows]))
            for r, pa in zip(fam_rows, adj):
                r['p_holm'] = pa
                r['significant'] = 'YES' if pa < 0.05 else 'no'
            rows += fam_rows

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, 'e13c_stats.csv'), index=False)
    pd.set_option('display.width', 200)
    print(out.round(4).to_string(index=False))

    # ─── Verdict tự động ────────────────────────────────────────────────────
    print('\n=== VERDICT (diff = model1 − model2; âm = model1 tốt hơn) ===')
    c3 = out[out.family == 'C3_arm1_noWS_vs_fullMLP']
    c4 = out[out.family == 'C4_arm2_duramat_config']
    arm1_lstm_wins = (c3['diff'] < 0) & (c3.significant == 'YES')
    arm2_lstm_wins = (c4['diff'] < 0) & (c4.significant == 'YES')
    print(f'  Arm1 (LSTM-noWS vs MLP-full): LSTM thắng sig. '
          f'{int(arm1_lstm_wins.sum())}/{len(c3)} cells')
    print(f'  Arm2 (DuraMAT-config, data lớn): LSTM thắng sig. '
          f'{int(arm2_lstm_wins.sum())}/{len(c4)} cells')
    if arm2_lstm_wins.all() and len(c4):
        print('  → H2 ĐƯỢC CỦNG CỐ: cùng cấu hình wind-blind, LSTM thắng trên '
              'data lớn (PVDAQ) nhưng thua trên site nhỏ (DuraMAT/e13b) — '
              'sign flip do lượng/chất dữ liệu, không phải thiếu gió.')
    elif (~arm2_lstm_wins).all() and len(c4):
        print('  → H1 ĐƯỢC CỦNG CỐ: mất gió là đủ để LSTM thua ngay cả trên '
              'data lớn — sign flip chủ yếu do feature availability.')
    else:
        print('  → Kết quả HỖN HỢP theo cell — báo cáo từng cell, không claim '
              'nguyên nhân đơn lẻ.')


if __name__ == '__main__':
    main()
