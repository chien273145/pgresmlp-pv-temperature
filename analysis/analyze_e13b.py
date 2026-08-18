"""
Phân tích E13b — Cross-site consistency của resolution law
===========================================================
Câu hỏi: dấu của gap (MLP − LSTM) theo Δt có tái hiện trên 4 site không?
  - PVDAQ CA  (từ results/e13_multires_sweep.csv, lấy 3 tier 15/30/60)
  - DuraMAT CO/FL/OR (results/e13b_duramat_coarse.csv)

Output:
  results/e13b_stats.csv                  — paired tests per (site, res, mode)
  figures/Fig4_cross_site_gap.pdf|.png    — gap ± CI, zero line, 4 site

Chạy được trên dữ liệu PARTIAL (bỏ qua cell chưa đủ seeds, in cảnh báo) —
số cuối cùng chỉ lấy khi e13b chạy đủ n=20.
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = r'c:\Users\admin\NCKH\results'
FIG_DIR = r'c:\Users\admin\NCKH\figures'
RES = [15, 30, 60]
MIN_SEEDS = 5          # cell cần >= 5 seeds chung mới đưa vào phân tích
# Tiêu chí inclusion (khớp guard của e13b): cell sequence dưới ngưỡng bị loại
# khỏi so sánh gap (vd CO 60-min: seq test chỉ 15-24 điểm -> vô nghĩa thống kê)
MIN_SEQ_TRAIN, MIN_SEQ_TEST = 200, 50

# Site identity — 4 màu Okabe-Ito (validate lại nếu đổi)
SITE_C = {'CA': '#0072B2', 'CO': '#D55E00', 'FL': '#009E73', 'OR': '#CC79A7'}
SITE_M = {'CA': 'o', 'CO': 's', 'FL': '^', 'OR': 'D'}


def load():
    frames = []
    p13 = os.path.join(R, 'e13_multires_sweep.csv')
    if os.path.exists(p13):
        a = pd.read_csv(p13)
        a = a[a.resolution.isin(RES)].copy()
        a['site'] = 'CA'
        frames.append(a[['site', 'resolution', 'mode', 'model', 'seed', 'MAE',
                         'n_train', 'n_test']])
    p13b = os.path.join(R, 'e13b_duramat_coarse.csv')
    if os.path.exists(p13b):
        b = pd.read_csv(p13b)
        frames.append(b[['site', 'resolution', 'mode', 'model', 'seed', 'MAE',
                         'n_train', 'n_test']])
    return pd.concat(frames, ignore_index=True)


def paired_gap(df, site, res, mode):
    """Gap = MAE(PG-ResMLP) − MAE(LSTM), paired theo seed. None nếu thiếu."""
    cell = df[(df.site == site) & (df.resolution == res) & (df['mode'] == mode)]
    lstm_rows = cell[cell.model == 'LSTM']
    if len(lstm_rows) and (lstm_rows.n_train.iloc[0] < MIN_SEQ_TRAIN
                           or lstm_rows.n_test.iloc[0] < MIN_SEQ_TEST):
        print(f'  [excluded] {site} {res}min {mode}: seq n_train/'
              f'n_test = {lstm_rows.n_train.iloc[0]}/{lstm_rows.n_test.iloc[0]}'
              f' < {MIN_SEQ_TRAIN}/{MIN_SEQ_TEST} (inclusion criteria)')
        return None
    a = cell[cell.model == 'PG-ResMLP'].set_index('seed')['MAE']
    b = lstm_rows.set_index('seed')['MAE']
    common = a.index.intersection(b.index)
    if len(common) < MIN_SEEDS:
        return None
    d = (a[common] - b[common]).values      # dương = LSTM tốt hơn
    t, p = st.ttest_rel(a[common].values, b[common].values)
    try:
        _, pw = st.wilcoxon(a[common].values, b[common].values)
    except ValueError:
        pw = np.nan
    return dict(site=site, resolution=res, mode=mode, n=len(common),
                gap_mean=d.mean(), gap_std=d.std(ddof=1),
                ci95=st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)),
                t=t, p_raw=p, p_wilcoxon=pw,
                mlp_mean=a[common].mean(), lstm_mean=b[common].mean())


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    df = load()
    sites = [s for s in ('CA', 'CO', 'FL', 'OR') if s in df.site.unique()]
    rows = []
    for mode in ('nowcast', 'forecast'):
        fam = []
        for site in sites:
            for res in RES:
                r = paired_gap(df, site, res, mode)
                if r is None:
                    print(f'  [skip] {site} {res}min {mode}: chưa đủ '
                          f'{MIN_SEEDS} seeds chung')
                    continue
                fam.append(r)
        if fam:
            adj = holm(np.array([r['p_raw'] for r in fam]))
            for r, pa in zip(fam, adj):
                r['p_holm'] = pa
                r['significant'] = 'YES' if pa < 0.05 else 'no'
            rows += fam

    if not rows:
        print('Chưa có cell nào đủ dữ liệu.')
        return
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, 'e13b_stats.csv'), index=False)
    pd.set_option('display.width', 200)
    print(out.round(4).to_string(index=False))

    # ── Sign-consistency summary ─────────────────────────────────────────────
    print('\n=== Sign consistency (gap>0: LSTM tốt hơn | gap<0: MLP tốt hơn) ===')
    for mode in ('nowcast', 'forecast'):
        for res in RES:
            cells = out[(out['mode'] == mode) & (out.resolution == res)]
            if len(cells) == 0:
                continue
            signs = ['+' if g > 0 else '−' for g in cells.gap_mean]
            sigs = ['*' if s == 'YES' else '' for s in cells.significant]
            desc = '  '.join(f'{si}:{sg}{sg2}' for si, sg, sg2
                             in zip(cells.site, signs, sigs))
            print(f'  {mode:9} {res:2d}min: {desc}')

    # ── Figure: gap ± CI, small multiples theo mode ──────────────────────────
    from fig_style import apply_journal_style, save_journal, DOUBLE_COL
    apply_journal_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), sharey=True)
    for ax, mode, title in [(axes[0], 'nowcast', '(a) Nowcasting'),
                            (axes[1], 'forecast',
                             '(b) One-step forecasting (horizon $=\\Delta t$)')]:
        ax.axhline(0, color='0.4', lw=0.8, ls='-', zorder=1)
        for site in sites:
            sub = out[(out['mode'] == mode) & (out.site == site)]
            if len(sub) == 0:
                continue
            sub = sub.sort_values('resolution')
            ax.errorbar(sub.resolution, sub.gap_mean, yerr=sub.ci95,
                        color=SITE_C[site], marker=SITE_M[site], ms=4.5,
                        lw=1.5, capsize=2.5, capthick=0.8, elinewidth=0.9,
                        label=site, zorder=3)
        ax.set_xscale('log')
        ax.set_xticks(RES)
        ax.set_xticklabels([str(r) for r in RES])
        ax.minorticks_off()
        ax.set_xlabel('Sampling interval $\\Delta t$ (min)')
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('MAE gap: PG-ResMLP $-$ LSTM (°C)\n'
                       '$>0$: LSTM better')
    axes[0].legend(frameon=False, fontsize=7.5, title='Site', title_fontsize=7.5)
    fig.tight_layout()
    save_journal(fig, 'Fig4_cross_site_gap', FIG_DIR)


if __name__ == '__main__':
    main()
