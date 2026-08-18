"""
Phân tích E13 — thống kê + figure đinh cho bài Q1/Q2
=====================================================
Input : results/e13_multires_sweep.csv  (chạy lại được khi nâng n seeds)
Output: results/e13_stats.csv           (paired tests, Holm–Bonferroni)
        figures/Fig3_resolution_sweep.pdf|.png  (2 panel: nowcast | forecast)

Stats:
  - Paired theo seed (cùng seed = cùng khởi tạo/chuỗi ngẫu nhiên đã kiểm soát).
  - 3 family so sánh, Holm–Bonferroni trong từng family (5 test/family):
      F1: nowcast  LSTM vs PG-ResMLP  (câu hỏi memory-value)
      F2: forecast LSTM vs PG-ResMLP
      F3: forecast Hybrid vs LSTM     (giá trị attention)
  - Kèm Wilcoxon signed-rank (robust với outlier seeds) + Cohen's d (paired).
"""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r'c:\Users\admin\NCKH\results\e13_multires_sweep.csv'
STATS_OUT = r'c:\Users\admin\NCKH\results\e13_stats.csv'
FIG_DIR = r'c:\Users\admin\NCKH\figures'
os.makedirs(FIG_DIR, exist_ok=True)

RES = [5, 10, 15, 30, 60]
C = {'PG-ResMLP': '#D55E00', 'LSTM': '#0072B2', 'Hybrid': '#009E73',
     'Faiman': '#6e6e6e'}          # Okabe-Ito subset, validator PASS
M = {'PG-ResMLP': 'o', 'LSTM': 's', 'Hybrid': '^'}

df = pd.read_csv(CSV)
n_seeds = df[df.model != 'Faiman'].groupby(
    ['resolution', 'mode', 'model'])['seed'].count().min()
print(f'n seeds (min per cell) = {n_seeds}')


def paired(a, b):
    """a, b: MAE theo seed (đã align). Trả về dict test paired a vs b."""
    d = a - b
    t, p = st.ttest_rel(a, b)
    try:
        w, pw = st.wilcoxon(a, b)
    except ValueError:
        pw = np.nan
    cd = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    return dict(mean_diff=d.mean(), t=t, p_raw=p, p_wilcoxon=pw, cohen_d=cd)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def get(res, mode, model):
    s = df[(df.resolution == res) & (df['mode'] == mode) & (df.model == model)]
    return s.sort_values('seed')['MAE'].values


rows = []
for fam, mode, m1, m2 in [('F1_nowcast_LSTM_vs_MLP', 'nowcast', 'LSTM', 'PG-ResMLP'),
                          ('F2_forecast_LSTM_vs_MLP', 'forecast', 'LSTM', 'PG-ResMLP'),
                          ('F3_forecast_Hybrid_vs_LSTM', 'forecast', 'Hybrid', 'LSTM')]:
    fam_rows = []
    for res in RES:
        a, b = get(res, mode, m1), get(res, mode, m2)
        r = paired(a, b)
        r.update(family=fam, resolution=res, comparison=f'{m1} - {m2}',
                 mean_1=a.mean(), mean_2=b.mean(), n=len(a))
        fam_rows.append(r)
    p_adj = holm(np.array([r['p_raw'] for r in fam_rows]))
    for r, pa in zip(fam_rows, p_adj):
        r['p_holm'] = pa
        r['significant'] = 'YES' if pa < 0.05 else 'no'
    rows += fam_rows

out = pd.DataFrame(rows)[['family', 'resolution', 'comparison', 'n',
                          'mean_1', 'mean_2', 'mean_diff', 't', 'p_raw',
                          'p_holm', 'p_wilcoxon', 'cohen_d', 'significant']]
out.to_csv(STATS_OUT, index=False)
pd.set_option('display.width', 200)
print(out.round(4).to_string(index=False))

# ───────────────────────────── figure ─────────────────────────────────────────
from fig_style import apply_journal_style, save_journal, DOUBLE_COL
apply_journal_style()
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.1), sharey=True)

for ax, mode, models, title in [
        (axes[0], 'nowcast', ['PG-ResMLP', 'LSTM'],
         '(a) Nowcasting — estimate $T(t)$ from inputs at $t$'),
        (axes[1], 'forecast', ['PG-ResMLP', 'LSTM', 'Hybrid'],
         '(b) One-step forecasting — horizon $=\\Delta t$')]:
    # vùng hằng số thời gian nhiệt τ ≈ 7–12 min
    ax.axvspan(7, 12, color='0.92', zorder=0)
    if mode == 'nowcast':
        fai = [df[(df.resolution == r) & (df['mode'] == 'nowcast') &
                  (df.model == 'Faiman')]['MAE'].iloc[0] for r in RES]
        ax.plot(RES, fai, ls=':', color=C['Faiman'], lw=1.4, marker='x',
                ms=5, label='Faiman (physics)', zorder=2)
    for mo in models:
        mu = np.array([get(r, mode, mo).mean() for r in RES])
        se = np.array([get(r, mode, mo).std(ddof=1) / np.sqrt(len(get(r, mode, mo)))
                       for r in RES])
        tcrit = st.t.ppf(0.975, df=len(get(RES[0], mode, mo)) - 1)
        ax.plot(RES, mu, color=C[mo], marker=M[mo], ms=4.5, lw=1.6,
                label=mo, zorder=3)
        ax.fill_between(RES, mu - tcrit * se, mu + tcrit * se,
                        color=C[mo], alpha=0.18, lw=0, zorder=1)
    ax.set_xscale('log')
    ax.set_xticks(RES)
    ax.set_xticklabels([str(r) for r in RES])
    ax.minorticks_off()
    ax.set_xlabel('Sampling interval $\\Delta t$ (min)')
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5,
              loc='upper left' if mode == 'forecast' else 'center left')

axes[0].set_ylabel('Test MAE (°C), 2023 hold-out')
axes[0].annotate('$\\tau \\approx$ 7–12 min', xy=(9.2, 3.08),
                 ha='center', fontsize=7, color='0.35')
# crossover annotation (panel a): LSTM thắng có ý nghĩa ở 5–15 min, hoà ở 30,
# đảo chiều có ý nghĩa ở 60 (n=20, Holm) — giao điểm mean nằm giữa 30–60 min
axes[0].annotate('crossover\n30–60 min', xy=(42, 1.44), ha='center',
                 fontsize=7, color='0.35',
                 xytext=(42, 1.85),
                 arrowprops=dict(arrowstyle='-', color='0.6', lw=0.7))

fig.tight_layout()
save_journal(fig, 'Fig3_resolution_sweep', FIG_DIR)

# ───────────────────── câu phát biểu định lượng cho draft ─────────────────────
print('\n=== LSTM gain (MLP − LSTM, °C; dương = LSTM tốt hơn) ===')
for mode in ('nowcast', 'forecast'):
    gains = {r: get(r, mode, 'PG-ResMLP').mean() - get(r, mode, 'LSTM').mean()
             for r in RES}
    print(f'  {mode:9}: ' + '  '.join(f'{r}min:{g:+.3f}' for r, g in gains.items()))
