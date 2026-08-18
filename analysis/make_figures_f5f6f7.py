"""
Sinh 3 figure còn thiếu cho bài Q1 (F5 conformal, F6 Hanoi field, F7 Pareto).
Nguồn số: results/e14_conformal_uq.csv (post-audit), processed/hanoi_real_4day_
with_wind.csv, results/e13*/e13d CSVs. Palette Okabe-Ito (validator PASS).
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

R = r'c:\Users\admin\NCKH\results'
FIG = r'c:\Users\admin\NCKH\figures'
from fig_style import (apply_journal_style, save_journal, SINGLE_COL,
                       DOUBLE_COL, ORANGE as C_ORANGE, BLUE as C_BLUE,
                       GREEN as C_GREEN, PURPLE as C_PURPLE, GRAY as C_GRAY)
apply_journal_style()
save = save_journal


# ═══ F5 — Conformal coverage & width per GHI band ═════════════════════════════
e14 = pd.read_csv(os.path.join(R, 'e14_conformal_uq.csv'))
g = e14.groupby(['alpha', 'method', 'band'])[['coverage', 'mean_width']].mean()
bands = ['[0,200)', '[200,500)', '[500,inf)']
blabels = ['0–200', '200–500', '≥500']
x = np.arange(3)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9))
ax = axes[0]
for alpha, color, marker, lab in [(0.10, C_BLUE, 'o', '90% nominal'),
                                  (0.05, C_ORANGE, 's', '95% nominal')]:
    cov = [g.loc[(alpha, 'mondrian', b), 'coverage'] for b in bands]
    ax.plot(x, cov, marker=marker, ms=6, lw=1.6, color=color,
            label=f'Mondrian ({lab.split()[0]})', zorder=3)
    ax.axhline(1 - alpha, color=color, ls=':', lw=1.0, zorder=1)
ax.set_xticks(x); ax.set_xticklabels(blabels)
ax.set_xlabel('GHI band (W m$^{-2}$)')
ax.set_ylabel('Empirical coverage (2023 test)')
ax.set_ylim(0.85, 1.005)
ax.set_title('(a) Coverage vs. nominal (dotted)', fontsize=8.5)
ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc='lower left')

ax = axes[1]
w = 0.35
for i, (alpha, color, lab) in enumerate([(0.10, C_BLUE, '90%'),
                                         (0.05, C_ORANGE, '95%')]):
    hw = [g.loc[(alpha, 'mondrian', b), 'mean_width'] / 2 for b in bands]
    ax.bar(x + (i - 0.5) * w, hw, w * 0.92, color=color, label=lab, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(blabels)
ax.set_xlabel('GHI band (W m$^{-2}$)')
ax.set_ylabel('Interval half-width ±(°C)')
ax.set_title('(b) Adaptive interval width per band', fontsize=8.5)
ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, title='Confidence')
fig.tight_layout()
save(fig, 'Fig5_conformal')

# ═══ F6 — Hanoi field PoC time series ═════════════════════════════════════════
hn = pd.read_csv(r'c:\Users\admin\NCKH\processed\hanoi_real_4day_with_wind.csv')
hn['datetime'] = pd.to_datetime(hn['datetime'])
hn = hn.sort_values('datetime')
# resample 5-min cho đồ thị đỡ dày (1-min gốc)
hn5 = hn.set_index('datetime')[['T_module', 'T_Faiman', 'T_ambient',
                                'GHI']].resample('5min').mean().dropna()

fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 3.7), sharex=True,
                         height_ratios=[2.1, 1])
ax = axes[0]
test_start = pd.Timestamp('2026-04-20')
ax.axvspan(test_start, hn5.index.max(), color='0.93', zorder=0)
ax.plot(hn5.index, hn5['T_module'], color=C_ORANGE, lw=1.3,
        label='Measured $T_{module}$', zorder=3)
ax.plot(hn5.index, hn5['T_Faiman'], color=C_GRAY, lw=1.0, ls='--',
        label='Faiman baseline', zorder=2)
ax.plot(hn5.index, hn5['T_ambient'], color=C_BLUE, lw=1.0, ls=':',
        label='$T_{ambient}$', zorder=2)
ax.set_ylabel('Temperature (°C)')
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='upper left')
ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.annotate('held-out\ntest day', xy=(test_start + pd.Timedelta(hours=7), 37.5),
            ha='center', fontsize=7, color='0.35')
ax.annotate('PG-ResMLP (4-day fine-tune): MAE = 0.296 ± 0.003 °C\n'
            'Faiman US: 0.541 °C | Linear: 0.444 °C  (day-5 test, n=5 seeds)',
            xy=(0.01, 0.02), xycoords='axes fraction', fontsize=7,
            color='0.25')
ax.set_title('(a) Rooftop deployment, Hanoi — 94 h continuous (overcast '
             'period)', fontsize=8.5)

ax = axes[1]
ax.axvspan(test_start, hn5.index.max(), color='0.93', zorder=0)
ax.plot(hn5.index, hn5['GHI'], color=C_GREEN, lw=1.0, zorder=2)
ax.set_ylabel('GHI (W m$^{-2}$)')
ax.set_title('(b) Irradiance (BH1750 lux-derived, ±15%)', fontsize=8.5)
ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
fig.tight_layout()
save(fig, 'Fig6_hanoi_field')

# ═══ F7 — Pareto: accuracy vs deployable size (5-min nowcast) ═════════════════
e13 = pd.read_csv(os.path.join(R, 'e13_multires_sweep.csv'))
e13d = pd.read_csv(os.path.join(R, 'e13d_classical_baselines.csv'))


def mmean(df, model, mode='nowcast', res=5):
    m = df[(df.model == model) & (df['mode'] == mode)]
    if 'resolution' in df.columns:
        m = m[m.resolution == res]
    return m['MAE'].mean()


# (size_bytes, exact?) — Faiman 3 float; LR 5 float; PG-ResMLP 705×4 = 2820;
# LSTM 20289×4 = 81156; XGB/RF: serialized ước lượng (caption ghi rõ "≈")
models = [
    ('Faiman', 12, mmean(e13d, 'Faiman'), C_GRAY, 'x', True),
    ('Linear', 20, mmean(e13d, 'LinearReg'), C_GRAY, '+', True),
    ('XGBoost', 150e3, mmean(e13d, 'XGBoost'), C_GREEN, '^', False),
    ('Random Forest', 5e6, mmean(e13d, 'RandomForest'), C_PURPLE, 'D', False),
    ('PG-ResMLP', 2820, mmean(e13, 'PG-ResMLP'), C_ORANGE, 'o', True),
    ('LSTM-64', 81156, mmean(e13, 'LSTM'), C_BLUE, 's', True),
]
fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
def size_label(size):
    if size >= 1e6:
        return f'≈{size/1e6:.0f} MB'
    if size >= 1e3:
        return f'{size/1e3:.3g} kB'
    return f'{size:.0f} B'


for name, size, mae, color, marker, exact in models:
    ax.scatter(size, mae, s=45, color=color, marker=marker, zorder=3)
    lab = size_label(size) if exact else '≈' + size_label(size).lstrip('≈')
    off, ha = {'Faiman': ((9, -8), 'left'), 'Linear': ((8, 4), 'left'),
               'PG-ResMLP': ((0, 9), 'center'),
               'LSTM-64': ((-8, -6), 'right'),
               'XGBoost': ((-6, 8), 'right'),
               'Random Forest': ((-2, -20), 'right')}[name]
    ax.annotate(f'{name}\n{lab}', xy=(size, mae), xytext=off,
                textcoords='offset points', fontsize=7, color='0.2', ha=ha)
# vùng khả thi MCU (<=100 kB flash cho model)
ax.axvspan(1, 100e3, color='0.95', zorder=0)
ax.annotate('MCU-feasible\n(≤100 kB)', xy=(40, 2.30), fontsize=7,
            color='0.4')
ax.set_xlim(4, 2e7)
ax.set_xscale('log')
ax.set_xlabel('Deployable model size (bytes, log)')
ax.set_ylabel('Test MAE (°C), 5-min nowcast')
ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
save(fig, 'Fig7_pareto')
print('DONE')
