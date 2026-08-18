"""F1 (system pipeline) + F2 (PG-ResMLP architecture) — vector schematics,
journal style (fig_style). Thay thế bản PNG cũ của JST."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from fig_style import (apply_journal_style, save_journal, SINGLE_COL,
                       DOUBLE_COL, ORANGE, BLUE, GREEN, GRAY)
apply_journal_style()

FILL = dict(sensor='#EAF2F8', phys='#FDF2E9', model='#FDEBD0',
            out='#EAFAF1', app='#F4F6F6')
EDGE = '0.45'


def box(ax, x, y, w, h, text, fill, fs=7, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                fc=fill, ec=EDGE, lw=0.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, weight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, style='-|>', color='0.35'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, lw=0.9, color=color,
                                 zorder=1))


# ═══ F1 — pipeline ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis('off')

# sensors
box(ax, 1, 24, 13, 6, 'T$_{ambient}$', FILL['sensor'])
box(ax, 1, 16, 13, 6, 'GHI / POA', FILL['sensor'])
box(ax, 1, 8, 13, 6, 'Wind speed\n(or const.)', FILL['sensor'])
ax.text(7.5, 32, 'Weather inputs', ha='center', fontsize=7.5, color='0.3')

# feature block
box(ax, 21, 17.5, 20, 10,
    'Faiman physics\n$T_F = T_a + \\frac{POA}{U_0 + U_1 WS}$',
    FILL['phys'], fs=7)
box(ax, 21, 8, 20, 6, 'GHI ring buffer\n(30-min lag, 24 B)', FILL['phys'])
for ys, ye in [(27, 24.5), (19, 22), (11, 20)]:
    arrow(ax, 14, ys, 21, ye)
arrow(ax, 14, 19, 21, 11)

# model
box(ax, 48, 12, 17, 12, 'PG-ResMLP\n$\\Delta_\\theta$: 4$\\to$32$\\to$16$\\to$1\n'
    '705 params, 2.8 kB', FILL['model'], fs=7, weight='bold')
arrow(ax, 41, 22.5, 48, 20)
arrow(ax, 41, 11, 48, 15)

# residual sum + conformal
box(ax, 70, 17, 12, 8, '$\\hat{T} = T_F + \\hat{\\Delta}$\n'
    '$\\pm\\; \\hat{q}[\\mathrm{band}]$', FILL['out'], fs=7)
arrow(ax, 65, 18, 70, 20)
arrow(ax, 31, 27.5, 76, 27.5)          # T_F skip connection
arrow(ax, 76, 27.5, 76, 25)
ax.text(53, 29, 'physics skip connection', fontsize=6.5, color='0.4',
        ha='center')

# applications
box(ax, 87, 24, 12, 5.5, 'TCPR / PR audit', FILL['app'], fs=6.5)
box(ax, 87, 17, 12, 5.5, 'Thermal alarms', FILL['app'], fs=6.5)
box(ax, 87, 10, 12, 5.5, 'SCADA backfill', FILL['app'], fs=6.5)
for ye in (26.5, 20, 13):
    arrow(ax, 82, 21, 87, ye)

# deployment strip
ax.add_patch(FancyBboxPatch((1, 0.6), 98, 4.6,
                            boxstyle='round,pad=0.012', fc='0.96',
                            ec=EDGE, lw=0.7))
ax.text(50, 2.9, 'ESP32 (Xtensa LX6, 240 MHz) — pure C float32: '
        '31 μs/inference, ≈5–7 μJ, 350 kB SRAM free, '
        'on-device ≡ desktop (±0.06 m°C)', ha='center', fontsize=6.8,
        color='0.25')
save_journal(fig, 'Fig1_pipeline')

# ═══ F2 — architecture ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.4))
ax.set_xlim(0, 60); ax.set_ylim(0, 92); ax.axis('off')

feats = ['$T_{Faiman}$', '$T_{ambient}$', 'WS (or RH)', '$GHI_{lag30}$']
for i, f in enumerate(feats):
    box(ax, 2 + i * 14.5, 82, 13, 7, f, FILL['sensor'], fs=6.8)
    arrow(ax, 8.5 + i * 14.5, 82, 30 - (1.5 - i) * 4, 75)

box(ax, 13, 68, 34, 7, 'StandardScaler (in firmware)', FILL['phys'])
arrow(ax, 30, 68, 30, 63)
box(ax, 13, 51, 34, 12, 'Dense 32, ReLU\n4×32 + 32 = 160 params',
    FILL['model'])
arrow(ax, 30, 51, 30, 46)
box(ax, 13, 34, 34, 12, 'Dense 16, ReLU\n32×16 + 16 = 528 params',
    FILL['model'])
arrow(ax, 30, 34, 30, 29)
box(ax, 13, 22, 34, 7, 'Dense 1 (linear): $\\hat{\\Delta}$ — 17 params',
    FILL['model'])
arrow(ax, 30, 22, 30, 15.5)

box(ax, 13, 7, 34, 8.5, '$\\hat{T}_{module} = T_{Faiman} + \\hat{\\Delta}$',
    FILL['out'], fs=7.5, weight='bold')
# residual skip: từ feature T_Faiman vòng bên trái xuống sum
arrow(ax, 2.5, 82, 2.5, 11)
arrow(ax, 2.5, 11, 13, 11)
ax.text(4.5, 45, 'residual (physics) skip', fontsize=6.3, color='0.4',
        rotation=90, va='center')
ax.text(30, 1.5, 'Total: 705 parameters = 2,820 B float32',
        ha='center', fontsize=7, color='0.25')
save_journal(fig, 'Fig2_architecture')
print('DONE')
