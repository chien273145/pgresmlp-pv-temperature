"""
Journal figure style — chuẩn Elsevier (Applied Energy / Solar Energy).
Dùng chung cho MỌI figure của bài Q1: import và gọi apply_journal_style().

Chuẩn áp dụng:
  - Vector PDF với font TrueType nhúng (pdf.fonttype=42 — Type-3 bị nhiều
    nhà xuất bản từ chối), PNG 600 dpi cho bản review.
  - Arial (chuẩn Elsevier; fallback Helvetica/DejaVu Sans), cỡ 7–9 pt
    tại kích thước in cuối.
  - Bề rộng đúng cột: single 90 mm (3.54"), 1.5-col 140 mm (5.51"),
    double 190 mm (7.48").
  - Đường ≥0.5 pt; không bold tiêu đề; minor ticks tắt; spines top/right ẩn
    (từng axes tự xử lý).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Bề rộng cột (inch) — Elsevier artwork sizing
SINGLE_COL = 3.54      # 90 mm
COL_1_5 = 5.51         # 140 mm
DOUBLE_COL = 7.48      # 190 mm

# Palette Okabe-Ito (validator dataviz PASS: light + CVD)
ORANGE, BLUE, GREEN, PURPLE, GRAY = ('#D55E00', '#0072B2', '#009E73',
                                     '#CC79A7', '#6e6e6e')


def apply_journal_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 8.5,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.5,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.4,
        'pdf.fonttype': 42,          # TrueType — bắt buộc cho nhiều NXB
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': True,
    })


def save_journal(fig, name, fig_dir=r'c:\Users\admin\NCKH\figures'):
    """PDF vector (chính) + PNG 600 dpi (review) + PNG 300 dpi (xem nhanh)."""
    fig.savefig(os.path.join(fig_dir, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, f'{name}.png'), dpi=600,
                bbox_inches='tight')
    print(f'saved {name} (.pdf vector + .png 600dpi)')
