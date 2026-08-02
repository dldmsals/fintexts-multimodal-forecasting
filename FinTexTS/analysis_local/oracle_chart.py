import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── data ──────────────────────────────────────────────────────────────────────
ratios = [0.0, 0.5, 0.8, 1.0]
film_linq  = [0.518, 0.779, 0.908, 0.996]   # FiLM (Linq 64d)
tmoe_bert  = [0.533, 0.772, 0.904, 0.998]   # TextMoE (BERT)

# ── palette (validated default, slots 1 & 2) ─────────────────────────────────
C1 = "#2a78d6"   # blue  — FiLM Linq  (slot 1)
C2 = "#1baf7a"   # aqua  — TextMoE BERT (slot 2)
SURFACE  = "#fcfcfb"
GRID     = "#e1e0d9"
INK      = "#0b0b0b"
INK2     = "#52514e"
MUTED    = "#898781"

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# ── shaded region: gap between "real text" and oracle ─────────────────────────
real_text_avg = (film_linq[0] + tmoe_bert[0]) / 2  # ~0.526
oracle_avg    = (film_linq[-1] + tmoe_bert[-1]) / 2  # ~0.997
ax.axhspan(real_text_avg, oracle_avg, xmin=0, xmax=1,
           color="#e1e0d9", alpha=0.35, zorder=0, linewidth=0)

# bracket annotation (right side)
ax.annotate("", xy=(1.04, oracle_avg), xytext=(1.04, real_text_avg),
            xycoords=("axes fraction", "data"),
            textcoords=("axes fraction", "data"),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
ax.text(1.055, (oracle_avg + real_text_avg) / 2,
        "임베딩\n정보량\n격차",
        va="center", ha="left", fontsize=8.5, color=MUTED,
        transform=ax.get_yaxis_transform())

# ── random baseline ───────────────────────────────────────────────────────────
ax.axhline(0.5, color=MUTED, lw=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(1.02, 0.5, "무작위 (0.50)",
        va="center", ha="left", fontsize=8, color=MUTED,
        transform=ax.get_yaxis_transform())

# ── series lines ──────────────────────────────────────────────────────────────
ax.plot(ratios, film_linq, color=C1, lw=2, marker="o",
        markersize=8, markerfacecolor=C1, markeredgecolor=SURFACE,
        markeredgewidth=2, zorder=4, label="FiLM  (Linq 64d)")

ax.plot(ratios, tmoe_bert, color=C2, lw=2, marker="o",
        markersize=8, markerfacecolor=C2, markeredgecolor=SURFACE,
        markeredgewidth=2, zorder=4, label="TextMoE  (BERT 384d)")

# ── endpoint labels ───────────────────────────────────────────────────────────
def label_point(ax, x, y, text, color, offset_x=0.02, offset_y=0.0,
                ha="left"):
    ax.text(x + offset_x, y + offset_y, text,
            fontsize=8.5, color=color, va="center", ha=ha,
            fontweight="600")

# ratio = 0 (real text)
label_point(ax, 0.0, film_linq[0],  "0.518", C1, offset_x=0.02, offset_y=-0.028)
label_point(ax, 0.0, tmoe_bert[0],  "0.533", C2, offset_x=0.02, offset_y= 0.022)

# ratio = 1.0 (oracle)
label_point(ax, 1.0, film_linq[-1],  "0.996", C1, offset_x=-0.03,
            offset_y=-0.028, ha="right")
label_point(ax, 1.0, tmoe_bert[-1],  "0.998", C2, offset_x=-0.03,
            offset_y= 0.022, ha="right")

# ── x-axis annotations ────────────────────────────────────────────────────────
ax.text(0.0, 0.435, "실제 텍스트\n(임베딩만)", ha="center",
        fontsize=8, color=INK2)
ax.text(1.0, 0.435, "Oracle\n(정답 전체 주입)", ha="center",
        fontsize=8, color=INK2)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.08, 1.12)
ax.set_ylim(0.42, 1.05)
ax.set_xticks([0.0, 0.5, 0.8, 1.0])
ax.set_xticklabels(["0.0", "0.5", "0.8", "1.0"],
                   fontsize=10, color=INK2)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.50", "0.60", "0.70", "0.80", "0.90", "1.00"],
                   fontsize=9.5, color=INK2)

ax.set_xlabel("Oracle 주입 비율", fontsize=11, color=INK2, labelpad=8)
ax.set_ylabel("Direction Accuracy  (↑ 높을수록 좋음)", fontsize=11,
              color=INK2, labelpad=8)

# ── grid ──────────────────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.8, linestyle="-")
ax.xaxis.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

# ── title ─────────────────────────────────────────────────────────────────────
ax.set_title("Oracle 진단: 정보를 주입할수록 두 모델 모두 동일하게 상승",
             fontsize=12.5, color=INK, fontweight="700",
             pad=14, loc="left")
ax.text(0, 1.01,
        "→ 병목은 아키텍처가 아니라 임베딩의 정보량",
        transform=ax.transAxes,
        fontsize=9.5, color=INK2, va="bottom")

# ── legend ────────────────────────────────────────────────────────────────────
legend = ax.legend(fontsize=9.5, frameon=True, loc="upper left",
                   framealpha=1, edgecolor=GRID,
                   facecolor=SURFACE, borderpad=0.7,
                   handlelength=1.6, handletextpad=0.5)
for text in legend.get_texts():
    text.set_color(INK2)

plt.tight_layout(rect=[0, 0, 0.88, 1])
out = "/home/eb/LG_AI/FinTexTS/analysis_local/oracle_diraccuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"saved → {out}")
