"""
Generate Figure 2: Domain gap bar chart (val vs test, both models).
Run from the paper/ directory:
    python make_domain_gap_figure.py
Output: figure/domain_gap.png  (and .pdf)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figure", exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
models     = ["DirectPoseCNN\n(λ_t=2.0, λ_r=1.0)", "CNN+GRU\n(T=10)"]
val_trans  = [14.64, 20.76]   # %
test_trans = [41.05, 67.22]   # %
val_rot    = [92.46, 92.24]   # degrees
test_rot   = [94.69, 92.32]   # degrees  (1.6526 rad, 1.6113 rad)

# ── Layout ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
fig.subplots_adjust(wspace=0.38)

x      = np.arange(len(models))
width  = 0.35

colors_val  = ["#4C72B0", "#4C72B0"]
colors_test = ["#DD8452", "#DD8452"]

# ── Left panel: Translation error ────────────────────────────────────────────
ax = axes[0]
bars_val  = ax.bar(x - width/2, val_trans,  width, label="Val (synthetic)",
                   color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.6)
bars_test = ax.bar(x + width/2, test_trans, width, label="Test (real)",
                   color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.6)

# Amplification arrows
for i in range(len(models)):
    ax.annotate(
        f"×{test_trans[i]/val_trans[i]:.2f}",
        xy=(x[i] + width/2, test_trans[i]),
        xytext=(x[i] + width/2, test_trans[i] + 2.5),
        ha="center", va="bottom", fontsize=7.5, color="#CC4400",
        arrowprops=dict(arrowstyle="-", color="#CC4400", lw=0.8),
    )

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8.5)
ax.set_ylabel("Relative translation error (%)", fontsize=9)
ax.set_title("Translation Domain Gap", fontsize=10, fontweight="bold")
ax.set_ylim(0, 80)
ax.legend(fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

# Value labels on bars
for bar in bars_val:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5)
for bar in bars_test:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5)

# ── Right panel: Rotation error ───────────────────────────────────────────────
ax = axes[1]
bars_val  = ax.bar(x - width/2, val_rot,  width, label="Val (synthetic)",
                   color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.6)
bars_test = ax.bar(x + width/2, test_rot, width, label="Test (real)",
                   color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.6)

# Delta labels above test bars
for i in range(len(models)):
    delta = test_rot[i] - val_rot[i]
    sign  = "+" if delta >= 0 else ""
    ax.annotate(
        f"{sign}{delta:.2f}°",
        xy=(x[i] + width/2, test_rot[i]),
        xytext=(x[i] + width/2, test_rot[i] + 0.4),
        ha="center", va="bottom", fontsize=7.5,
        color="#CC4400" if delta > 0 else "#2A7A2A",
    )

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8.5)
ax.set_ylabel("Rotation error (°)", fontsize=9)
ax.set_title("Rotation Domain Gap", fontsize=10, fontweight="bold")
ax.set_ylim(80, 100)
ax.legend(fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

for bar in bars_val:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}°", ha="center", va="bottom", fontsize=7.5)
for bar in bars_test:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}°", ha="center", va="bottom", fontsize=7.5)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.suptitle("Synthetic-to-Real Domain Gap", fontsize=11, fontweight="bold", y=1.01)
fig.savefig("figure/domain_gap.png", dpi=200, bbox_inches="tight")
fig.savefig("figure/domain_gap.pdf", bbox_inches="tight")
print("Saved: figure/domain_gap.png  and  figure/domain_gap.pdf")
