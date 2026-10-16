"""
Fig. 10 — Distribution of forest-to-agriculture conversion by slope band
and the window of intervention.

Panel (a): Converted vs remaining unconverted area per slope band,
           with the 8–15° belt highlighted as the window of intervention.
Panel (b): Trajectory of conversion share by slope band across the
           three phases (2000–2010, 2010–2020, 2020–2025).

Reads:
  - window_of_intervention.csv
  - total_slope_area.csv
No hard-coded numbers.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BASE = Path(r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2\CONPONENT_MAPS_CON2")
WOI_CSV = BASE / "window_of_intervention.csv"
TOTAL_CSV = BASE / "total_slope_area.csv"
OUTDIR = Path(
    r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2\OUTPUT_CON2"
    r"\RESULTS_SECTION_3_4\Figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

BANDS = ["< 8°", "8–15°", "15–25°", "> 25°"]
PHASES = ["2000–2010", "2010–2020", "2020–2025"]

# Palette — colour-blind safe
COL_CONVERTED = "#B2182B"     # deep red
COL_REMAINING = "#2166AC"     # deep blue
COL_WINDOW    = "#F4A582"     # highlight orange

# Colors for the 4 slope bands in panel (b)
BAND_COLORS = {
    "< 8°":   "#F4D03F",      # gold
    "8–15°":  "#E67E22",      # orange
    "15–25°": "#B2182B",      # deep red
    "> 25°":  "#7B1FA2",      # purple
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10.5, "axes.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "savefig.dpi": 300,
})

# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
print("Loading data ...")
total = pd.read_csv(TOTAL_CSV, encoding="utf-8-sig")
woi   = pd.read_csv(WOI_CSV,   encoding="utf-8-sig")

# Normalize column names
total = total.rename(columns={
    "Dải dốc": "band",
    "Diện tích (km²)": "total_km2",
    "Diện tích đã chuyển đổi (ha)": "converted_ha",
    "Diện tích còn lại (ha)": "remaining_ha",
})
woi = woi.rename(columns={
    "Giai đoạn": "phase",
    "Dải dốc": "band",
    "Tỷ lệ (%)": "share_pct",
})

# Convert ha → km²
total["converted_km2"] = total["converted_ha"] / 100.0
total["remaining_km2"] = total["remaining_ha"] / 100.0

# Order bands
total["band"] = pd.Categorical(total["band"], categories=BANDS, ordered=True)
total = total.sort_values("band").reset_index(drop=True)

woi["band"]  = pd.Categorical(woi["band"],  categories=BANDS,  ordered=True)
woi["phase"] = pd.Categorical(woi["phase"], categories=PHASES, ordered=True)
woi = woi.sort_values(["phase", "band"]).reset_index(drop=True)

print(total[["band", "total_km2", "converted_km2", "remaining_km2"]]
      .to_string(index=False))
print()
print(woi[["phase", "band", "share_pct"]].to_string(index=False))

# Derived: converted share per band
total["converted_pct"] = (
    100 * total["converted_km2"] / total["total_km2"]
).round(2)
total["remaining_pct"] = (
    100 * total["remaining_km2"] / total["total_km2"]
).round(2)

# ------------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2, figsize=(15, 6.8), dpi=300,
    gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.24},
)

# ==================================================================
# Panel (a) — Converted vs remaining, stacked, 8–15° highlighted
# ==================================================================
ax = axes[0]

x = np.arange(len(BANDS))
w = 0.62

conv_vals = total["converted_km2"].values
rem_vals  = total["remaining_km2"].values

bars_conv = ax.bar(x, conv_vals, width=w,
                   color=COL_CONVERTED, edgecolor="white",
                   linewidth=1.3, label="Converted to agriculture",
                   zorder=3)
bars_rem  = ax.bar(x, rem_vals, width=w, bottom=conv_vals,
                   color=COL_REMAINING, edgecolor="white",
                   linewidth=1.3, label="Remaining unconverted",
                   zorder=3)

# Percent annotation inside each segment
for i in range(len(BANDS)):
    # Converted %
    if conv_vals[i] > 300:          # only label if segment is tall enough
        ax.text(x[i], conv_vals[i] / 2,
                f"{total['converted_pct'].iloc[i]:.1f}%",
                ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold",
                zorder=5)
    # Remaining %
    ax.text(x[i], conv_vals[i] + rem_vals[i] / 2,
            f"{total['remaining_pct'].iloc[i]:.1f}%",
            ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold",
            zorder=5)

# Highlight the 8–15° belt (index 1)
idx_win = BANDS.index("8–15°")
ax.add_patch(FancyBboxPatch(
    (idx_win - w/2 - 0.06, 0),
    w + 0.12, conv_vals[idx_win] + rem_vals[idx_win] + 800,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    facecolor="none", edgecolor=COL_WINDOW,
    linewidth=3.0, linestyle="--",
    zorder=6, alpha=0.95,
))

# Arrow + label
arrow = FancyArrowPatch(
    (idx_win + 0.9, conv_vals[idx_win] + rem_vals[idx_win] * 0.55),
    (idx_win + 0.15, conv_vals[idx_win] + rem_vals[idx_win] * 0.55),
    arrowstyle="-|>", mutation_scale=20,
    color=COL_WINDOW, linewidth=2.4, zorder=7,
)
ax.add_patch(arrow)
ax.text(
    idx_win + 0.95, conv_vals[idx_win] + rem_vals[idx_win] * 0.55,
    "Window of\nintervention\n(8–15°)",
    ha="left", va="center",
    fontsize=11, fontweight="bold",
    color=COL_WINDOW,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
              edgecolor=COL_WINDOW, linewidth=1.5, alpha=0.97),
    zorder=8,
)

ax.set_xticks(x)
ax.set_xticklabels(BANDS, fontweight="bold")
ax.set_xlabel("Slope band", fontweight="bold", labelpad=8)
ax.set_ylabel("Area (km²)", fontweight="bold", labelpad=8)
ax.set_title(
    "(a) Converted vs. remaining unconverted area by slope band",
    fontweight="bold", pad=12,
)
ax.legend(loc="upper right", framealpha=0.96,
          edgecolor="#B0B0B0", fontsize=10.5)
ax.grid(alpha=0.22, lw=0.6, axis="y")
ax.set_axisbelow(True)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_ylim(0, max(conv_vals + rem_vals) * 1.15)

# ==================================================================
# Panel (b) — Trajectory of conversion share by slope band
# ==================================================================
ax = axes[1]

# Pivot to matrix: rows = bands, cols = phases
pivot = woi.pivot(index="band", columns="phase", values="share_pct")
pivot = pivot.reindex(index=BANDS, columns=PHASES)

markers = {"< 8°": "o", "8–15°": "s", "15–25°": "^", "> 25°": "D"}

for band in BANDS:
    y = pivot.loc[band].values
    ax.plot(
        PHASES, y,
        marker=markers[band], ms=9.5, lw=2.6,
        color=BAND_COLORS[band],
        label=f"{band}",
        markeredgecolor="white", markeredgewidth=1.3,
        zorder=4,
    )
    # End-of-line % annotation
    ax.text(
        len(PHASES) - 1 + 0.10, y[-1],
        f"  {y[-1]:.1f}%",
        ha="left", va="center",
        fontsize=10.5, fontweight="bold",
        color=BAND_COLORS[band],
        zorder=5,
    )

# Emphasis on 8–15° — thicker dashed trail
y_win = pivot.loc["8–15°"].values
ax.plot(
    PHASES, y_win,
    marker="s", ms=11, lw=3.4, linestyle="--",
    color=BAND_COLORS["8–15°"],
    markerfacecolor="white", markeredgewidth=2.4,
    zorder=5, alpha=0.95,
)

ax.set_xticks(range(len(PHASES)))
ax.set_xticklabels(PHASES, fontweight="bold")
ax.set_xlim(-0.15, len(PHASES) - 0.5)
ax.set_ylim(0, 72)
ax.set_xlabel("Phase", fontweight="bold", labelpad=8)
ax.set_ylabel("Share of phase conversion (%)", fontweight="bold", labelpad=8)
ax.set_title(
    "(b) Trajectory of conversion share by slope band",
    fontweight="bold", pad=12,
)
ax.legend(loc="center right", framealpha=0.96,
          edgecolor="#B0B0B0", fontsize=10, title="Slope band",
          title_fontsize=10)
ax.grid(alpha=0.22, lw=0.6)
ax.set_axisbelow(True)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Upslope annotation
ax.annotate(
    "Upslope shift: 1.19 % yr⁻¹",
    xy=(1.85, 34), xytext=(0.7, 58),
    fontsize=10.5, fontweight="bold",
    color="#404040",
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7",
              edgecolor="#888888", linewidth=1.1, alpha=0.95),
    arrowprops=dict(arrowstyle="-|>", color="#404040",
                    lw=1.6, connectionstyle="arc3,rad=-0.25"),
    zorder=6,
)

# ------------------------------------------------------------------
# Suptitle
# ------------------------------------------------------------------
fig.suptitle(
    "Distribution of forest-to-agriculture conversion by slope band "
    "and the window of intervention",
    fontsize=13.5, fontweight="bold", y=1.015,
)
fig.tight_layout()

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
png = OUTDIR / "Fig10_SlopeZones_Intervention.png"
pdf = OUTDIR / "Fig10_SlopeZones_Intervention.pdf"
tif = OUTDIR / "Fig10_SlopeZones_Intervention.tiff"

fig.savefig(png, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
fig.savefig(tif, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nSaved:")
print(f"  {png}")
print(f"  {pdf}")
print(f"  {tif}")