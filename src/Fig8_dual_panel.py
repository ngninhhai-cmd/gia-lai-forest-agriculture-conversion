"""
Fig. 8 (v1) — Effect of predictors on forest-to-agriculture conversion probability
Partial dependence plots (PDP) for 5 predictors, each panel showing:
  - Top marginal: distribution of observed predictor values
  - Main: PDP curve with 95% CI ribbon
  - Bottom: rug plot of observations
  - Annotation: beta, z, p, Direct/Indirect/Total impacts
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import KNN
from spreg import ML_Lag

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
FISHNET = Path(r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2\OUTPUT_CON2\Fishnet_Master_7vars.gpkg")
IMPACTS = Path(
    r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2\OUTPUT_CON2"
    r"\RESULTS_SECTION_3_3\Tables\Table_SAR_Impacts_fullmatrix.csv"
)
OUTDIR = Path(
    r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2\OUTPUT_CON2"
    r"\RESULTS_SECTION_3_3\Figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

X_COLS = ["dist_road_km", "slope_mean", "soil_basalt",
          "pop_density", "GDP_norm_rich"]

LABELS = [
    "Distance to major road",
    "Mean slope",
    "Basaltic soil",
    "Population density",
    "Normalised GDP",
]
UNITS = ["km", "%", "fraction (0–1)", "people km⁻²", "rich index (0–1)"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.labelsize": 11,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 10, "axes.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "savefig.dpi": 300,
})

# ------------------------------------------------------------------
# 1. LOAD + FIT SAR
# ------------------------------------------------------------------
print("Loading fishnet ...")
gdf = gpd.read_file(FISHNET, layer="fishnet_master")
gdf["cx"] = gdf.geometry.centroid.x
gdf["cy"] = gdf.geometry.centroid.y
df = gdf.dropna(subset=["conversion_binary"] + X_COLS + ["cx", "cy"]).copy()
print(f"n = {len(df)}")

coords = np.column_stack([df["cx"].values, df["cy"].values])
w = KNN.from_array(coords, k=8)
w.transform = "r"

X = df[X_COLS].values.astype(float)
y = df["conversion_binary"].values.reshape(-1, 1)

print("Fitting SAR ML_Lag ...")
sar = ML_Lag(y, X, w=w, name_x=X_COLS, name_y="Y")

betas_all = np.array(sar.betas).flatten()
se_all = np.array(sar.std_err).flatten()
z_all = np.array(sar.z_stat).reshape(-1, 2)

intercept = betas_all[0]
beta_pred = betas_all[1:6]
se_pred = se_all[1:6]
z_pred = z_all[1:6, 0]
p_pred = z_all[1:6, 1]

print(f"rho = {float(sar.rho):.6f}")

# ------------------------------------------------------------------
# 2. IMPACTS
# ------------------------------------------------------------------
impacts = pd.read_csv(IMPACTS, encoding="utf-8-sig").set_index("Predictor")
impacts = impacts.reindex(X_COLS)

# ------------------------------------------------------------------
# 3. HELPER
# ------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

# ------------------------------------------------------------------
# 4. FIGURE — 1x5
# ------------------------------------------------------------------
fig = plt.figure(figsize=(22, 6.5), dpi=300)
outer = gridspec.GridSpec(
    1, 5, figure=fig, wspace=0.34,
    left=0.045, right=0.985, top=0.86, bottom=0.11,
)

x_means = X.mean(axis=0)

for i, name in enumerate(X_COLS):
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[i],
        height_ratios=[0.55, 3.4, 0.35], hspace=0.03,
    )
    ax_h = fig.add_subplot(inner[0])   # marginal histogram
    ax_m = fig.add_subplot(inner[1])   # PDP
    ax_r = fig.add_subplot(inner[2])   # rug

    x_obs = X[:, i]
    x_lo, x_hi = np.percentile(x_obs, [1, 99])
    x_grid = np.linspace(x_lo, x_hi, 300)

    # PDP: hold other predictors at their mean
    logit_base = intercept + sum(
        beta_pred[j] * x_means[j] for j in range(5) if j != i
    )
    logit_curve = logit_base + beta_pred[i] * x_grid
    p_curve = sigmoid(logit_curve)

    logit_lo = logit_base + (beta_pred[i] - 1.96 * se_pred[i]) * x_grid
    logit_hi = logit_base + (beta_pred[i] + 1.96 * se_pred[i]) * x_grid
    p_lo = sigmoid(logit_lo)
    p_hi = sigmoid(logit_hi)

    color = "#B2182B" if beta_pred[i] < 0 else "#2166AC"

    # -- Main PDP --
    ax_m.fill_between(x_grid, p_lo, p_hi, alpha=0.20,
                      color=color, linewidth=0)
    ax_m.plot(x_grid, p_curve, color=color, lw=2.6, zorder=3)
    ax_m.axhline(0.5, color="#888888", lw=0.9, ls=":", alpha=0.6, zorder=1)
    ax_m.set_xlim(x_lo, x_hi)
    ax_m.set_ylim(0, 1)
    ax_m.set_ylabel("P(conversion)" if i == 0 else "",
                    fontweight="bold")
    ax_m.set_xlabel(f"{LABELS[i]} ({UNITS[i]})",
                    fontweight="bold", fontsize=10.5)
    ax_m.grid(alpha=0.22, lw=0.6)
    ax_m.set_axisbelow(True)
    ax_m.tick_params(axis="both", which="major", labelsize=10)

    # -- Annotation box --
    imp = impacts.loc[name]
    txt = (
        f"$\\beta$ = {beta_pred[i]:+.5f} {stars(p_pred[i])}\n"
        f"$z$ = {z_pred[i]:+.2f}\n"
        f"\n"
        f"Direct    = {imp['Direct']:+.5f}\n"
        f"Indirect  = {imp['Indirect']:+.5f}\n"
        f"Total     = {imp['Total']:+.5f}"
    )
    ax_m.text(
        0.03, 0.97, txt, transform=ax_m.transAxes,
        fontsize=8.8, va="top", ha="left", family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor=color, linewidth=1.3, alpha=0.96),
        zorder=5,
    )

    # -- Top marginal histogram --
    bins = np.linspace(x_lo, x_hi, 45)
    ax_h.hist(x_obs, bins=bins, color=color, alpha=0.55,
              edgecolor="white", linewidth=0.3)
    ax_h.set_xlim(x_lo, x_hi)
    ax_h.axis("off")

    # -- Bottom rug --
    ax_r.scatter(x_obs, np.zeros_like(x_obs),
                 s=1.2, alpha=0.28, color=color,
                 marker="|", linewidths=0.8)
    ax_r.set_xlim(x_lo, x_hi)
    ax_r.set_yticks([])
    ax_r.set_xticks([])
    for sp in ["top", "right", "left", "bottom"]:
        ax_r.spines[sp].set_visible(False)

    # -- Panel title --
    ax_h.set_title(f"({chr(97 + i)}) {LABELS[i]}",
                   fontweight="bold", pad=6, fontsize=11.5)

# -- Suptitle --
fig.suptitle(
    "Effect of predictors on forest-to-agriculture conversion probability "
    "(SAR ML_Lag; n = 15,886)",
    fontsize=13.5, fontweight="bold", y=0.97,
)

# -- Save --
png = OUTDIR / "Fig8_v1_EffectCurves.png"
pdf = OUTDIR / "Fig8_v1_EffectCurves.pdf"
tif = OUTDIR / "Fig8_v1_EffectCurves.tiff"
fig.savefig(png, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
fig.savefig(tif, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nSaved:")
print(f"  {png}")
print(f"  {pdf}")
print(f"  {tif}")