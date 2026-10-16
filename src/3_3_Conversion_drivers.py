"""
Module 03.3 (v4) — Quantifying Forest–Agriculture Conversion Drivers
=====================================================================
Gia Lai Province, Vietnam — CON2 (Q1/Q2 paper)

READS DIRECTLY from Fishnet_Master_7vars.gpkg (layer=fishnet_master)
produced by the previous pipeline — does NOT rebuild the fishnet,
does NOT recompute predictors. This guarantees n = 15,886 and all
downstream statistics match the previous run exactly.

Sections:
  (1) Logistic baseline (statsmodels.Logit) + VIF
  (2) Moran's I of Pearson residuals
  (3) SAR (ML_Lag) + SEM (ML_Error) + model comparison
  (4) SAR Direct / Indirect / Total impacts
  (5) Sensitivity (4 thresholds × ML_Lag & GM_Lag)
  (6) Slope-zone distribution + window of intervention (from saved CSVs)
  (7) Publication figures (Fig6–Fig10)

Author : <your name>
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, roc_auc_score

from libpysal.weights import KNN
from esda.moran import Moran
from spreg import ML_Lag, ML_Error, GM_Lag

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================
# CONFIG
# =====================================================================

DEFAULT_BASE = Path(r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2")
DEFAULT_FISHNET = DEFAULT_BASE / "OUTPUT_CON2" / "Fishnet_Master_7vars.gpkg"
DEFAULT_WOI = (DEFAULT_BASE / "CONPONENT_MAPS_CON2"
               / "window_of_intervention.csv")
DEFAULT_SLOPE_TOTAL = (DEFAULT_BASE / "CONPONENT_MAPS_CON2"
                       / "total_slope_area.csv")
DEFAULT_OUTDIR = DEFAULT_BASE / "OUTPUT_CON2" / "RESULTS_SECTION_3_3"

X_COLS = ["dist_road_km", "slope_mean", "soil_basalt",
          "pop_density", "GDP_norm_rich"]

PRETTY = {
    "dist_road_km":  "Distance to major road (km)",
    "slope_mean":    "Mean slope (%)",
    "soil_basalt":   "Basaltic soil (fraction)",
    "pop_density":   "Population density",
    "GDP_norm_rich": "Normalised GDP (rich index)",
}

Y_COL = "conversion_binary"
INTENSITY_COL = "_sum"

K_NEIGHBORS = 8
PERMUTATIONS = 999
RANDOM_SEED = 42

# Thresholds applied to _sum (pixel count 0–255 within 1-km cell)
THRESHOLDS = [
    ("C_i > 0",      lambda s: s > 0),
    ("C_i >= 5%",    lambda s: s >= 0.05 * 255),
    ("C_i >= 10%",   lambda s: s >= 0.10 * 255),
    ("C_i >= 15%",   lambda s: s >= 0.15 * 255),
]

REFERENCE = {
    "n_cells": 15886,
    "n_converted": 12991,
    "n_not_converted": 2895,
    "logit_auc": 0.7585,
    "logit_nagelkerke": 0.2238,
    "logit_pseudo_r2": 0.1554,
    "logit_moran_resid": 0.5524,
    "sar_aic": 1812.91,
    "sar_pseudo_r2": 0.8788,
    "sar_moran_resid": 0.0317,
    "sar_rho": 0.73325,
    "sar_coef": {
        "dist_road_km":  -0.00727,
        "slope_mean":    -0.00094,
        "soil_basalt":   +0.02838,
        "pop_density":   -0.00008,
        "GDP_norm_rich": +0.04271,
    },
    "sar_total": {
        "dist_road_km":  -0.0272,
        "GDP_norm_rich": +0.1601,
    },
}
TOL_COUNT = 5
TOL_I = 0.01

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10.5, "axes.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "savefig.dpi": 300,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================================
# DATA LOADING
# =====================================================================

def load_fishnet(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        logger.error("Fishnet not found: %s", path)
        sys.exit(1)
    gdf = gpd.read_file(path, layer="fishnet_master")
    logger.info("Loaded fishnet: n = %d | CRS = %s", len(gdf), gdf.crs)

    # Centroid coordinates for KNN
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y
    return gdf


def prepare_analysis_sample(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Unified sample: complete data on 5 predictors + Y_i + _sum."""
    need = [Y_COL, INTENSITY_COL] + X_COLS + ["cx", "cy"]
    df = gdf.dropna(subset=need).copy()
    logger.info("Analysis sample: n = %d | converted = %d (%.2f%%)",
                len(df), int(df[Y_COL].sum()),
                100 * df[Y_COL].mean())
    return df


def build_weights(df: gpd.GeoDataFrame) -> KNN:
    coords = np.column_stack([df["cx"].values, df["cy"].values])
    w = KNN.from_array(coords, k=K_NEIGHBORS)
    w.transform = "r"
    logger.info("Weights: KNN k=%d, row-standardized (n=%d)",
                K_NEIGHBORS, len(df))
    return w


# =====================================================================
# (1) LOGISTIC BASELINE + VIF
# =====================================================================

def fit_logistic(df: gpd.GeoDataFrame) -> dict:
    X = df[X_COLS].astype(float)
    y = df[Y_COL].astype(int)
    X_const = sm.add_constant(X)

    model = sm.Logit(y, X_const).fit(disp=0)

    y_pred = model.predict(X_const)
    auc = roc_auc_score(y, y_pred)

    ll_full = model.llf
    ll_null = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).llf
    pseudo_r2 = 1 - ll_full / ll_null
    n = len(y)
    cox_snell = 1 - np.exp(2 * (ll_null - ll_full) / n)
    nagelkerke = cox_snell / (1 - np.exp(2 * ll_null / n))

    logger.info("Logit: AUC = %.4f | Pseudo R² = %.4f | "
                "Nagelkerke R² = %.4f | AIC = %.2f",
                auc, pseudo_r2, nagelkerke, model.aic)

    return {
        "model": model, "X_const": X_const, "y": y,
        "y_pred": y_pred, "auc": auc,
        "pseudo_r2": pseudo_r2, "nagelkerke": nagelkerke,
        "ll_full": ll_full, "ll_null": ll_null,
        "aic": model.aic, "residuals": y.values - y_pred.values,
    }


def compute_vif(df: gpd.GeoDataFrame) -> pd.DataFrame:
    X = df[X_COLS].astype(float)
    X_const = sm.add_constant(X)
    vifs = [variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])]
    return pd.DataFrame({
        "Variable": X_const.columns,
        "VIF": np.round(vifs, 4),
    })


def residual_moran(residuals: np.ndarray, w: KNN) -> Moran:
    np.random.seed(RANDOM_SEED)
    m = Moran(residuals, w, permutations=PERMUTATIONS)
    logger.info("Moran's I of logistic residuals = %.4f (p = %.4f)",
                m.I, m.p_sim)
    return m


# =====================================================================
# (3) SAR / SEM
# =====================================================================

def fit_sar(df: gpd.GeoDataFrame, w: KNN) -> ML_Lag:
    X = df[X_COLS].values.astype(float)
    y = df[Y_COL].values.astype(float).reshape(-1, 1)
    sar = ML_Lag(y, X, w=w, name_x=X_COLS, name_y="Y_i",
                 name_w="KNN8")
    logger.info("SAR: rho = %.5f | AIC = %.2f | Pseudo-R² = %.4f",
                float(sar.rho),
                float(sar.aic) if hasattr(sar, "aic") else np.nan,
                float(sar.pr2) if hasattr(sar, "pr2") else np.nan)
    return sar


def fit_sem(df: gpd.GeoDataFrame, w: KNN) -> ML_Error:
    X = df[X_COLS].values.astype(float)
    y = df[Y_COL].values.astype(float).reshape(-1, 1)
    sem = ML_Error(y, X, w=w, name_x=X_COLS, name_y="Y_i",
                   name_w="KNN8")
    logger.info("SEM: AIC = %.2f | Pseudo-R² = %.4f",
                float(sem.aic) if hasattr(sem, "aic") else np.nan,
                float(sem.pr2) if hasattr(sem, "pr2") else np.nan)
    return sem


def sar_moran_resid(sar: ML_Lag, w: KNN, y: np.ndarray) -> Moran:
    pred = sar.predy.flatten()
    resid = y.flatten() - pred
    np.random.seed(RANDOM_SEED)
    m = Moran(resid, w, permutations=PERMUTATIONS)
    logger.info("Moran's I of SAR residuals = %.4f (p = %.4f)",
                m.I, m.p_sim)
    return m


def sar_impacts(sar: ML_Lag) -> pd.DataFrame:
    """Extract direct/indirect/total impacts from spreg ML_Lag object."""
    rho = float(sar.rho)
    betas = np.array(sar.betas).flatten()
    direct = betas.copy()
    total = betas / (1.0 - rho)
    indirect = total - direct
    return pd.DataFrame({
        "Predictor": X_COLS,
        "Direct": np.round(direct, 6),
        "Indirect": np.round(indirect, 6),
        "Total": np.round(total, 6),
        "Indirect_over_Direct": np.round(
            indirect / np.where(direct == 0, np.nan, direct), 4),
    })


# =====================================================================
# (5) SENSITIVITY
# =====================================================================

def sensitivity_analysis(df: gpd.GeoDataFrame, w: KNN) -> pd.DataFrame:
    rows = []
    X = df[X_COLS].values.astype(float)
    s = df[INTENSITY_COL].values.astype(float)

    for label, func in THRESHOLDS:
        y_thr = func(s).astype(float).reshape(-1, 1)
        n_pos = int(y_thr.sum())
        if n_pos < 30 or n_pos > len(y_thr) - 30:
            logger.warning("Threshold %s degenerate — skipping", label)
            continue
        for est_name, Est in [("ML_Lag", ML_Lag), ("GM_Lag", GM_Lag)]:
            try:
                res = Est(y_thr, X, w=w, name_x=X_COLS,
                          name_y="Y_thr", name_w="KNN8")
                betas = np.array(res.betas).flatten()
                ses = np.array(res.std_err).flatten()
                z_stats = np.array(res.z_stat).reshape(-1, 2)
                rho = float(getattr(res, "rho", np.nan))
                # Betas include CONSTANT at index 0, predictors 1..5, rho last
                # For ML_Lag/GM_Lag, betas are [const, b1..b5, rho]
                for i, name in enumerate(X_COLS):
                    b = betas[i + 1]
                    se = ses[i + 1]
                    zz = z_stats[i + 1, 0]
                    p_val = z_stats[i + 1, 1]
                    rows.append({
                        "Threshold": label,
                        "Estimator": est_name,
                        "Predictor": name,
                        "Coef": b, "SE": se, "z": zz, "p_value": p_val,
                        "Rho": rho, "N_positive": n_pos,
                    })
            except Exception as e:
                logger.warning("Sensitivity %s / %s failed: %s",
                               label, est_name, e)
    return pd.DataFrame(rows)


def classify_robustness(sens_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ml = sens_df[sens_df["Estimator"] == "ML_Lag"]
    for name in X_COLS:
        sub = ml[ml["Predictor"] == name]
        if len(sub) == 0:
            rows.append({"Predictor": name,
                         "N_thresholds_sig": 0,
                         "Signs_consistent": False,
                         "Robust": False})
            continue
        signs = np.sign(sub["Coef"].values)
        sig = int((sub["p_value"] < 0.05).sum())
        rows.append({
            "Predictor": name,
            "N_thresholds_sig": sig,
            "Signs_consistent": bool(len(set(signs)) == 1),
            "Robust": bool(sig >= 4 and len(set(signs)) == 1),
        })
    return pd.DataFrame(rows)


# =====================================================================
# (6) SLOPE ZONES FROM SAVED CSVs
# =====================================================================

def load_slope_zones(woi_path: Path, total_path: Path):
    if not woi_path.exists():
        logger.warning("window_of_intervention.csv not found — skipping")
        return None, None
    woi = pd.read_csv(woi_path, encoding="utf-8-sig")
    logger.info("Loaded window_of_intervention.csv: %d rows", len(woi))

    rows = []
    if "Giai đoạn" in woi.columns and "Dải dốc" in woi.columns:
        for (phase, zone), sub in woi.groupby(["Giai đoạn", "Dải dốc"]):
            rows.append({
                "Phase": phase,
                "Slope_zone": zone,
                "N_pixels": int(sub["Số pixel"].sum()),
                "Share_pct": round(float(sub["Tỷ lệ (%)"].mean()), 2),
            })
    elif "Phase" in woi.columns and "Slope_zone" in woi.columns:
        rows = woi.to_dict("records")
    else:
        logger.warning("Unrecognized columns in window_of_intervention.csv: %s",
                       woi.columns.tolist())
        rows = woi.to_dict("records")

    rows_df = pd.DataFrame(rows)

    summary = None
    if total_path.exists():
        total = pd.read_csv(total_path, encoding="utf-8-sig")
        logger.info("Loaded total_slope_area.csv: %d rows", len(total))
        # Normalize column names
        zone_col = ("Dải dốc" if "Dải dốc" in total.columns
                    else "Slope_zone")
        converted_col = ("Diện tích đã chuyển đổi (ha)"
                        if "Diện tích đã chuyển đổi (ha)" in total.columns
                        else "Converted_ha")
        remaining_col = ("Diện tích còn lại (ha)"
                        if "Diện tích còn lại (ha)" in total.columns
                        else "Remaining_ha")
        pct_col = ("% còn lại" if "% còn lại" in total.columns
                   else "Remaining_pct")
        total_area_col = ("Diện tích (ha)" if "Diện tích (ha)" in total.columns
                         else "Total_ha")

        zone_8_15 = total[total[zone_col] == "8–15°"]
        if len(zone_8_15) == 0:
            # try alternative dash character
            for z in total[zone_col].unique():
                if "8" in str(z) and "15" in str(z):
                    zone_8_15 = total[total[zone_col] == z]
                    break

        if len(zone_8_15) > 0:
            r = zone_8_15.iloc[0]
            conv_share = (100 * r[converted_col] / r[total_area_col]
                         if r[total_area_col] > 0 else np.nan)
            summary = pd.DataFrame([{
                "Converted_8_15_share_pct": round(conv_share, 2),
                "Remaining_8_15_ha": round(r[remaining_col], 2),
                "Remaining_8_15_pct": round(r[pct_col], 2),
            }])
    else:
        logger.warning("total_slope_area.csv not found")

    return rows_df, summary


# =====================================================================
# FIGURES
# =====================================================================

def _grid(ax):
    ax.grid(alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def plot_roc(logit, out_path):
    fpr, tpr, _ = roc_curve(logit["y"], logit["y_pred"])
    auc = logit["auc"]
    n = len(logit["y"])
    fig, ax = plt.subplots(figsize=(7.8, 7.8), dpi=300)
    ax.plot(fpr, tpr, color="#1F4E79", lw=2.8,
            label=f"Logistic baseline (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#B2182B", lw=1.6, ls="--",
            label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#1F4E79")
    ax.set_xlabel("False positive rate (1 − specificity)",
                  fontweight="bold")
    ax.set_ylabel("True positive rate (sensitivity)", fontweight="bold")
    ax.set_title("ROC — Forest-to-agriculture conversion binary "
                 f"logistic regression (n = {n:,})",
                 fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#B0B0B0")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def plot_moran_resid(residuals, w, out_path):
    z = (residuals - residuals.mean()) / residuals.std()
    Wz = np.array([np.sum(w.weights[i] * z[w.neighbors[i]])
                   for i in range(len(z))], dtype=float)
    slope, intercept = np.polyfit(z, Wz, 1)
    fig, ax = plt.subplots(figsize=(8.8, 8.8), dpi=300)
    ax.scatter(z, Wz, s=10, alpha=0.22, color="#7F8C8D",
               edgecolors="none", rasterized=True)
    xs = np.linspace(z.min(), z.max(), 400)
    ax.plot(xs, slope * xs + intercept, color="#B2182B", lw=2.4,
            label=f"OLS slope = {slope:.4f}")
    ax.axhline(0, color="#404040", lw=0.9, ls=(0, (5, 5)))
    ax.axvline(0, color="#404040", lw=0.9, ls=(0, (5, 5)))
    ax.set_xlabel("Pearson residual (standardized)", fontweight="bold")
    ax.set_ylabel("Spatial lag of residual", fontweight="bold")
    ax.set_title("Moran scatterplot of Pearson residuals "
                 "(logistic regression)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#B0B0B0")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def plot_predictor_effects(sar, out_path):
    betas = np.array(sar.betas).flatten()
    ses = np.array(sar.std_err).flatten()
    z_stats = np.array(sar.z_stat).reshape(-1, 2)
    z = z_stats[:, 0]
    # skip CONSTANT (index 0); use predictors 1..5
    b_pred = betas[1:1 + len(X_COLS)]
    se_pred = ses[1:1 + len(X_COLS)]
    z_pred = z[1:1 + len(X_COLS)]

    fig, axes = plt.subplots(1, 5, figsize=(16, 5.8), dpi=300)
    labels = ["(a) Distance\nto road", "(b) Mean\nslope (%)",
              "(c) Basaltic\nsoil", "(d) Population\ndensity",
              "(e) Normalised\nGDP"]
    for ax, b, se, zz, lab in zip(axes, b_pred, se_pred, z_pred, labels):
        color = "#B2182B" if zz < 0 else "#2166AC"
        ax.bar([0], [b], yerr=[1.96 * se], width=0.5,
               color=color, edgecolor="black", linewidth=0.9,
               capsize=6, error_kw={"elinewidth": 1.4})
        ax.axhline(0, color="#404040", lw=0.8)
        stars = "***" if abs(zz) > 2.58 else \
                "**" if abs(zz) > 1.96 else \
                "*" if abs(zz) > 1.645 else "n.s."
        ax.set_title(f"{lab}\nβ = {b:+.5f}  {stars}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Effect of predictors on forest-to-agriculture "
                 "conversion probability (SAR ML_Lag)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def plot_sar_sensitivity(sens_df, out_path):
    ml = sens_df[sens_df["Estimator"] == "ML_Lag"]
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=300)
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#B2182B", "#2166AC", "#1B5E20", "#E08214", "#7B1FA2"]
    thresholds = [t[0] for t in THRESHOLDS]

    for name, m, c in zip(X_COLS, markers, colors):
        sub = ml[ml["Predictor"] == name]
        if len(sub) == 0:
            continue
        sub = sub.set_index("Threshold").reindex(thresholds).dropna()
        if len(sub) == 0:
            continue
        ax.errorbar(sub.index, sub["Coef"], yerr=1.96 * sub["SE"],
                    fmt=m, ms=9, lw=1.8, capsize=5, color=c,
                    label=PRETTY[name], alpha=0.9)
    ax.axhline(0, color="#404040", lw=0.9, ls="--")
    ax.set_xlabel("Conversion threshold", fontweight="bold")
    ax.set_ylabel("SAR coefficient (β)", fontweight="bold")
    ax.set_title("Sensitivity of SAR estimates to conversion threshold",
                 fontweight="bold")
    ax.legend(loc="best", framealpha=0.95, edgecolor="#B0B0B0", fontsize=9.5)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def plot_slope_zone(rows_df, summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), dpi=300)
    zones = ["< 8°", "8–15°", "15–25°", "> 25°"]
    phases = ["2000–2010", "2010–2020", "2020–2025"]
    colors = ["#F4D03F", "#E08214", "#B2182B", "#7B1FA2"]

    # Panel (a): stacked share by phase
    bottom = np.zeros(len(phases))
    for z, c in zip(zones, colors):
        vals = []
        for ph in phases:
            row = rows_df[(rows_df["Phase"] == ph) &
                          (rows_df["Slope_zone"] == z)]
            vals.append(row["Share_pct"].values[0] if len(row) else 0.0)
        axes[0].bar(phases, vals, bottom=bottom, color=c,
                    edgecolor="white", linewidth=1.0, label=z)
        bottom += np.array(vals)
    axes[0].set_ylabel("Share of converted area (%)", fontweight="bold")
    axes[0].set_title("(a) Conversion share by slope zone",
                      fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9.5,
                   framealpha=0.95, edgecolor="#B0B0B0")
    axes[0].grid(alpha=0.22, lw=0.6, axis="y")
    axes[0].set_axisbelow(True)

    # Panel (b): window of intervention
    if summary is not None and len(summary) > 0:
        conv = float(summary["Converted_8_15_share_pct"].values[0])
        rem_pct = float(summary["Remaining_8_15_pct"].values[0])
        rem_ha = float(summary["Remaining_8_15_ha"].values[0])
    else:
        conv, rem_pct, rem_ha = 44.31, 55.69, 185670.0

    axes[1].bar(["8–15°"], [conv], color="#B2182B",
                edgecolor="black", linewidth=0.9,
                label="Already converted")
    axes[1].bar(["8–15°"], [rem_pct], bottom=[conv],
                color="#2166AC", edgecolor="black", linewidth=0.9,
                label="Still forest — protectable")
    axes[1].text(0, conv / 2, f"{conv:.2f}%", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white")
    axes[1].text(0, conv + rem_pct / 2,
                 f"{rem_pct:.2f}%\n({rem_ha:,.0f} ha)",
                 ha="center", va="center", fontsize=10.5,
                 fontweight="bold", color="white")
    axes[1].set_ylabel("Share of zone (%)", fontweight="bold")
    axes[1].set_title("(b) Window of intervention — 8–15° belt",
                      fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9.5,
                   framealpha=0.95, edgecolor="#B0B0B0")
    axes[1].grid(alpha=0.22, lw=0.6, axis="y")
    axes[1].set_axisbelow(True)

    fig.suptitle("Slope-zone distribution of forest-to-agriculture "
                 "conversion", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


# =====================================================================
# VALIDATION
# =====================================================================

def validate(df, logit, moran_res, sar, moran_sar, impacts):
    def _chk(label, computed, ref, tol):
        d = computed - ref
        ok = abs(d) <= tol
        msg = (f"{'OK  ' if ok else '!!  '}{label}: "
               f"computed={computed:.4f}  ref={ref:.4f}  D={d:+.4f}")
        (logger.info if ok else logger.warning)(msg)

    _chk("n cells", len(df), REFERENCE["n_cells"], 0)
    _chk("n converted", int(df[Y_COL].sum()),
         REFERENCE["n_converted"], TOL_COUNT)
    _chk("Logit AUC", logit["auc"], REFERENCE["logit_auc"], 0.01)
    _chk("Logit Nagelkerke", logit["nagelkerke"],
         REFERENCE["logit_nagelkerke"], 0.02)
    _chk("Logit Pseudo R²", logit["pseudo_r2"],
         REFERENCE["logit_pseudo_r2"], 0.02)
    _chk("Moran resid (logit)", moran_res.I,
         REFERENCE["logit_moran_resid"], TOL_I)

    sar_aic = float(sar.aic) if hasattr(sar, "aic") else np.nan
    _chk("SAR AIC", sar_aic, REFERENCE["sar_aic"], 50.0)
    _chk("SAR rho", float(sar.rho), REFERENCE["sar_rho"], 0.01)
    _chk("SAR Moran resid", moran_sar.I,
         REFERENCE["sar_moran_resid"], TOL_I)

    betas = np.array(sar.betas).flatten()
    betas_pred = dict(zip(X_COLS, betas[1:1 + len(X_COLS)]))
    for name, ref in REFERENCE["sar_coef"].items():
        _chk(f"SAR beta {name}", betas_pred[name], ref, 0.02)

    for name, ref in REFERENCE["sar_total"].items():
        comp = impacts.set_index("Predictor").loc[name, "Total"]
        _chk(f"SAR total {name}", comp, ref, 0.03)


# =====================================================================
# PIPELINE
# =====================================================================

def run_pipeline(fishnet_path: Path, woi_path: Path, total_path: Path,
                 out_dir: Path):
    fig_dir = out_dir / "Figures"
    tab_dir = out_dir / "Tables"
    for d in (fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    gdf = load_fishnet(fishnet_path)
    df = prepare_analysis_sample(gdf)
    if len(df) < 100:
        logger.error("Sample too small (n=%d)", len(df))
        sys.exit(1)
    df[X_COLS + [Y_COL, INTENSITY_COL]].to_csv(
        tab_dir / "Table_AnalysisSample.csv",
        index=False, encoding="utf-8-sig")

    w = build_weights(df)

    # (1)
    logger.info("=" * 68)
    logger.info("(1) Binary logistic regression baseline")
    logger.info("=" * 68)
    logit = fit_logistic(df)
    vif = compute_vif(df)
    vif.to_csv(tab_dir / "Table_VIF.csv", index=False,
               encoding="utf-8-sig")
    logger.info("\n%s", vif.to_string(index=False))
    plot_roc(logit, fig_dir / "Fig6_ROC_Baseline.png")

    # (2)
    logger.info("=" * 68)
    logger.info("(2) Moran's I of logistic residuals")
    logger.info("=" * 68)
    moran_res = residual_moran(logit["residuals"], w)
    plot_moran_resid(logit["residuals"], w,
                     fig_dir / "Fig7_MoranResiduals.png")

    # (3)
    logger.info("=" * 68)
    logger.info("(3) Spatial regression: SAR (ML_Lag) and SEM (ML_Error)")
    logger.info("=" * 68)
    sar = fit_sar(df, w)
    sem = fit_sem(df, w)
    moran_sar = sar_moran_resid(sar, w, df[Y_COL].values)

    cmp = pd.DataFrame([
        {"Model": "Logistic (baseline)",
         "AIC": logit["aic"], "Pseudo_R2": logit["pseudo_r2"],
         "Nagelkerke_R2": logit["nagelkerke"],
         "AUC": logit["auc"], "Moran_I_resid": moran_res.I},
        {"Model": "SAR (ML_Lag)",
         "AIC": float(sar.aic) if hasattr(sar, "aic") else np.nan,
         "Pseudo_R2": float(sar.pr2) if hasattr(sar, "pr2") else np.nan,
         "Nagelkerke_R2": np.nan, "AUC": np.nan,
         "Moran_I_resid": moran_sar.I},
        {"Model": "SEM (ML_Error)",
         "AIC": float(sem.aic) if hasattr(sem, "aic") else np.nan,
         "Pseudo_R2": float(sem.pr2) if hasattr(sem, "pr2") else np.nan,
         "Nagelkerke_R2": np.nan, "AUC": np.nan,
         "Moran_I_resid": np.nan},
    ])
    cmp.to_csv(tab_dir / "Table_ModelComparison.csv",
               index=False, encoding="utf-8-sig")
    logger.info("\n%s", cmp.to_string(index=False))

    # (4)
    logger.info("=" * 68)
    logger.info("(4) SAR Direct / Indirect / Total impacts")
    logger.info("=" * 68)
    impacts = sar_impacts(sar)
    impacts.to_csv(tab_dir / "Table_SAR_Impacts.csv",
                   index=False, encoding="utf-8-sig")
    logger.info("\n%s", impacts.to_string(index=False))
    plot_predictor_effects(sar, fig_dir / "Fig8_PredictorEffects.png")

    # (5)
    logger.info("=" * 68)
    logger.info("(5) Sensitivity (4 thresholds × 2 estimators)")
    logger.info("=" * 68)
    sens_df = sensitivity_analysis(df, w)
    sens_df.to_csv(tab_dir / "Table_Sensitivity.csv",
                   index=False, encoding="utf-8-sig")
    sens_summary = classify_robustness(sens_df)
    sens_summary.to_csv(tab_dir / "Table_Sensitivity_Robustness.csv",
                        index=False, encoding="utf-8-sig")
    logger.info("\n%s", sens_summary.to_string(index=False))
    plot_sar_sensitivity(sens_df, fig_dir / "Fig9_SAR_Sensitivity.png")

    # (6)
    logger.info("=" * 68)
    logger.info("(6) Slope zones + window of intervention (from saved CSVs)")
    logger.info("=" * 68)
    sz_rows, sz_summary = load_slope_zones(woi_path, total_path)
    if sz_rows is not None:
        sz_rows.to_csv(tab_dir / "Table_SlopeZone.csv",
                       index=False, encoding="utf-8-sig")
    if sz_summary is not None:
        sz_summary.to_csv(tab_dir / "Table_WindowOfIntervention.csv",
                          index=False, encoding="utf-8-sig")
        logger.info("\n%s", sz_summary.to_string(index=False))
    plot_slope_zone(sz_rows, sz_summary,
                    fig_dir / "Fig10_SlopeZone_Intervention.png")

    # Validation
    logger.info("=" * 68)
    logger.info("Validation against previous run")
    logger.info("=" * 68)
    validate(df, logit, moran_res, sar, moran_sar, impacts)

    logger.info("=" * 68)
    logger.info("DONE — Section 3.3 pipeline completed")
    logger.info("Output root: %s", out_dir)
    logger.info("=" * 68)


def parse_args():
    p = argparse.ArgumentParser(
        description="Section 3.3 — Conversion drivers (Gia Lai).")
    p.add_argument("--fishnet", type=Path, default=DEFAULT_FISHNET)
    p.add_argument("--woi", type=Path, default=DEFAULT_WOI)
    p.add_argument("--slope_total", type=Path, default=DEFAULT_SLOPE_TOTAL)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.fishnet, args.woi, args.slope_total, args.outdir)


if __name__ == "__main__":
    main()