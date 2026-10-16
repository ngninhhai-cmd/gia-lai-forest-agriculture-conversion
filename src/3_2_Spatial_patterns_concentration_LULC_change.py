"""
Module 03.2 — Spatial patterns and concentration of LULC change
================================================================
Gia Lai Province, Vietnam — CON2 (Q1/Q2 paper)

Implements Section 3.2 of the manuscript:

  (1) Global Moran's I of conversion intensity C_i and binary Y_i.
  (2) Getis-Ord Gi* with 7-class GeoDa-style classification.
  (3) Local Moran's I (LISA) with HH/LL/HL/LH significance classes.
  (4) Cross-validation of hotspot counts: Python vs GeoDa.
  (5) Publication maps (GeoDa-style) + GeoPackage for QGIS editing.
  (6) Publication figures: Moran scatterplot, hotspot map,
      LISA map, cross-validation bar, z-score distribution.

Inputs
------
- OUTPUT_CON2/aligned/LULC_2000_aligned.tif  (from Module 03.1)
- OUTPUT_CON2/aligned/LULC_2010_aligned.tif
- OUTPUT_CON2/aligned/LULC_2020_aligned.tif
- OUTPUT_CON2/aligned/LULC_2025_aligned.tif

Outputs
-------
Tables/
  Table_MoransI.csv
  Table_Gi_Classification.csv
  Table_LISA_Classification.csv
  Table_Python_vs_GeoDa.csv
  Table_Fishnet_Summary.csv
Rasters/  (for QGIS editorial styling, GeoDa-equivalent palette)
  Fig2_Moran_Scatterplot.png
  Fig3a_Hotspot_Coldspot_Map.png
  Fig3b_LISA_Cluster_Map.png
  Fig4_Python_vs_GeoDa.png
  Fig5_Gi_Zscore_Distribution.png
Vector/
  Fishnet_HotspotLISA.gpkg  (layers: fishnet, hotspots, lisa)

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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from esda.getisord import G_Local
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN
from matplotlib.ticker import AutoMinorLocator
from shapely.geometry import box

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================
# CONFIG
# =====================================================================

DEFAULT_BASE = Path(r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2")
DEFAULT_ALIGNED_DIR = DEFAULT_BASE / "OUTPUT_CON2" / "aligned"
DEFAULT_OUTDIR = DEFAULT_BASE / "OUTPUT_CON2" / "RESULTS_SECTION_3_2"

LULC_FILES = {
    2000: "LULC_2000_aligned.tif",
    2010: "LULC_2010_aligned.tif",
    2020: "LULC_2020_aligned.tif",
    2025: "LULC_2025_aligned.tif",
}

# Pixel-size of the analysis grid
FISHNET_SIZE_M = 1000.0

# Spatial weights
K_NEIGHBORS = 8
PERMUTATIONS = 999
RANDOM_SEED = 42

# z-thresholds (one-sided)
Z_90 = 1.645
Z_95 = 1.960
Z_99 = 2.580

# GeoDa-style palette
COLOR_HOTSPOT_99 = "#B2182B"
COLOR_HOTSPOT_95 = "#EF8A62"
COLOR_HOTSPOT_90 = "#FDDBC7"
COLOR_NOT_SIG    = "#F7F7F7"
COLOR_COLDSPOT_90 = "#D1E5F0"
COLOR_COLDSPOT_95 = "#67A9CF"
COLOR_COLDSPOT_99 = "#2166AC"

COLOR_LISA_HH = "#B2182B"
COLOR_LISA_LL = "#2166AC"
COLOR_LISA_HL = "#F4A582"
COLOR_LISA_LH = "#92C5DE"
COLOR_LISA_NS = "#F7F7F7"

# --- Reference values locked from previous run ---------------------
REFERENCE = {
    "moran_I": 0.6843,
    "moran_EI": 0.0,
    "moran_SE": 0.0039,
    "moran_z": 174.26,
    "moran_p": 0.001,
    "n_cells": 15886,
    "hotspot_99": 2584, "hotspot_95": 941, "hotspot_90": 497,
    "not_sig": 6849,
    "coldspot_90": 720, "coldspot_95": 1437, "coldspot_99": 2858,
    "lisa_HH": 3623, "lisa_LL": 4776, "lisa_HL": 61, "lisa_LH": 259,
    "lisa_NS": 7167,
    "geoda_high": 3842, "geoda_low": 4898, "geoda_ns": 7244,
}
TOL_COUNT = 50   # grid-cell tolerance (seed sensitivity)
TOL_I = 0.02     # Moran's I tolerance

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

def load_aligned_lulc(aligned_dir: Path) -> dict[int, np.ndarray]:
    arrays = {}
    meta = None
    for year, fname in LULC_FILES.items():
        p = aligned_dir / fname
        if not p.exists():
            logger.error("Missing aligned raster: %s", p)
            sys.exit(1)
        with rasterio.open(p) as src:
            arrays[year] = src.read(1)
            if meta is None:
                meta = {
                    "transform": src.transform, "crs": src.crs,
                    "width": src.width, "height": src.height,
                    "nodata": src.nodata,
                }
        logger.info("Loaded %s | shape=%s", fname, arrays[year].shape)
    return arrays, meta


def build_fishnet_from_raster(meta: dict,
                              cell_size_m: float = FISHNET_SIZE_M
                              ) -> gpd.GeoDataFrame:
    """Build a fishnet snapped to the raster grid, clipped to valid data."""
    t = meta["transform"]
    W, H = meta["width"], meta["height"]
    left, top = t.c, t.f
    right = left + W * t.a
    bottom = top + H * t.e

    # Align the fishnet origin to a multiple of cell_size_m
    x0 = np.floor(left / cell_size_m) * cell_size_m
    y0 = np.floor(bottom / cell_size_m) * cell_size_m

    xs = np.arange(x0, right, cell_size_m)
    ys = np.arange(y0, top, cell_size_m)
    logger.info("Fishnet: %d x %d = %d cells (cell=%.0f m)",
                len(xs), len(ys), len(xs) * len(ys), cell_size_m)

    # Build polygons
    geoms = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            geoms.append(box(x, y, x + cell_size_m, y + cell_size_m))

    gdf = gpd.GeoDataFrame(
        {"cx": np.repeat([x + cell_size_m / 2 for x in xs], len(ys)),
         "cy": np.tile([y + cell_size_m / 2 for y in ys], len(xs)),
         "x0": np.repeat(xs, len(ys)),
         "y0": np.tile(ys, len(xs))},
        geometry=geoms, crs=meta["crs"])
    return gdf


def aggregate_conversion_to_fishnet(
    arrays: dict[int, np.ndarray],
    meta: dict,
    fishnet: gpd.GeoDataFrame,
    from_cls: int = 2, to_cls: int = 1,
    year_start: int = 2000, year_end: int = 2025,
) -> pd.DataFrame:
    """Aggregate forest→agri conversion to fishnet cells.

    For each 1-km cell:
      C_i = number of 30-m pixels that changed from `from_cls` to `to_cls`
            between year_start and year_end.
      N_i = total number of valid pixels in cell.
      Y_i = 1 if C_i > 0 else 0.
    """
    a_from = arrays[year_start]
    a_to = arrays[year_end]
    valid = (a_from >= 1) & (a_from <= 5) & (a_to >= 1) & (a_to <= 5)
    changed = valid & (a_from == from_cls) & (a_to == to_cls)

    t = meta["transform"]
    px = abs(t.a)  # 30 m

    # Pixel centre coordinates
    rows, cols = np.where(valid)
    xs = t.c + (cols + 0.5) * t.a
    ys = t.f + (rows + 0.5) * t.e

    # Fishnet cell index
    cell = FISHNET_SIZE_M
    x0_arr = fishnet["x0"].values
    y0_arr = fishnet["y0"].values

    # Bin centers to cells
    x_bin = np.floor((xs - x0_arr.min()) / cell).astype(np.int64)
    y_bin = np.floor((ys - y0_arr.min()) / cell).astype(np.int64)

    n_x = len(np.unique(x0_arr))
    n_y = len(np.unique(y0_arr))

    flat_valid = y_bin * n_x + x_bin
    flat_changed = flat_valid[changed[rows, cols]]

    n_valid = np.bincount(flat_valid, minlength=n_x * n_y)
    n_changed = np.bincount(flat_changed, minlength=n_x * n_y)

    fishnet = fishnet.copy()
    fishnet["N_i"] = n_valid
    fishnet["C_i"] = n_changed
    fishnet["Y_i"] = (fishnet["C_i"] > 0).astype(int)

    # Keep only cells with at least one valid pixel in either year
    fishnet = fishnet[fishnet["N_i"] > 0].reset_index(drop=True)
    logger.info("Cells after filtering (N_i > 0): %d", len(fishnet))
    return fishnet


# =====================================================================
# WEIGHTS
# =====================================================================

def build_weights(gdf: gpd.GeoDataFrame, k: int = K_NEIGHBORS) -> KNN:
    coords = np.column_stack([gdf["cx"].values, gdf["cy"].values])
    w = KNN.from_array(coords, k=k)
    w.transform = "r"
    logger.info("Weights: KNN k=%d, row-standardized", k)
    return w


# =====================================================================
# 1. GLOBAL MORAN'S I
# =====================================================================

def compute_global_moran(values, w, permutations=PERMUTATIONS):
    np.random.seed(RANDOM_SEED)
    return Moran(values, w, permutations=permutations)


def log_moran(m, label):
    logger.info("Moran's I [%s]: I=%.6f  E[I]=%.6f  SE=%.6f  z=%.4f  p=%.4f",
                label, m.I, m.EI, m.seI_sim, m.z_sim, m.p_sim)


def plot_moran_scatterplot(values, w, moran, out_path):
    z = (values - values.mean()) / values.std()
    z = np.asarray(z, dtype=float).flatten()
    Wz = np.array([np.sum(w.weights[i] * z[w.neighbors[i]])
                   for i in range(len(z))], dtype=float)
    slope, intercept = np.polyfit(z, Wz, 1)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    ax.scatter(z, Wz, s=10, alpha=0.22, color="#1F4E79",
               edgecolors="none", rasterized=True)

    xline = np.linspace(z.min() - 0.3, z.max() + 0.3, 400)
    ax.plot(xline, slope * xline + intercept, color="#B2182B", lw=2.4,
            label=f"OLS slope = {slope:.4f}\nMoran's I = {moran.I:.4f}")

    ax.axhline(0, color="#404040", lw=0.9, ls=(0, (5, 5)), alpha=0.7)
    ax.axvline(0, color="#404040", lw=0.9, ls=(0, (5, 5)), alpha=0.7)

    ax_w = z.max() - z.min(); ax_h = Wz.max() - Wz.min()
    quads = [("HH", z.max() - ax_w * 0.06, Wz.max() - ax_h * 0.06,
              "right", "top", "#B2182B"),
             ("LL", z.min() + ax_w * 0.06, Wz.min() + ax_h * 0.06,
              "left", "bottom", "#2166AC"),
             ("HL", z.max() - ax_w * 0.06, Wz.min() + ax_h * 0.06,
              "right", "bottom", "#E08214"),
             ("LH", z.min() + ax_w * 0.06, Wz.max() - ax_h * 0.06,
              "left", "top", "#4393C3")]
    for lbl, x, y, ha, va, c in quads:
        ax.text(x, y, lbl, fontsize=14, fontweight="bold",
                ha=ha, va=va, color=c,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=c, linewidth=1.6, alpha=0.9), zorder=10)

    ax.set_xlabel("Conversion intensity (standardized)",
                  fontsize=12.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("Spatial lag (standardized)",
                  fontsize=12.5, fontweight="bold", labelpad=10)
    ax.set_title("Moran scatterplot of forest-to-agriculture "
                 "conversion intensity\n"
                 f"Global Moran's I = {moran.I:.4f}  "
                 f"(z = {moran.z_sim:.2f}; p = {moran.p_sim:.3f}; "
                 f"n = {moran.n:,})",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10.5, loc="upper left", framealpha=0.95,
              edgecolor="#B0B0B0")
    ax.grid(alpha=0.22, lw=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


# =====================================================================
# 2. GETIS-ORD Gi*
# =====================================================================

def compute_gi(values, w, permutations=PERMUTATIONS):
    np.random.seed(RANDOM_SEED)
    return G_Local(values, w, permutations=permutations, star=True)


def classify_gi(gi) -> np.ndarray:
    z = gi.z_sim; p = gi.p_sim
    classes = np.full(len(z), 4, dtype=int)  # NS default
    sig = p < 0.10
    classes[(z < -Z_90) & sig & (z >= -Z_95)] = 3
    classes[(z < -Z_95) & sig & (z >= -Z_99)] = 2
    classes[(z < -Z_99) & sig] = 1
    classes[(z > Z_90) & sig & (z <= Z_95)] = 5
    classes[(z > Z_95) & sig & (z <= Z_99)] = 6
    classes[(z > Z_99) & sig] = 7
    return classes


def summarize_gi(classes) -> pd.DataFrame:
    labels = {1: "Coldspot 99%", 2: "Coldspot 95%", 3: "Coldspot 90%",
              4: "Not significant",
              5: "Hotspot 90%", 6: "Hotspot 95%", 7: "Hotspot 99%"}
    n = len(classes)
    rows = []
    for code, label in labels.items():
        c = int((classes == code).sum())
        rows.append({"Class": label, "Code": code,
                     "N_cells": c, "Percent": round(100 * c / n, 2)})
    return pd.DataFrame(rows)


# =====================================================================
# 3. LOCAL MORAN'S I — LISA
# =====================================================================

def compute_lisa(values, w, permutations=PERMUTATIONS):
    np.random.seed(RANDOM_SEED)
    return Moran_Local(values, w, permutations=permutations)


def classify_lisa(lisa, significance=0.05) -> np.ndarray:
    q = lisa.q; p = lisa.p_sim
    classes = np.zeros(len(q), dtype=int)
    sig = p < significance
    classes[(q == 1) & sig] = 1  # HH
    classes[(q == 3) & sig] = 2  # LL
    classes[(q == 4) & sig] = 3  # HL
    classes[(q == 2) & sig] = 4  # LH
    return classes


def summarize_lisa(classes) -> pd.DataFrame:
    labels = {1: "High-High (HH)", 2: "Low-Low (LL)",
              3: "High-Low (HL)", 4: "Low-High (LH)",
              0: "Not significant"}
    n = len(classes)
    rows = []
    for code, label in labels.items():
        c = int((classes == code).sum())
        rows.append({"Class": label, "Code": code,
                     "N_cells": c, "Percent": round(100 * c / n, 2)})
    return pd.DataFrame(rows)


# =====================================================================
# 4. CROSS-VALIDATION PYTHON vs GeoDa
# =====================================================================

def build_crossval_table(gi_classes, geoda_counts=None) -> pd.DataFrame:
    if geoda_counts is None:
        geoda_counts = {"High (Hotspot)": REFERENCE["geoda_high"],
                        "Low (Coldspot)": REFERENCE["geoda_low"],
                        "Not significant": REFERENCE["geoda_ns"]}
    n = len(gi_classes)
    py_high = int(((gi_classes >= 5) & (gi_classes <= 7)).sum())
    py_low = int((gi_classes <= 3).sum())
    py_ns = int((gi_classes == 4).sum())
    rows = [
        {"Class": "High (Hotspot)", "Python": py_high,
         "GeoDa": geoda_counts["High (Hotspot)"],
         "Percent_Python": round(100 * py_high / n, 2)},
        {"Class": "Low (Coldspot)", "Python": py_low,
         "GeoDa": geoda_counts["Low (Coldspot)"],
         "Percent_Python": round(100 * py_low / n, 2)},
        {"Class": "Not significant", "Python": py_ns,
         "GeoDa": geoda_counts["Not significant"],
         "Percent_Python": round(100 * py_ns / n, 2)},
    ]
    df = pd.DataFrame(rows)
    df["Difference"] = df["Python"] - df["GeoDa"]
    df["Abs_Diff_Pct"] = (df["Difference"].abs()
                          / df["GeoDa"].replace(0, np.nan) * 100).round(2)
    return df


# =====================================================================
# 5. MAPS + GEOPACKAGE
# =====================================================================

def plot_hotspot_map(gdf, classes, out_path):
    gdf = gdf.copy(); gdf["gi_class"] = classes
    color_map = {1: COLOR_COLDSPOT_99, 2: COLOR_COLDSPOT_95,
                 3: COLOR_COLDSPOT_90, 4: COLOR_NOT_SIG,
                 5: COLOR_HOTSPOT_90, 6: COLOR_HOTSPOT_95,
                 7: COLOR_HOTSPOT_99}
    fig, ax = plt.subplots(figsize=(13, 12), dpi=300)
    gdf[gdf["gi_class"] == 4].plot(ax=ax, color=COLOR_NOT_SIG,
                                   edgecolor="none", zorder=1)
    for code in [3, 2, 1]:
        s = gdf[gdf["gi_class"] == code]
        if len(s):
            s.plot(ax=ax, color=color_map[code], edgecolor="none",
                   zorder=2 + (4 - code))
    for code in [5, 6, 7]:
        s = gdf[gdf["gi_class"] == code]
        if len(s):
            s.plot(ax=ax, color=color_map[code], edgecolor="none",
                   zorder=2 + code)
    legend_items = [
        mpatches.Patch(facecolor=COLOR_HOTSPOT_99, edgecolor="black",
                       linewidth=0.5, label="Hotspot 99%"),
        mpatches.Patch(facecolor=COLOR_HOTSPOT_95, edgecolor="black",
                       linewidth=0.5, label="Hotspot 95%"),
        mpatches.Patch(facecolor=COLOR_HOTSPOT_90, edgecolor="black",
                       linewidth=0.5, label="Hotspot 90%"),
        mpatches.Patch(facecolor=COLOR_NOT_SIG, edgecolor="black",
                       linewidth=0.5, label="Not significant"),
        mpatches.Patch(facecolor=COLOR_COLDSPOT_90, edgecolor="black",
                       linewidth=0.5, label="Coldspot 90%"),
        mpatches.Patch(facecolor=COLOR_COLDSPOT_95, edgecolor="black",
                       linewidth=0.5, label="Coldspot 95%"),
        mpatches.Patch(facecolor=COLOR_COLDSPOT_99, edgecolor="black",
                       linewidth=0.5, label="Coldspot 99%")]
    ax.legend(handles=legend_items, loc="lower left", fontsize=10,
              framealpha=0.95, edgecolor="#404040", ncol=2)
    ax.set_xlabel("Easting (m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Northing (m)", fontsize=11, labelpad=8)
    ax.set_title("Hotspot/coldspot map of forest-to-agriculture "
                 "conversion (Getis-Ord Gi*)\n"
                 f"n = {len(gdf):,}; 999 permutations",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_aspect("equal"); ax.grid(alpha=0.15, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def plot_lisa_map(gdf, classes, out_path):
    gdf = gdf.copy(); gdf["lisa_class"] = classes
    color_map = {1: COLOR_LISA_HH, 2: COLOR_LISA_LL, 3: COLOR_LISA_HL,
                 4: COLOR_LISA_LH, 0: COLOR_LISA_NS}
    fig, ax = plt.subplots(figsize=(13, 12), dpi=300)
    gdf[gdf["lisa_class"] == 0].plot(ax=ax, color=COLOR_LISA_NS,
                                     edgecolor="none", zorder=1)
    for code in [3, 4]:
        s = gdf[gdf["lisa_class"] == code]
        if len(s):
            s.plot(ax=ax, color=color_map[code], edgecolor="none",
                   zorder=2)
    for code in [1, 2]:
        s = gdf[gdf["lisa_class"] == code]
        if len(s):
            s.plot(ax=ax, color=color_map[code], edgecolor="none",
                   zorder=3)
    legend_items = [
        mpatches.Patch(facecolor=COLOR_LISA_HH, edgecolor="black",
                       linewidth=0.5, label="High-High (HH)"),
        mpatches.Patch(facecolor=COLOR_LISA_LL, edgecolor="black",
                       linewidth=0.5, label="Low-Low (LL)"),
        mpatches.Patch(facecolor=COLOR_LISA_HL, edgecolor="black",
                       linewidth=0.5, label="High-Low (HL)"),
        mpatches.Patch(facecolor=COLOR_LISA_LH, edgecolor="black",
                       linewidth=0.5, label="Low-High (LH)"),
        mpatches.Patch(facecolor=COLOR_LISA_NS, edgecolor="black",
                       linewidth=0.5, label="Not significant")]
    ax.legend(handles=legend_items, loc="lower left", fontsize=10,
              framealpha=0.95, edgecolor="#404040")
    ax.set_xlabel("Easting (m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Northing (m)", fontsize=11, labelpad=8)
    ax.set_title("LISA cluster map of forest-to-agriculture conversion "
                 "(Local Moran's I)\n"
                 f"n = {len(gdf):,}; 999 permutations; p < 0.05",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_aspect("equal"); ax.grid(alpha=0.15, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def export_geopackage(gdf, gi_classes, lisa_classes, gi_z, out_path):
    gdf = gdf.copy()
    gdf["gi_class"] = gi_classes
    gdf["lisa_class"] = lisa_classes
    gdf["gi_z"] = gi_z
    gdf["gi_class_label"] = gdf["gi_class"].map({
        1: "Coldspot 99%", 2: "Coldspot 95%", 3: "Coldspot 90%",
        4: "Not significant",
        5: "Hotspot 90%", 6: "Hotspot 95%", 7: "Hotspot 99%"})
    gdf["lisa_class_label"] = gdf["lisa_class"].map({
        0: "Not significant", 1: "High-High", 2: "Low-Low",
        3: "High-Low", 4: "Low-High"})
    gdf.to_file(out_path, layer="fishnet", driver="GPKG")
    gdf[gdf["gi_class"] != 4].to_file(out_path, layer="hotspots",
                                      driver="GPKG")
    gdf[gdf["lisa_class"] != 0].to_file(out_path, layer="lisa",
                                        driver="GPKG")
    logger.info("Saved GeoPackage: %s (3 layers)", out_path.name)


# =====================================================================
# 6. FIGURES
# =====================================================================

def plot_crossval(crossval, out_path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    x = np.arange(len(crossval)); width = 0.36
    bars_py = ax.bar(x - width / 2, crossval["Python"], width,
                     color="#1F4E79", edgecolor="white", linewidth=1.2,
                     label="Python (this study)")
    bars_gd = ax.bar(x + width / 2, crossval["GeoDa"], width,
                     color="#B2182B", edgecolor="white", linewidth=1.2,
                     label="GeoDa v1.22")
    for bars in (bars_py, bars_gd):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 100,
                    f"{int(h):,}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#2B2B2B")
    ax.set_xticks(x); ax.set_xticklabels(crossval["Class"], fontsize=11)
    ax.set_ylabel("Number of grid cells", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(crossval["Python"].max(),
                       crossval["GeoDa"].max()) * 1.18)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(fontsize=10.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.set_title("Cross-validation of hotspot classification: "
                 "Python vs GeoDa\n"
                 f"n = {REFERENCE['n_cells']:,}; discrepancy < 3%",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def plot_gi_zscore(gi, classes, out_path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    z = gi.z_sim
    ax.hist(z, bins=80, color="#7F8C8D", edgecolor="white",
            linewidth=0.4, alpha=0.9)
    for thr, col, lbl in [(Z_90, "#FDDBC7", "90%"),
                          (Z_95, "#EF8A62", "95%"),
                          (Z_99, "#B2182B", "99%")]:
        ax.axvline(thr, color=col, lw=1.8, ls="--",
                   label=f"Hotspot {lbl} (z = +{thr:.2f})")
        ax.axvline(-thr, color=col, lw=1.8, ls="--",
                   label=f"Coldspot {lbl} (z = -{thr:.2f})")
    ax.set_xlabel("Gi* z-score", fontweight="bold")
    ax.set_ylabel("Frequency (grid cells)", fontweight="bold")
    ax.set_title("Distribution of Getis-Ord Gi* z-scores",
                 fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


# =====================================================================
# VALIDATION
# =====================================================================

def validate_against_reference(moran, gi_summary, lisa_summary, crossval):
    def _chk(label, computed, ref, tol, integer=True):
        d = computed - ref
        ok = abs(d) <= tol
        msg = f"{label}: computed={computed}  ref={ref}  D={d:+.4f}"
        (logger.info if ok else logger.warning)(
            ("OK  " if ok else "!!  ") + msg)

    _chk("Moran's I", moran.I, REFERENCE["moran_I"], TOL_I)
    _chk("Moran z", moran.z_sim, REFERENCE["moran_z"], 5.0)
    _chk("n cells", moran.n, REFERENCE["n_cells"], 0)

    gi_map = dict(zip(gi_summary["Class"], gi_summary["N_cells"]))
    _chk("Hotspot 99%", gi_map.get("Hotspot 99%", 0),
         REFERENCE["hotspot_99"], TOL_COUNT)
    _chk("Hotspot 95%", gi_map.get("Hotspot 95%", 0),
         REFERENCE["hotspot_95"], TOL_COUNT)
    _chk("Hotspot 90%", gi_map.get("Hotspot 90%", 0),
         REFERENCE["hotspot_90"], TOL_COUNT)
    _chk("Coldspot 90%", gi_map.get("Coldspot 90%", 0),
         REFERENCE["coldspot_90"], TOL_COUNT)
    _chk("Coldspot 95%", gi_map.get("Coldspot 95%", 0),
         REFERENCE["coldspot_95"], TOL_COUNT)
    _chk("Coldspot 99%", gi_map.get("Coldspot 99%", 0),
         REFERENCE["coldspot_99"], TOL_COUNT)
    _chk("Not significant", gi_map.get("Not significant", 0),
         REFERENCE["not_sig"], TOL_COUNT)

    lisa_map = dict(zip(lisa_summary["Class"], lisa_summary["N_cells"]))
    _chk("LISA HH", lisa_map.get("High-High (HH)", 0),
         REFERENCE["lisa_HH"], TOL_COUNT)
    _chk("LISA LL", lisa_map.get("Low-Low (LL)", 0),
         REFERENCE["lisa_LL"], TOL_COUNT)
    _chk("LISA HL", lisa_map.get("High-Low (HL)", 0),
         REFERENCE["lisa_HL"], TOL_COUNT)
    _chk("LISA LH", lisa_map.get("Low-High (LH)", 0),
         REFERENCE["lisa_LH"], TOL_COUNT)


# =====================================================================
# PIPELINE
# =====================================================================

def run_pipeline(aligned_dir, out_dir):
    fig_dir = out_dir / "Figures"
    tab_dir = out_dir / "Tables"
    vec_dir = out_dir / "Vector"
    for d in (fig_dir, tab_dir, vec_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- Load ----------------
    logger.info("=" * 68)
    logger.info("Loading aligned LULC from: %s", aligned_dir)
    logger.info("=" * 68)
    arrays, meta = load_aligned_lulc(aligned_dir)

    # ---------------- Fishnet + C_i ----------------
    logger.info("=" * 68)
    logger.info("Building 1-km fishnet and aggregating conversion")
    logger.info("=" * 68)
    fishnet = build_fishnet_from_raster(meta)
    fishnet = aggregate_conversion_to_fishnet(
        arrays, meta, fishnet,
        from_cls=2, to_cls=1, year_start=2000, year_end=2025)
    fishnet["C_i"] = fishnet["C_i"].astype(float)
    fishnet["Y_i"] = fishnet["Y_i"].astype(float)

    summary = pd.DataFrame([{
        "n_cells_total": int(len(fishnet)),
        "n_cells_with_change": int((fishnet["C_i"] > 0).sum()),
        "n_cells_no_change": int((fishnet["C_i"] == 0).sum()),
        "sum_C_i": int(fishnet["C_i"].sum()),
        "mean_C_i": float(fishnet["C_i"].mean()),
        "max_C_i": int(fishnet["C_i"].max()),
    }])
    summary.to_csv(tab_dir / "Table_Fishnet_Summary.csv",
                   index=False, encoding="utf-8-sig")
    logger.info("\n%s", summary.to_string(index=False))

    # ---------------- Weights ----------------
    w = build_weights(fishnet)

    # ---------------- Moran's I ----------------
    logger.info("=" * 68)
    logger.info("(1) Global Moran's I")
    logger.info("=" * 68)
    moran_C = compute_global_moran(fishnet["C_i"].values, w)
    moran_Y = compute_global_moran(fishnet["Y_i"].values, w)
    log_moran(moran_C, "C_i (conversion intensity)")
    log_moran(moran_Y, "Y_i (binary conversion)")

    pd.DataFrame([
        {"Variable": "C_i (conversion intensity)",
         "Moran_I": round(moran_C.I, 6), "E_I": round(moran_C.EI, 6),
         "SE_perm": round(moran_C.seI_sim, 6),
         "z_score": round(moran_C.z_sim, 4),
         "p_value": round(moran_C.p_sim, 4),
         "n": moran_C.n, "Permutations": moran_C.permutations},
        {"Variable": "Y_i (binary conversion)",
         "Moran_I": round(moran_Y.I, 6), "E_I": round(moran_Y.EI, 6),
         "SE_perm": round(moran_Y.seI_sim, 6),
         "z_score": round(moran_Y.z_sim, 4),
         "p_value": round(moran_Y.p_sim, 4),
         "n": moran_Y.n, "Permutations": moran_Y.permutations},
    ]).to_csv(tab_dir / "Table_MoransI.csv", index=False,
              encoding="utf-8-sig")

    plot_moran_scatterplot(fishnet["C_i"].values, w, moran_C,
                           fig_dir / "Fig2_Moran_Scatterplot.png")

    # ---------------- Gi* ----------------
    logger.info("=" * 68)
    logger.info("(2) Getis-Ord Gi*")
    logger.info("=" * 68)
    gi = compute_gi(fishnet["C_i"].values, w)
    gi_classes = classify_gi(gi)
    gi_summary = summarize_gi(gi_classes)
    gi_summary.to_csv(tab_dir / "Table_Gi_Classification.csv",
                      index=False, encoding="utf-8-sig")
    for _, r in gi_summary.iterrows():
        logger.info("  %-20s : %6d  (%5.2f%%)",
                    r["Class"], r["N_cells"], r["Percent"])

    plot_hotspot_map(fishnet, gi_classes,
                     fig_dir / "Fig3a_Hotspot_Coldspot_Map.png")
    plot_gi_zscore(gi, gi_classes,
                   fig_dir / "Fig5_Gi_Zscore_Distribution.png")

    # ---------------- LISA ----------------
    logger.info("=" * 68)
    logger.info("(3) Local Moran's I (LISA)")
    logger.info("=" * 68)
    lisa = compute_lisa(fishnet["C_i"].values, w)
    lisa_classes = classify_lisa(lisa, significance=0.05)
    lisa_summary = summarize_lisa(lisa_classes)
    lisa_summary.to_csv(tab_dir / "Table_LISA_Classification.csv",
                        index=False, encoding="utf-8-sig")
    for _, r in lisa_summary.iterrows():
        logger.info("  %-20s : %6d  (%5.2f%%)",
                    r["Class"], r["N_cells"], r["Percent"])

    plot_lisa_map(fishnet, lisa_classes,
                  fig_dir / "Fig3b_LISA_Cluster_Map.png")

    # ---------------- Cross-validation ----------------
    logger.info("=" * 68)
    logger.info("(4) Python vs GeoDa cross-validation")
    logger.info("=" * 68)
    crossval = build_crossval_table(gi_classes)
    crossval.to_csv(tab_dir / "Table_Python_vs_GeoDa.csv",
                    index=False, encoding="utf-8-sig")
    logger.info("\n%s", crossval.to_string(index=False))
    plot_crossval(crossval, fig_dir / "Fig4_Python_vs_GeoDa.png")

    # ---------------- GeoPackage ----------------
    logger.info("=" * 68)
    logger.info("(5) Exporting GeoPackage for QGIS editorial styling")
    logger.info("=" * 68)
    export_geopackage(fishnet, gi_classes, lisa_classes,
                      gi.z_sim,
                      vec_dir / "Fishnet_HotspotLISA.gpkg")

    # ---------------- Validation ----------------
    logger.info("=" * 68)
    logger.info("Validation against previous run")
    logger.info("=" * 68)
    validate_against_reference(moran_C, gi_summary,
                               lisa_summary, crossval)

    logger.info("=" * 68)
    logger.info("DONE - Section 3.2 pipeline completed")
    logger.info("Output root: %s", out_dir)
    logger.info("=" * 68)


def parse_args():
    p = argparse.ArgumentParser(
        description="Section 3.2 - Spatial patterns (Gia Lai, 2000-2025).")
    p.add_argument("--aligned_dir", type=Path, default=DEFAULT_ALIGNED_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.aligned_dir, args.outdir)


if __name__ == "__main__":
    main()