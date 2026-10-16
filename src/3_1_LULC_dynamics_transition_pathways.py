"""
Module 03.1 (v2) — Land-use change dynamics and transition pathways (2000–2025)
Gia Lai Province, Vietnam — CON2 (Q1/Q2 paper)

Changelog v2:
- Auto-align all rasters to LULC_2000 reference grid (nearest-neighbour).
- Aligned rasters written to OUTPUT_CON2/aligned/.
- Alignment QC table exported for Methods section.
- TOL relaxed to 20 km² to absorb sub-pixel realignment effects.

Author : <your name>
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.ticker import AutoMinorLocator
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.measure import label as sk_label, regionprops

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================
# CONFIG
# =====================================================================

DEFAULT_BASE = Path(r"D:\8_PHD. CANDIDATE\PhD_GIS\CON2")
DEFAULT_RASTER_DIR = DEFAULT_BASE / "OUTPUT_CON2"
DEFAULT_ALIGNED_DIR = DEFAULT_RASTER_DIR / "aligned"
DEFAULT_OUTDIR = DEFAULT_BASE / "OUTPUT_CON2" / "RESULTS_SECTION_3_1"

LULC_FILES = {
    2000: "LULC_2000_3406.tif",
    2010: "LULC_2010_3406.tif",
    2020: "LULC_2020_3406.tif",
    2025: "LULC_2025_3406.tif",
}

CLASS_NAMES = {1: "Agriculture", 2: "Forest", 3: "Built-up",
               4: "Water", 5: "Other"}
CLASS_COLORS = {1: "#F4D03F", 2: "#1B5E20", 3: "#C0392B",
                4: "#2E86C1", 5: "#95A5A6"}

PHASES = [(2000, 2010), (2010, 2020), (2020, 2025), (2000, 2025)]

REFERENCE = {
    "forest_km2_2000": 9694.62,
    "forest_km2_2025": 5574.98,
    "agri_km2_2000": 4910.24,
    "agri_km2_2025": 8943.97,
    "forest_loss_2000_2010_km2": -2645.71,
    "forest_gain_2020_2025_km2": +26.06,
}
TOL = 20.0

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
# ALIGNMENT
# =====================================================================

def align_rasters(raster_dir: Path, aligned_dir: Path) -> dict[int, Path]:
    aligned_dir.mkdir(parents=True, exist_ok=True)

    ref_year = 2000
    ref_path = raster_dir / LULC_FILES[ref_year]

    with rasterio.open(ref_path) as ref:
        ref_meta = {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "nodata": ref.nodata if ref.nodata is not None else -9999.0,
        }
        logger.info("Reference grid: %s  (%d x %d, %s)",
                    ref_path.name, ref.width, ref.height, ref.crs)

    aligned_paths: dict[int, Path] = {}
    qc_rows = []

    for year, fname in LULC_FILES.items():
        src_path = raster_dir / fname
        out_path = aligned_dir / f"LULC_{year}_aligned.tif"

        with rasterio.open(src_path) as src:
            same_grid = (
                src.width == ref_meta["width"]
                and src.height == ref_meta["height"]
                and src.transform == ref_meta["transform"]
                and src.crs == ref_meta["crs"]
            )
            if same_grid:
                logger.info("%s — already aligned", fname)
                arr_out = src.read(1)
            else:
                logger.info("%s — realigning", fname)
                dst = np.full((ref_meta["height"], ref_meta["width"]),
                              ref_meta["nodata"], dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_meta["transform"],
                    dst_crs=ref_meta["crs"],
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata if src.nodata is not None else -9999.0,
                    dst_nodata=ref_meta["nodata"],
                )
                arr_out = dst

        valid_mask = (arr_out >= 1) & (arr_out <= 5)
        n_valid = int(valid_mask.sum())
        n_nodata = int((arr_out == ref_meta["nodata"]).sum())
        n_other = int(arr_out.size - n_valid - n_nodata)

        out_meta = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "width": ref_meta["width"], "height": ref_meta["height"],
            "crs": ref_meta["crs"], "transform": ref_meta["transform"],
            "nodata": ref_meta["nodata"], "compress": "LZW",
        }
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(arr_out.astype(np.float32), 1)

        aligned_paths[year] = out_path
        qc_rows.append({
            "Year": year, "Source_file": fname,
            "Aligned_file": out_path.name, "Ref_year": ref_year,
            "Width": ref_meta["width"], "Height": ref_meta["height"],
            "Valid_pixels": n_valid, "Nodata_pixels": n_nodata,
            "Other_pixels": n_other, "Realigned": not same_grid,
        })
        logger.info("  -> %s | valid=%d  nodata=%d  other=%d",
                    out_path.name, n_valid, n_nodata, n_other)

    pd.DataFrame(qc_rows).to_csv(
        aligned_dir / "Table_AlignmentQC.csv",
        index=False, encoding="utf-8-sig")
    logger.info("Saved QC: Table_AlignmentQC.csv")

    return aligned_paths


def load_aligned_stack(aligned_paths):
    arrays = {}
    meta = None
    for year, path in sorted(aligned_paths.items()):
        with rasterio.open(path) as src:
            arr = src.read(1)
            arrays[year] = arr
            if meta is None:
                meta = {
                    "transform": src.transform, "crs": src.crs,
                    "height": src.height, "width": src.width,
                    "pixel_area_m2": src.transform.a * (-src.transform.e),
                    "nodata": src.nodata,
                }
            logger.info("Loaded %s | shape=%s", path.name, arr.shape)
    return arrays, meta


def write_geotiff(arr, meta, out_path, dtype="uint8"):
    out_meta = {
        "driver": "GTiff", "dtype": dtype, "count": 1,
        "height": meta["height"], "width": meta["width"],
        "crs": meta["crs"], "transform": meta["transform"],
        "compress": "LZW",
        "nodata": 255 if dtype == "uint8" else meta.get("nodata", -9999.0),
    }
    with rasterio.open(out_path, "w", **out_meta) as dst:
        if dtype == "uint8":
            dst.write(np.clip(arr, 0, 254).astype(np.uint8), 1)
        else:
            dst.write(arr.astype(dtype), 1)
    logger.info("Saved raster: %s", out_path.name)


# =====================================================================
# (1) AREA / TRANSITION / TURNOVER / ANNUAL RATE
# =====================================================================

def area_by_class(arr, pixel_area_m2):
    km2 = pixel_area_m2 / 1_000_000.0
    return pd.Series({CLASS_NAMES[c]: round(int(np.sum(arr == c)) * km2, 4)
                      for c in CLASS_NAMES})


def compute_area_table(arrays, pixel_area_m2):
    rows = []
    for yr, arr in sorted(arrays.items()):
        s = area_by_class(arr, pixel_area_m2); s.name = yr
        rows.append(s)
    df = pd.DataFrame(rows).T
    df.index.name = "Class"
    return df


def transition_matrix(a_from, a_to, pixel_area_m2):
    km2 = pixel_area_m2 / 1_000_000.0
    mat = np.zeros((5, 5), dtype=np.float64)
    valid = (a_from >= 1) & (a_from <= 5) & (a_to >= 1) & (a_to <= 5)
    src = a_from[valid].astype(np.int64) - 1
    dst = a_to[valid].astype(np.int64) - 1
    np.add.at(mat, (src, dst), km2)
    labels = [CLASS_NAMES[i] for i in range(1, 6)]
    return pd.DataFrame(mat, index=labels, columns=labels).round(4)


def turnover_table(mat):
    rows = []
    total_area = mat.values.sum()
    for cls in mat.index:
        gain = mat.loc[:, cls].sum() - mat.loc[cls, cls]
        loss = mat.loc[cls, :].sum() - mat.loc[cls, cls]
        net = gain - loss
        total = gain + loss
        swap = total - abs(net)
        rows.append({
            "Class": cls,
            "Initial_km2": round(mat.loc[cls, :].sum(), 4),
            "Final_km2": round(mat.loc[:, cls].sum(), 4),
            "Gain_km2": round(gain, 4), "Loss_km2": round(loss, 4),
            "Net_km2": round(net, 4), "Swap_km2": round(swap, 4),
            "Total_change_km2": round(total, 4),
            "Net_pct_of_total": round(100 * abs(net) / total_area, 4),
            "Swap_pct_of_total": round(100 * swap / total_area, 4),
        })
    return pd.DataFrame(rows)


def annual_rate_puyravaud(a1, a2, t1, t2):
    if a1 <= 0 or a2 <= 0:
        return np.nan
    return 100.0 * np.log(a2 / a1) / (t2 - t1)


def annual_rate_table(area_df):
    rows = []
    for cls in area_df.index:
        for t1, t2 in PHASES:
            a1 = area_df.loc[cls, t1]; a2 = area_df.loc[cls, t2]
            r = annual_rate_puyravaud(a1, a2, t1, t2)
            rows.append({
                "Class": cls, "Phase": f"{t1}-{t2}", "Years": t2 - t1,
                "Area_start_km2": a1, "Area_end_km2": a2,
                "Delta_km2": round(a2 - a1, 4),
                "Delta_pct": round(100 * (a2 / a1 - 1), 4) if a1 > 0 else np.nan,
                "Annual_rate_pct_per_yr": round(r, 4) if np.isfinite(r) else np.nan,
            })
    return pd.DataFrame(rows)


# =====================================================================
# (2) LANDSCAPE METRICS
# =====================================================================

def _perimeter_m(prop, pix_side):
    m = prop.image
    pad = np.pad(m.astype(np.uint8), 1, mode="constant", constant_values=0)
    boundary = (pad[1:-1, 1:-1] == 1) & (
        (pad[:-2, 1:-1] == 0) | (pad[2:, 1:-1] == 0) |
        (pad[1:-1, :-2] == 0) | (pad[1:-1, 2:] == 0))
    return float(boundary.sum()) * pix_side


def landscape_metrics(arr, pixel_area_m2, classes=(1, 2, 3, 4, 5)):
    pix_side = np.sqrt(pixel_area_m2)
    total_area = float(np.sum(np.isin(arr, classes))) * pixel_area_m2
    rows = []
    for cls in classes:
        mask = (arr == cls)
        if not mask.any():
            rows.append({"Class": CLASS_NAMES[cls], "NP": 0,
                         "MPS_ha": 0.0, "LPI_pct": 0.0, "PCI": 0.0})
            continue
        lbl = sk_label(mask, connectivity=2)
        props = regionprops(lbl)
        areas_m2 = np.array([p.area * pixel_area_m2 for p in props])
        perims_m = np.array([_perimeter_m(p, pix_side) for p in props])
        NP = len(props)
        MPS_ha = areas_m2.mean() / 10_000.0
        LPI_pct = 100.0 * areas_m2.max() / total_area
        num = np.sum(perims_m); den = np.sum(perims_m * np.sqrt(areas_m2))
        if den > 0 and total_area > 0:
            pci_raw = (1.0 - num / den) * (
                1.0 - 1.0 / np.sqrt(total_area)) ** -1
            PCI = max(0.0, min(pci_raw, 1.0)) * 100.0
        else:
            PCI = 0.0
        rows.append({"Class": CLASS_NAMES[cls], "NP": int(NP),
                     "MPS_ha": round(MPS_ha, 2),
                     "LPI_pct": round(LPI_pct, 4), "PCI": round(PCI, 2)})
    return pd.DataFrame(rows)


def compute_landscape_table(arrays, pixel_area_m2):
    frames = []
    for yr, arr in sorted(arrays.items()):
        df = landscape_metrics(arr, pixel_area_m2); df["Year"] = yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[
        ["Year", "Class", "NP", "MPS_ha", "LPI_pct", "PCI"]]


# =====================================================================
# (3) FOREST SWAP / NET
# =====================================================================

def forest_swap_net(a_from, a_to, pixel_area_m2, t1, t2):
    km2 = pixel_area_m2 / 1_000_000.0
    init = a_from == 2; fin = a_to == 2
    gain_pix = int(np.sum(~init & fin))
    loss_pix = int(np.sum(init & ~fin))
    pers_pix = int(np.sum(init & fin))
    gain = gain_pix * km2; loss = loss_pix * km2
    net = gain - loss; total = gain + loss; swap = total - abs(net)
    return {
        "Phase": f"{t1}-{t2}",
        "Forest_start_km2": round((gain_pix + pers_pix) * km2, 4),
        "Forest_end_km2": round((loss_pix + pers_pix) * km2, 4),
        "Gain_km2": round(gain, 4), "Loss_km2": round(loss, 4),
        "Net_km2": round(net, 4), "Swap_km2": round(swap, 4),
        "Total_change_km2": round(total, 4),
        "Swap_share_pct": round(100 * swap / total, 3) if total > 0 else 0.0,
        "Net_share_pct": round(100 * abs(net) / total, 3) if total > 0 else 0.0,
        "Gain_to_loss_ratio": round(gain / loss, 3) if loss > 0 else np.nan,
    }


def forest_swap_net_all(arrays, pixel_area_m2):
    return pd.DataFrame([
        forest_swap_net(arrays[t1], arrays[t2], pixel_area_m2, t1, t2)
        for t1, t2 in PHASES])


# =====================================================================
# (4) CHANGE RASTERS
# =====================================================================

def build_change_raster(a_from, a_to):
    out = np.zeros_like(a_from, dtype=np.uint8)
    valid = (a_from >= 1) & (a_from <= 5) & (a_to >= 1) & (a_to <= 5)
    persist = valid & (a_from == a_to)
    f2a = valid & (a_from == 2) & (a_to == 1)
    f2o = valid & (a_from == 2) & (a_to != 1) & (a_to != 2)
    a2f = valid & (a_from == 1) & (a_to == 2)
    o2f = valid & (a_from != 1) & (a_from != 2) & (a_to == 2)
    o2a = valid & (a_from != 1) & (a_from != 2) & (a_to == 1)
    out[f2a] = 1; out[f2o] = 2; out[a2f] = 3; out[o2f] = 4; out[o2a] = 5
    other = valid & ~persist & ~(f2a | f2o | a2f | o2f | o2a)
    out[other] = 6
    out[~valid] = 255
    return out


def write_change_rasters(arrays, meta, out_dir):
    for (t1, t2) in PHASES:
        change = build_change_raster(arrays[t1], arrays[t2])
        write_geotiff(change, meta,
                      out_dir / f"Change_{t1}_{t2}.tif", dtype="uint8")


# =====================================================================
# (5) FIGURES
# =====================================================================

def _style_axis(ax):
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def fig_area_trajectory(area_df, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    for cls in area_df.index:
        code = list(CLASS_NAMES.values()).index(cls) + 1
        ax.plot(list(area_df.columns), area_df.loc[cls].values,
                marker="o", ms=8, lw=2.4, color=CLASS_COLORS[code], label=cls)
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Area (km$^2$)", fontweight="bold")
    ax.set_title("Land-use area trajectory, Gia Lai (2000-2025)",
                 fontweight="bold")
    ax.legend(loc="center left", framealpha=0.95, edgecolor="#B0B0B0")
    _style_axis(ax); fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def fig_transition_heatmap(mats, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
    for ax, (label, mat) in zip(axes.flat, mats.items()):
        m = mat.values.copy()
        im = ax.imshow(m, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(mat.columns, rotation=35, ha="right")
        ax.set_yticklabels(mat.index)
        vmax = np.nanmax(m)
        for i in range(5):
            for j in range(5):
                v = m[i, j]
                col = "white" if v > vmax * 0.55 else "#2B2B2B"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=10.5, color=col, fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("To", fontweight="bold")
        ax.set_ylabel("From", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="km$^2$")
    fig.suptitle("LULC transition matrices (km$^2$)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig); logger.info("Saved: %s", out_path.name)


def fig_swap_net(swap_df, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    x = np.arange(len(swap_df)); w = 0.38
    ax.bar(x - w/2, swap_df["Net_km2"], w, color="#B2182B",
           edgecolor="white", linewidth=1.1, label="Net change")
    ax.bar(x + w/2, swap_df["Swap_km2"], w, color="#2166AC",
           edgecolor="white", linewidth=1.1, label="Swap (reallocation)")
    for i, r in swap_df.iterrows():
        ax.text(x[i] - w/2, r["Net_km2"], f"{r['Net_km2']:,.0f}",
                ha="center", va="bottom" if r["Net_km2"] >= 0 else "top",
                fontsize=10.5, fontweight="bold")
        ax.text(x[i] + w/2, r["Swap_km2"], f"{r['Swap_km2']:,.0f}",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.axhline(0, color="#404040", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(swap_df["Phase"])
    ax.set_ylabel("Forest area (km$^2$)", fontweight="bold")
    ax.set_title("Forest: Swap vs. Net decomposition per phase",
                 fontweight="bold")
    ax.legend(loc="best", framealpha=0.95, edgecolor="#B0B0B0")
    _style_axis(ax); fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def fig_landscape_metrics(lm_df, out_path):
    metrics = ["NP", "MPS_ha", "LPI_pct", "PCI"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=300)
    for ax, metric in zip(axes.flat, metrics):
        for cls in lm_df["Class"].unique():
            sub = lm_df[lm_df["Class"] == cls].sort_values("Year")
            code = list(CLASS_NAMES.values()).index(cls) + 1
            ax.plot(sub["Year"], sub[metric], marker="o", ms=7, lw=2.2,
                    color=CLASS_COLORS[code], label=cls)
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Year", fontweight="bold")
        _style_axis(ax)
    axes.flat[0].legend(loc="best", framealpha=0.95, edgecolor="#B0B0B0")
    fig.suptitle("Landscape structure metrics (2000-2025)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig); logger.info("Saved: %s", out_path.name)


def fig_annual_rate(rate_df, out_path):
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)
    classes = list(rate_df["Class"].unique())
    phases = list(rate_df["Phase"].unique())
    x = np.arange(len(classes)); w = 0.20
    phase_colors = ["#B2182B", "#E08214", "#2166AC", "#1B5E20"]
    for i, ph in enumerate(phases):
        sub = rate_df[rate_df["Phase"] == ph].set_index("Class").reindex(classes)
        vals = sub["Annual_rate_pct_per_yr"].values
        ax.bar(x + (i - 1.5) * w, vals, w, color=phase_colors[i],
               edgecolor="white", linewidth=1.0, label=ph)
    ax.axhline(0, color="#404040", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Annual rate of change (%/yr, Puyravaud)",
                  fontweight="bold")
    ax.set_title("Annual rate of change per class and phase",
                 fontweight="bold")
    ax.legend(loc="best", ncol=2, framealpha=0.95, edgecolor="#B0B0B0")
    _style_axis(ax); fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", out_path.name)


# =====================================================================
# VALIDATION
# =====================================================================

def validate_against_reference(area_df, swap_df):
    def _check(label, computed, ref):
        delta = computed - ref
        if abs(delta) <= TOL:
            logger.info("OK  %s: computed=%.2f  ref=%.2f  D=%.2f",
                        label, computed, ref, delta)
        else:
            logger.warning("!!  %s: computed=%.2f  ref=%.2f  D=%.2f",
                           label, computed, ref, delta)

    _check("Forest 2000 (km2)", area_df.loc["Forest", 2000],
           REFERENCE["forest_km2_2000"])
    _check("Forest 2025 (km2)", area_df.loc["Forest", 2025],
           REFERENCE["forest_km2_2025"])
    _check("Agriculture 2000 (km2)", area_df.loc["Agriculture", 2000],
           REFERENCE["agri_km2_2000"])
    _check("Agriculture 2025 (km2)", area_df.loc["Agriculture", 2025],
           REFERENCE["agri_km2_2025"])
    p1 = swap_df.loc[swap_df["Phase"] == "2000-2010", "Net_km2"].values[0]
    _check("Forest Net 2000-2010 (km2)", p1,
           REFERENCE["forest_loss_2000_2010_km2"])
    p3 = swap_df.loc[swap_df["Phase"] == "2020-2025", "Net_km2"].values[0]
    _check("Forest Net 2020-2025 (km2)", p3,
           REFERENCE["forest_gain_2020_2025_km2"])


# =====================================================================
# PIPELINE
# =====================================================================

def run_pipeline(raster_dir, aligned_dir, out_dir):
    fig_dir = out_dir / "Figures"
    tab_dir = out_dir / "Tables"
    ras_dir = out_dir / "Rasters"
    for d in (fig_dir, tab_dir, ras_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 68)
    logger.info("STEP 0 - Aligning rasters to LULC_2000 reference grid")
    logger.info("=" * 68)
    aligned_paths = align_rasters(raster_dir, aligned_dir)

    logger.info("=" * 68)
    logger.info("Loading aligned LULC stack")
    logger.info("=" * 68)
    arrays, meta = load_aligned_stack(aligned_paths)
    pixel_area_m2 = meta["pixel_area_m2"]

    logger.info("=" * 68)
    logger.info("(1) Area, transition matrices, turnover, annual rate")
    logger.info("=" * 68)
    area_df = compute_area_table(arrays, pixel_area_m2)
    area_df.to_csv(tab_dir / "Table_Area_by_Class.csv",
                   encoding="utf-8-sig")
    logger.info("\n%s", area_df.round(2).to_string())

    mats = {}
    turnover_frames = []
    for t1, t2 in PHASES:
        mat = transition_matrix(arrays[t1], arrays[t2], pixel_area_m2)
        mats[f"{t1}-{t2}"] = mat
        mat.to_csv(tab_dir / f"Table_TransitionMatrix_{t1}_{t2}.csv",
                   encoding="utf-8-sig")
        tn = turnover_table(mat); tn["Phase"] = f"{t1}-{t2}"
        turnover_frames.append(tn)

    turnover_df = pd.concat(turnover_frames, ignore_index=True)[
        ["Phase", "Class", "Initial_km2", "Final_km2", "Gain_km2",
         "Loss_km2", "Net_km2", "Swap_km2", "Total_change_km2",
         "Net_pct_of_total", "Swap_pct_of_total"]]
    turnover_df.to_csv(tab_dir / "Table_Turnover.csv",
                       index=False, encoding="utf-8-sig")

    rate_df = annual_rate_table(area_df)
    rate_df.to_csv(tab_dir / "Table_AnnualRateChange.csv",
                   index=False, encoding="utf-8-sig")

    logger.info("=" * 68)
    logger.info("(2) Landscape metrics (NP, MPS, LPI, PCI)")
    logger.info("=" * 68)
    lm_df = compute_landscape_table(arrays, pixel_area_m2)
    lm_df.to_csv(tab_dir / "Table_LandscapeMetrics.csv",
                 index=False, encoding="utf-8-sig")
    logger.info("\n%s", lm_df.to_string(index=False))

    logger.info("=" * 68)
    logger.info("(3) Forest: Swap vs. Net decomposition")
    logger.info("=" * 68)
    swap_df = forest_swap_net_all(arrays, pixel_area_m2)
    swap_df.to_csv(tab_dir / "Table_Forest_SwapNet.csv",
                   index=False, encoding="utf-8-sig")
    logger.info("\n%s", swap_df.to_string(index=False))

    logger.info("=" * 68)
    logger.info("(4) Writing change rasters for QGIS")
    logger.info("=" * 68)
    write_change_rasters(arrays, meta, ras_dir)

    logger.info("=" * 68)
    logger.info("(5) Publication figures")
    logger.info("=" * 68)
    fig_area_trajectory(area_df, fig_dir / "Fig_S1_AreaTrajectory.png")
    fig_transition_heatmap(mats, fig_dir / "Fig_S2_TransitionHeatmap.png")
    fig_swap_net(swap_df, fig_dir / "Fig_S3_SwapNet_Forest.png")
    fig_landscape_metrics(lm_df, fig_dir / "Fig_S4_LandscapeMetrics.png")
    fig_annual_rate(rate_df, fig_dir / "Fig_S5_AnnualRateChange.png")

    logger.info("=" * 68)
    logger.info("Validation against previous run")
    logger.info("=" * 68)
    validate_against_reference(area_df, swap_df)

    logger.info("=" * 68)
    logger.info("DONE - Section 3.1 pipeline completed")
    logger.info("Output root: %s", out_dir)
    logger.info("=" * 68)


def parse_args():
    p = argparse.ArgumentParser(
        description="Section 3.1 - LULC dynamics (Gia Lai, 2000-2025).")
    p.add_argument("--raster_dir", type=Path, default=DEFAULT_RASTER_DIR)
    p.add_argument("--aligned_dir", type=Path, default=DEFAULT_ALIGNED_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.raster_dir, args.aligned_dir, args.outdir)


if __name__ == "__main__":
    main()