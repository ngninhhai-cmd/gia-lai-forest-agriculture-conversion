# Spatiotemporal Dynamics of Forest–Agriculture Conversion in Gia Lai Province, Vietnam (2000–2025)

## Overview

Data and code for analysing forest–agriculture conversion in Gia Lai Province, Vietnam (2000–2025). The study integrates transition matrices, landscape metrics, hotspot analysis (Moran's I, Getis-Ord Gi*, LISA), and spatial regression (SAR ML_Lag).

## Repository structure

- `src/` — Python source code
- `data/raw/` — Raw rasters and vectors
- `data/processed/` — Aligned data and fishnet (n = 15,886 cells)
- `results/Tables/` — Output CSV tables
- `results/Figures/` — Output figures
- `results/Rasters/` — Change detection rasters

## Requirements

Python 3.10+ with packages listed in `requirements.txt`.

## Data preparation

Two files are distributed as ZIP archives. Unzip before running:

    cd data/raw
    unzip GL_DEM.zip
    unzip GI_Soil.zip

## Reproducing the analysis

    python src/03_1_land_use_dynamics.py
    python src/03_2_spatial_patterns.py
    python src/03_3_conversion_drivers.py

## Citation

If you use this data or code, please cite:

> Nguyen, N.-H. (2025). Spatiotemporal Dynamics of Forest–Agriculture Conversion in Gia Lai Province, Vietnam (2000–2025). *GeoJournal*.

## License

- Code: MIT License
- Data: CC-BY-4.0

## Contact

Ninh-Hai Nguyen — ngninhhai@hcmuaf.edu.vn