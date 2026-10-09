#!/usr/bin/env python
# coding: utf-8

"""
Generate Supporting Information Figures S1–S9.

Inputs
------
1. COMMON_SUPPORT_ERA5_TABLE
   Used for Fig. S9.

2. CROPPED_RASTER_ROOT
   Shelf folders containing annual 30 m ponding rasters.

3. SHAPEFILE_DIR
   Shelf shapefiles.

Figures: 
S1  Example George VI shelf-cropped raster and binary pond mask
S2  Common support for Cook, Fimbul, Mertz, and Shackleton
S3  Observed/expected ponded-pixel adjacency and R_PP
S4  Moran's I
S5  Lagged R_PP correlogram
S6  Connected-component behavior
S7  Shelf-balanced quantile curves and empirical knees
S8  Whole-shelf bootstrap, minimum-pixel sensitivity, and leave-one-out
S9  Temperature distribution, ponding prevalence, and shelf support

Environment variables

CODE_DIR
COMMON_SUPPORT_ERA5_TABLE
CROPPED_RASTER_ROOT
SHAPEFILE_DIR
SI_S1_S9_OUTPUT_DIR
YEAR_MIN
YEAR_MAX
N_WORKERS
ALL_TOUCHED
PRIMARY_MIN_PONDED_PIXELS
N_KNEE_BOOTSTRAP
RANDOM_SEED
OUTPUT_DPI
EXAMPLE_SHELF
EXAMPLE_YEAR
"""

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import glob
import hashlib
import json
import math
import multiprocessing as mp
import re
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask
from rasterio.plot import plotting_extent
from scipy import ndimage
from shapely.geometry import box
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Initial Settings

CODE_DIR = Path(
    os.environ.get(
        "CODE_DIR",
        "/raid01/mafields/tas/MODELS_filtered/ssp585/jupyter",
    )
)

COMMON_SUPPORT_ERA5_TABLE = Path(
    os.environ.get(
        "COMMON_SUPPORT_ERA5_TABLE",
        str(
            CODE_DIR
            / "corrected_crop_and_era5_alignment_outputs"
            / "observations_ERA5temp_ponding_COMMON_SUPPORT_2006_2020.parquet"
        ),
    )
)

CROPPED_RASTER_ROOT = Path(
    os.environ.get(
        "CROPPED_RASTER_ROOT",
        (
            "/raid01/mafields/project_two/"
            "Antarctic_wide_surface_meltwater_data_repair/"
            "corrected_shelf_crops_from_antwide_RECROPPED_TO_SHELF_BOUNDS"
        ),
    )
)

SHAPEFILE_DIR = Path(
    os.environ.get(
        "SHAPEFILE_DIR",
        "/raid01/mafields/tas/MODELS_filtered/ssp585/shape_files",
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "SI_S1_S9_OUTPUT_DIR",
        "si_outputs/figures_s1_s9",
    )
)

FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"

for directory in (
    OUTPUT_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    CACHE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

YEAR_MIN = int(
    os.environ.get(
        "YEAR_MIN",
        "2006",
    )
)

YEAR_MAX = int(
    os.environ.get(
        "YEAR_MAX",
        "2020",
    )
)

N_WORKERS = int(
    os.environ.get(
        "N_WORKERS",
        "8",
    )
)

MP_CONTEXT = os.environ.get(
    "MP_CONTEXT",
    "fork",
)

ALL_TOUCHED = (
    os.environ.get(
        "ALL_TOUCHED",
        "0",
    )
    == "1"
)

PRIMARY_MIN_PONDED_PIXELS = int(
    os.environ.get(
        "PRIMARY_MIN_PONDED_PIXELS",
        "100",
    )
)

MIN_VALID_PIXELS = int(
    os.environ.get(
        "MIN_VALID_PIXELS",
        "100",
    )
)

N_KNEE_BOOTSTRAP = int(
    os.environ.get(
        "N_KNEE_BOOTSTRAP",
        "1000",
    )
)

RANDOM_SEED = int(
    os.environ.get(
        "RANDOM_SEED",
        "42",
    )
)

OUTPUT_DPI = int(
    os.environ.get(
        "OUTPUT_DPI",
        "300",
    )
)

EXAMPLE_SHELF = os.environ.get(
    "EXAMPLE_SHELF",
    "George VI",
)

EXAMPLE_YEAR = int(
    os.environ.get(
        "EXAMPLE_YEAR",
        "2020",
    )
)

TARGET_CRS = os.environ.get(
    "TARGET_CRS",
    "EPSG:3031",
)

PONDED_VALUES = tuple(
    float(value)
    for value in os.environ.get(
        "PONDED_VALUES",
        "1",
    ).split(",")
    if value.strip()
)

CORRELOGRAM_LAGS_PIXELS = tuple(
    int(value)
    for value in os.environ.get(
        "CORRELOGRAM_LAGS_PIXELS",
        "1,2,3,5,10,20,33,50,100",
    ).split(",")
)

CUTOFFS_TO_TEST = tuple(
    int(value)
    for value in os.environ.get(
        "CUTOFFS_TO_TEST",
        "25,50,75,100,125,150,175,200,250,500",
    ).split(",")
)

KNEE_QUANTILES = tuple(
    float(value)
    for value in os.environ.get(
        "KNEE_QUANTILES",
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9",
    ).split(",")
)

KNEE_QUANTILES_TO_PLOT = (
    0.50,
    0.75,
    0.90,
)

KNEE_LOCAL_WINDOW_DEX = float(
    os.environ.get(
        "KNEE_LOCAL_WINDOW_DEX",
        "0.35",
    )
)

KNEE_MIN_LOCAL_POINTS = int(
    os.environ.get(
        "KNEE_MIN_LOCAL_POINTS",
        "20",
    )
)

KNEE_MIN_LOCAL_SHELVES = int(
    os.environ.get(
        "KNEE_MIN_LOCAL_SHELVES",
        "2",
    )
)

KNEE_GRID_SIZE = int(
    os.environ.get(
        "KNEE_GRID_SIZE",
        "250",
    )
)

TEMPERATURE_BIN_WIDTH_C = float(
    os.environ.get(
        "TEMPERATURE_BIN_WIDTH_C",
        "0.5",
    )
)

PIXEL_AREA_KM2_NOMINAL = 30.0**2 / 1_000_000.0

# Shelf names and regions

SHELF_REGION_MAP = {
    "LarsenB": "Antarctic Peninsula",
    "LarsenC": "Antarctic Peninsula",
    "LarsenD": "Antarctic Peninsula",
    "George VI": "Antarctic Peninsula",
    "Stange": "Antarctic Peninsula",
    "Pine Island": "Amundsen–Bellingshausen",
    "Thwaites": "Amundsen–Bellingshausen",
    "Crosson": "Amundsen–Bellingshausen",
    "Dotson": "Amundsen–Bellingshausen",
    "Getz": "Amundsen–Bellingshausen",
    "Abbot": "Amundsen–Bellingshausen",
    "Cosgrove": "Amundsen–Bellingshausen",
    "Venable": "Amundsen–Bellingshausen",
    "Nickerson": "Amundsen–Bellingshausen",
    "Sulzberger": "Amundsen–Bellingshausen",
    "Ross East": "Ross Sea",
    "Ross West": "Ross Sea",
    "Drygalski": "Ross Sea",
    "Nansen": "Ross Sea",
    "Land": "Ross Sea",
    "Filchner": "Weddell Sea",
    "Ronne": "Weddell Sea",
    "Brunt Stancomb": "Weddell Sea",
    "Riiser-Larsen": "Weddell Sea",
    "Fimbul": "Weddell Sea",
    "Ekstrom": "Weddell Sea",
    "Jelbart": "Weddell Sea",
    "Lazarev": "Weddell Sea",
    "Amery": "Amery",
    "West": "East Antarctica excluding Amery",
    "Shackleton": "East Antarctica excluding Amery",
    "Moscow University": "East Antarctica excluding Amery",
    "Totten": "East Antarctica excluding Amery",
    "Cook": "East Antarctica excluding Amery",
    "Mertz": "East Antarctica excluding Amery",
    "Mariner": "East Antarctica excluding Amery",
    "Nivl": "East Antarctica excluding Amery",
    "Prince Harald": "East Antarctica excluding Amery",
    "Quar": "East Antarctica excluding Amery",
    "Vigrid": "East Antarctica excluding Amery",
    "Withrow": "East Antarctica excluding Amery",
    "Holmes": "East Antarctica excluding Amery",
    "Baudouin": "East Antarctica excluding Amery",
    "Borchgrevink": "East Antarctica excluding Amery",
    "Conger Glenzer": "East Antarctica excluding Amery",
    "Atka": "East Antarctica excluding Amery",
    "Rennick": "East Antarctica excluding Amery",
}

KEY_MAP = {
    "abbot": "Abbot",
    "abbot3031": "Abbot",
    "amery": "Amery",
    "atka": "Atka",
    "baudouin": "Baudouin",
    "borchgrevink": "Borchgrevink",
    "bruntstancomb": "Brunt Stancomb",
    "congerglenzer": "Conger Glenzer",
    "cook": "Cook",
    "cosgrove": "Cosgrove",
    "crosson": "Crosson",
    "dotson": "Dotson",
    "drygalski": "Drygalski",
    "ekstrom": "Ekstrom",
    "filchner": "Filchner",
    "fimbul": "Fimbul",
    "georgevi": "George VI",
    "getz": "Getz",
    "holmes": "Holmes",
    "jelbart": "Jelbart",
    "land": "Land",
    "larsenb": "LarsenB",
    "larsenc": "LarsenC",
    "larsend": "LarsenD",
    "lazarev": "Lazarev",
    "mariner": "Mariner",
    "mertz": "Mertz",
    "moscowuniversity": "Moscow University",
    "nansen": "Nansen",
    "nickerson": "Nickerson",
    "nivl": "Nivl",
    "pine": "Pine Island",
    "pineisland": "Pine Island",
    "princeharald": "Prince Harald",
    "quar": "Quar",
    "rennick": "Rennick",
    "riiserlarsen": "Riiser-Larsen",
    "ronne": "Ronne",
    "rosseast": "Ross East",
    "rosswest": "Ross West",
    "shackleton": "Shackleton",
    "stange": "Stange",
    "sulzberger": "Sulzberger",
    "thwaites": "Thwaites",
    "totten": "Totten",
    "venable": "Venable",
    "vigrid": "Vigrid",
    "west": "West",
    "withrow": "Withrow",
}

# Universal Plot style

COLOR_POND = "#0072B2"
COLOR_ORANGE = "#D55E00"
COLOR_YELLOW = "#E69F00"
COLOR_GREEN = "#009E73"
COLOR_PURPLE = "#6A3D9A"
COLOR_GRAY = "0.45"
COLOR_LIGHT_GRAY = "0.82"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 150,
        "savefig.dpi": OUTPUT_DPI,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# General Utility functions

def info(message):
    print(f"[INFO] {message}", flush=True)


def warn(message):
    print(f"[WARN] {message}", flush=True)


def norm_shelf_name(value):
    return (
        str(value)
        .lower()
        .replace("ice shelf", "")
        .replace("iceshelf", "")
        .replace("shelf", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("–", "")
        .replace(".", "")
        .replace("'", "")
        .replace('"', "")
    )


def canonical_shelf_name(value):
    key = norm_shelf_name(value)
    return KEY_MAP.get(key, str(value).strip())


def assign_region(shelf):
    return SHELF_REGION_MAP.get(
        canonical_shelf_name(shelf),
        "Unknown region",
    )


def safe_filename(value):
    value = str(value).replace("–", "-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_")


def read_table(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    suffixes = [suffix.lower() for suffix in path.suffixes]

    if ".parquet" in suffixes or ".pq" in suffixes:
        return pd.read_parquet(path)

    if ".csv" in suffixes:
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported table type: {path}")


def atomic_csv_write(dataframe, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=index)
    temporary.replace(path)

    print("[SAVED]", path, flush=True)


def atomic_json_write(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    temporary.replace(path)
    print("[SAVED]", path, flush=True)


def save_figure(figure, basename):
    basename = FIGURE_DIR / basename

    png_path = Path(str(basename) + ".png")
    pdf_path = Path(str(basename) + ".pdf")

    figure.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    print("[SAVED]", png_path, flush=True)
    print("[SAVED]", pdf_path, flush=True)


def style_axis(axis):
    axis.grid(
        True,
        which="major",
        color="0.86",
        linewidth=0.8,
    )

    axis.grid(
        True,
        which="minor",
        color="0.93",
        linewidth=0.5,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def percent_formatter(value, position=None):
    return f"{100.0 * value:g}%"


def stable_seed(*parts):
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.full(len(quantiles), np.nan)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cumulative = (
        cumulative - 0.5 * weights
    ) / np.sum(weights)

    return np.interp(
        quantiles,
        cumulative,
        values,
    )


def expand_bounds(bounds, fraction=0.07):
    minimum_x, minimum_y, maximum_x, maximum_y = bounds

    width = max(maximum_x - minimum_x, 1.0)
    height = max(maximum_y - minimum_y, 1.0)

    return (
        minimum_x - fraction * width,
        minimum_y - fraction * height,
        maximum_x + fraction * width,
        maximum_y + fraction * height,
    )

#Process shelf folders, rasters, and shapefiles

def parse_year_from_raster(path):
    name = Path(path).name

    match = re.search(
        r"ANT_WIDE_(\d{4})_(\d{4})_30m_raster\.tif$",
        name,
    )

    if match:
        return int(match.group(1))

    match = re.search(
        r"(\d{4})_masked\.tif$",
        name,
    )

    if match:
        return int(match.group(1))

    match = re.search(r"(\d{4})", name)

    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not parse year from {name}"
    )


def infer_shelf_from_shapefile(path):
    stem = Path(path).stem

    for suffix in (
        "_outline",
        "_polygon",
        "_poly",
    ):
        stem = stem.replace(suffix, "")

    return canonical_shelf_name(
        stem.replace("_", " ")
    )


def discover_shapefiles():
    rows = []

    for path in sorted(
        SHAPEFILE_DIR.glob("*.shp")
    ):
        shelf = infer_shelf_from_shapefile(path)

        if "pre2002" in shelf.lower():
            continue

        rows.append(
            {
                "shelf": shelf,
                "shapefile_path": str(path),
            }
        )

    output = pd.DataFrame(rows)

    if output.empty:
        raise RuntimeError(
            f"No shapefiles were found in {SHAPEFILE_DIR}"
        )

    return output.drop_duplicates(
        "shelf",
        keep="first",
    )


def discover_rasters(folder):
    candidates = set()

    candidates.update(
        glob.glob(
            str(
                Path(folder)
                / "ANT_WIDE_*_*_30m_raster.tif"
            )
        )
    )

    candidates.update(
        glob.glob(
            str(
                Path(folder)
                / "*_masked.tif"
            )
        )
    )

    rows = []

    for path in sorted(candidates):
        try:
            year = parse_year_from_raster(path)
        except Exception:
            continue

        if YEAR_MIN <= year <= YEAR_MAX:
            rows.append(
                {
                    "year": year,
                    "raster_path": path,
                    "priority": int(
                        Path(path).name.startswith(
                            "ANT_WIDE_"
                        )
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["year", "priority"],
            ascending=[True, False],
        )
        .drop_duplicates(
            "year",
            keep="first",
        )
        .drop(columns="priority")
        .reset_index(drop=True)
    )


def build_inventory():
    shapefiles = discover_shapefiles()

    folder_rows = []

    for folder in sorted(
        path
        for path in CROPPED_RASTER_ROOT.iterdir()
        if path.is_dir()
    ):
        shelf = canonical_shelf_name(folder.name)

        if "pre2002" in shelf.lower():
            continue

        folder_rows.append(
            {
                "shelf": shelf,
                "folder_path": str(folder),
            }
        )

    folders = pd.DataFrame(folder_rows)

    if folders.empty:
        raise RuntimeError(
            f"No shelf folders found in {CROPPED_RASTER_ROOT}"
        )

    inventory = folders.merge(
        shapefiles,
        on="shelf",
        how="inner",
    )

    inventory["region"] = inventory[
        "shelf"
    ].map(assign_region)

    rows = []

    for record in inventory.itertuples(index=False):
        rasters = discover_rasters(
            record.folder_path
        )

        for raster in rasters.itertuples(index=False):
            rows.append(
                {
                    "shelf": record.shelf,
                    "region": record.region,
                    "folder_path": record.folder_path,
                    "shapefile_path": record.shapefile_path,
                    "year": int(raster.year),
                    "raster_path": raster.raster_path,
                }
            )

    output = pd.DataFrame(rows)

    if output.empty:
        raise RuntimeError(
            "No shelf-year raster tasks were discovered."
        )

    return output.sort_values(
        ["shelf", "year"]
    ).reset_index(drop=True)

# Raster and shelf geometry processing

def load_shelf_geometry(
    shapefile_path,
    raster_crs,
):
    geodata = gpd.read_file(
        shapefile_path
    )

    geodata = geodata.loc[
        geodata.geometry.notna()
        & ~geodata.geometry.is_empty
    ].copy()

    if geodata.empty:
        raise ValueError(
            f"No valid geometry in {shapefile_path}"
        )

    if geodata.crs is None:
        geodata = geodata.set_crs(
            TARGET_CRS
        )

    if str(geodata.crs) != str(raster_crs):
        geodata = geodata.to_crs(
            raster_crs
        )

    return geodata.geometry.unary_union


def pond_mask_from_values(values):
    return np.isin(
        np.asarray(values, dtype=float),
        PONDED_VALUES,
    )


def crop_to_valid_extent(
    pond,
    valid,
):
    rows, columns = np.nonzero(valid)

    if len(rows) == 0:
        return pond, valid

    row_min = rows.min()
    row_max = rows.max() + 1
    column_min = columns.min()
    column_max = columns.max() + 1

    return (
        pond[
            row_min:row_max,
            column_min:column_max,
        ],
        valid[
            row_min:row_max,
            column_min:column_max,
        ],
    )


def read_masks(
    raster_path,
    geometry,
):
    with rasterio.open(raster_path) as source:
        raw = source.read(1)
        transform = source.transform

        inside_shelf = geometry_mask(
            [geometry],
            out_shape=raw.shape,
            transform=transform,
            invert=True,
            all_touched=ALL_TOUCHED,
        )

        # The raster's rectangular extent is the available footprint.
        # Pixels within the shelf polygon and raster bounds are retained.
        valid = inside_shelf.astype(bool)

        values = raw.astype(float, copy=True)

        nodata_mask = ~np.isfinite(values)

        if source.nodata is not None:
            try:
                if np.isfinite(source.nodata):
                    nodata_mask |= raw == source.nodata
            except Exception:
                pass

        # Match the old recropped-raster convention:
        # nodata/255 inside geometric support is nonponded.
        values[nodata_mask] = 0.0

        pond = (
            pond_mask_from_values(values)
            & valid
        )

        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)

        pixel_area_km2 = (
            pixel_size_x
            * pixel_size_y
            / 1_000_000.0
        )

        raster_bounds_geometry = box(
            source.bounds.left,
            source.bounds.bottom,
            source.bounds.right,
            source.bounds.top,
        )

        shelf_area = float(geometry.area)
        overlap_area = float(
            geometry.intersection(
                raster_bounds_geometry
            ).area
        )

        coverage_fraction = (
            overlap_area / shelf_area
            if shelf_area > 0
            else np.nan
        )

        metadata = {
            "pixel_size_x_m": float(pixel_size_x),
            "pixel_size_y_m": float(pixel_size_y),
            "pixel_area_km2": float(pixel_area_km2),
            "n_valid_pixels": int(valid.sum()),
            "n_ponded_pixels": int(pond.sum()),
            "ponded_fraction": (
                float(pond.sum() / valid.sum())
                if valid.sum() > 0
                else np.nan
            ),
            "shelf_area_m2": shelf_area,
            "raster_overlap_area_m2": overlap_area,
            "coverage_fraction": coverage_fraction,
            "raster_bounds": tuple(source.bounds),
            "raster_crs": str(source.crs),
            "raster_nodata": source.nodata,
        }

    return (
        pond.astype(bool),
        valid.astype(bool),
        metadata,
    )

# Spatial Diagnostics

def exact_expected_pp_pairs(
    number_pairs,
    number_ponded,
    number_valid,
):
    if (
        number_pairs <= 0
        or number_valid < 2
        or number_ponded < 2
    ):
        return 0.0

    return float(
        number_pairs
        * number_ponded
        * (number_ponded - 1)
        / (
            number_valid
            * (number_valid - 1)
        )
    )


def rook_neighbor_metrics(
    pond,
    valid,
):
    values = pond.astype(float)

    horizontal_valid = (
        valid[:, :-1]
        & valid[:, 1:]
    )

    vertical_valid = (
        valid[:-1, :]
        & valid[1:, :]
    )

    first = np.concatenate(
        [
            values[:, :-1][horizontal_valid],
            values[:-1, :][vertical_valid],
        ]
    )

    second = np.concatenate(
        [
            values[:, 1:][horizontal_valid],
            values[1:, :][vertical_valid],
        ]
    )

    number_pairs = int(len(first))
    number_valid = int(valid.sum())
    number_ponded = int(pond.sum())

    if number_pairs == 0:
        return {
            "n_neighbor_pairs": 0,
            "ponded_ponded_neighbor_pairs": np.nan,
            "ponded_unponded_neighbor_pairs": np.nan,
            "unponded_unponded_neighbor_pairs": np.nan,
            "expected_ponded_ponded_pairs": np.nan,
            "r_pp": np.nan,
            "moran_i_rook": np.nan,
        }

    pp = int(
        np.sum(
            (first == 1)
            & (second == 1)
        )
    )

    pu = int(
        np.sum(first != second)
    )

    uu = int(
        np.sum(
            (first == 0)
            & (second == 0)
        )
    )

    expected_pp = exact_expected_pp_pairs(
        number_pairs,
        number_ponded,
        number_valid,
    )

    r_pp = (
        pp / expected_pp
        if expected_pp > 0
        else np.nan
    )

    valid_values = values[valid]
    mean_value = float(valid_values.mean())

    denominator = float(
        np.sum(
            (valid_values - mean_value) ** 2
        )
    )

    if denominator <= 0:
        moran_i = np.nan

    else:
        numerator_pairs = float(
            np.sum(
                (first - mean_value)
                * (second - mean_value)
            )
        )

        directed_weight_sum = (
            2 * number_pairs
        )

        moran_i = (
            number_valid
            / directed_weight_sum
            * (
                2 * numerator_pairs
            )
            / denominator
        )

    return {
        "n_neighbor_pairs": number_pairs,
        "ponded_ponded_neighbor_pairs": pp,
        "ponded_unponded_neighbor_pairs": pu,
        "unponded_unponded_neighbor_pairs": uu,
        "expected_ponded_ponded_pairs": expected_pp,
        "r_pp": (
            float(r_pp)
            if np.isfinite(r_pp)
            else np.nan
        ),
        "moran_i_rook": (
            float(moran_i)
            if np.isfinite(moran_i)
            else np.nan
        ),
    }


def shifted_pair_metrics(
    pond,
    valid,
    lag_pixels,
):
    values = pond.astype(float)

    first_parts = []
    second_parts = []

    if lag_pixels < values.shape[1]:
        pair_valid = (
            valid[:, :-lag_pixels]
            & valid[:, lag_pixels:]
        )

        if pair_valid.any():
            first_parts.append(
                values[:, :-lag_pixels][pair_valid]
            )

            second_parts.append(
                values[:, lag_pixels:][pair_valid]
            )

    if lag_pixels < values.shape[0]:
        pair_valid = (
            valid[:-lag_pixels, :]
            & valid[lag_pixels:, :]
        )

        if pair_valid.any():
            first_parts.append(
                values[:-lag_pixels, :][pair_valid]
            )

            second_parts.append(
                values[lag_pixels:, :][pair_valid]
            )

    if not first_parts:
        return {
            "lag_pixels": int(lag_pixels),
            "lag_n_pairs": 0,
            "lag_pp_pairs": np.nan,
            "lag_expected_pp_pairs": np.nan,
            "lag_r_pp": np.nan,
        }

    first = np.concatenate(first_parts)
    second = np.concatenate(second_parts)

    number_pairs = int(len(first))

    pp = int(
        np.sum(
            (first == 1)
            & (second == 1)
        )
    )

    expected_pp = exact_expected_pp_pairs(
        number_pairs,
        int(pond.sum()),
        int(valid.sum()),
    )

    r_pp = (
        pp / expected_pp
        if expected_pp > 0
        else np.nan
    )

    return {
        "lag_pixels": int(lag_pixels),
        "lag_n_pairs": number_pairs,
        "lag_pp_pairs": pp,
        "lag_expected_pp_pairs": expected_pp,
        "lag_r_pp": (
            float(r_pp)
            if np.isfinite(r_pp)
            else np.nan
        ),
    }


def connected_component_metrics(
    pond,
    valid,
    pixel_area_km2,
):
    labels, number_components = ndimage.label(
        pond,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    number_valid = int(valid.sum())
    number_ponded = int(pond.sum())

    if number_components == 0:
        return {
            "n_components": 0,
            "largest_component_pixels": 0,
            "largest_component_area_km2": 0.0,
            "largest_component_fraction_of_valid": 0.0,
            "largest_component_fraction_of_ponded": np.nan,
            "spans_horizontal": False,
            "spans_vertical": False,
            "spans_either": False,
        }

    component_sizes = np.bincount(
        labels.ravel()
    )[1:]

    largest = int(
        component_sizes.max()
    )

    rows = np.where(valid.any(axis=1))[0]
    columns = np.where(valid.any(axis=0))[0]

    spans_vertical = False
    spans_horizontal = False

    if len(rows) > 0 and len(columns) > 0:
        top = int(rows.min())
        bottom = int(rows.max())
        left = int(columns.min())
        right = int(columns.max())

        top_labels = set(
            np.unique(labels[top, valid[top, :]])
        )

        bottom_labels = set(
            np.unique(labels[bottom, valid[bottom, :]])
        )

        left_labels = set(
            np.unique(labels[valid[:, left], left])
        )

        right_labels = set(
            np.unique(labels[valid[:, right], right])
        )

        for label_set in (
            top_labels,
            bottom_labels,
            left_labels,
            right_labels,
        ):
            label_set.discard(0)

        spans_vertical = bool(
            top_labels & bottom_labels
        )

        spans_horizontal = bool(
            left_labels & right_labels
        )

    return {
        "n_components": int(number_components),
        "largest_component_pixels": largest,
        "largest_component_area_km2": (
            largest * pixel_area_km2
        ),
        "largest_component_fraction_of_valid": (
            largest / number_valid
            if number_valid > 0
            else np.nan
        ),
        "largest_component_fraction_of_ponded": (
            largest / number_ponded
            if number_ponded > 0
            else np.nan
        ),
        "spans_horizontal": spans_horizontal,
        "spans_vertical": spans_vertical,
        "spans_either": (
            spans_horizontal
            or spans_vertical
        ),
    }


# =============================================================================
# Process one shelf
# =============================================================================

def process_shelf_task(task):
    shelf = task["shelf"]
    region = task["region"]
    shapefile_path = task["shapefile_path"]
    raster_records = task["rasters"]

    scene_rows = []
    correlogram_rows = []
    error_rows = []

    try:
        first_raster = raster_records[0][
            "raster_path"
        ]

        with rasterio.open(first_raster) as source:
            raster_crs = source.crs

        geometry = load_shelf_geometry(
            shapefile_path,
            raster_crs,
        )

    except Exception as error:
        for raster_record in raster_records:
            error_rows.append(
                {
                    "shelf": shelf,
                    "region": region,
                    "year": raster_record["year"],
                    "raster_path": raster_record["raster_path"],
                    "error": (
                        f"geometry_load_failed: "
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

        return (
            scene_rows,
            correlogram_rows,
            error_rows,
        )

    for raster_record in raster_records:
        year = int(raster_record["year"])
        raster_path = raster_record["raster_path"]

        try:
            pond, valid, metadata = read_masks(
                raster_path,
                geometry,
            )

            if metadata["n_valid_pixels"] < MIN_VALID_PIXELS:
                raise ValueError(
                    "Too few valid pixels: "
                    f"{metadata['n_valid_pixels']}"
                )

            pond, valid = crop_to_valid_extent(
                pond,
                valid,
            )

            scene = {
                "shelf": shelf,
                "region": region,
                "year": year,
                "raster_path": raster_path,
                "shapefile_path": shapefile_path,
                **metadata,
            }

            scene.update(
                rook_neighbor_metrics(
                    pond,
                    valid,
                )
            )

            scene.update(
                connected_component_metrics(
                    pond,
                    valid,
                    metadata["pixel_area_km2"],
                )
            )

            scene_rows.append(scene)

            mean_pixel_size = float(
                np.mean(
                    [
                        metadata["pixel_size_x_m"],
                        metadata["pixel_size_y_m"],
                    ]
                )
            )

            for lag_pixels in CORRELOGRAM_LAGS_PIXELS:
                lag = shifted_pair_metrics(
                    pond,
                    valid,
                    lag_pixels,
                )

                correlogram_rows.append(
                    {
                        "shelf": shelf,
                        "region": region,
                        "year": year,
                        "raster_path": raster_path,
                        "distance_m": (
                            lag_pixels
                            * mean_pixel_size
                        ),
                        **lag,
                    }
                )

        except Exception as error:
            error_rows.append(
                {
                    "shelf": shelf,
                    "region": region,
                    "year": year,
                    "raster_path": raster_path,
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

    return (
        scene_rows,
        correlogram_rows,
        error_rows,
    )


def run_spatial_analysis(inventory):
    tasks = []

    for (
        shelf,
        region,
        shapefile_path,
        folder_path,
    ), group in inventory.groupby(
        [
            "shelf",
            "region",
            "shapefile_path",
            "folder_path",
        ],
        observed=True,
    ):
        tasks.append(
            {
                "shelf": shelf,
                "region": region,
                "shapefile_path": shapefile_path,
                "folder_path": folder_path,
                "rasters": (
                    group[
                        ["year", "raster_path"]
                    ]
                    .sort_values("year")
                    .to_dict(orient="records")
                ),
            }
        )

    scene_rows = []
    correlogram_rows = []
    error_rows = []

    if N_WORKERS <= 1:
        for task in tqdm(
            tasks,
            desc="Processing shelves",
        ):
            scenes, correlograms, errors = (
                process_shelf_task(task)
            )

            scene_rows.extend(scenes)
            correlogram_rows.extend(correlograms)
            error_rows.extend(errors)

    else:
        context = mp.get_context(
            MP_CONTEXT
        )

        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=context,
        ) as executor:

            futures = {
                executor.submit(
                    process_shelf_task,
                    task,
                ): task["shelf"]
                for task in tasks
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing shelves",
            ):
                shelf = futures[future]

                try:
                    scenes, correlograms, errors = (
                        future.result()
                    )

                    scene_rows.extend(scenes)
                    correlogram_rows.extend(correlograms)
                    error_rows.extend(errors)

                    info(
                        f"{shelf}: {len(scenes)} shelf-years"
                    )

                except Exception as error:
                    error_rows.append(
                        {
                            "shelf": shelf,
                            "region": assign_region(shelf),
                            "year": np.nan,
                            "raster_path": "",
                            "error": (
                                f"worker_failed: "
                                f"{type(error).__name__}: {error}"
                            ),
                        }
                    )

    scenes = pd.DataFrame(scene_rows)
    correlogram = pd.DataFrame(
        correlogram_rows
    )
    errors = pd.DataFrame(error_rows)

    atomic_csv_write(
        scenes,
        CACHE_DIR
        / "recomputed_scene_spatial_metrics.csv",
    )

    atomic_csv_write(
        correlogram,
        CACHE_DIR
        / "recomputed_scene_correlogram.csv",
    )

    atomic_csv_write(
        errors,
        CACHE_DIR
        / "recomputed_spatial_errors.csv",
    )

    return scenes, correlogram, errors

#FIGURE S1

def make_figure_s1(inventory):
    candidate = inventory.loc[
        (
            inventory["shelf"]
            .map(norm_shelf_name)
            == norm_shelf_name(
                EXAMPLE_SHELF
            )
        )
        & (
            inventory["year"]
            == EXAMPLE_YEAR
        )
    ]

    if candidate.empty:
        warn(
            f"No {EXAMPLE_SHELF} {EXAMPLE_YEAR} raster was found."
        )
        return

    record = candidate.iloc[0]

    with rasterio.open(
        record["raster_path"]
    ) as source:
        raw = source.read(1)
        extent = plotting_extent(
            raw,
            source.transform,
        )
        bounds = source.bounds
        raster_crs = source.crs
        nodata = source.nodata

    geometry = load_shelf_geometry(
        record["shapefile_path"],
        raster_crs,
    )

    shelf_geodata = gpd.GeoDataFrame(
        {"shelf": [record["shelf"]]},
        geometry=[geometry],
        crs=raster_crs,
    )

    with rasterio.open(
        record["raster_path"]
    ) as source:
        inside = geometry_mask(
            [geometry],
            out_shape=raw.shape,
            transform=source.transform,
            invert=True,
            all_touched=ALL_TOUCHED,
        )

    values = raw.astype(float, copy=True)

    nodata_mask = ~np.isfinite(values)

    if nodata is not None:
        try:
            if np.isfinite(nodata):
                nodata_mask |= raw == nodata
        except Exception:
            pass

    raw_display = values.copy()
    raw_display[nodata_mask] = np.nan
    raw_display[~inside] = np.nan

    clean = values.copy()
    clean[nodata_mask] = 0.0

    pond = (
        pond_mask_from_values(clean)
        & inside
    )

    binary_display = np.full(
        raw.shape,
        np.nan,
    )

    binary_display[inside] = 0.0
    binary_display[pond] = 1.0

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.6),
        constrained_layout=True,
    )

    image = axes[0].imshow(
        raw_display,
        extent=extent,
        origin="upper",
        cmap="gray_r",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    shelf_geodata.boundary.plot(
        ax=axes[0],
        color="#00BFC4",
        linewidth=1.0,
    )

    axes[0].set_title(
        "(a) Raster values inside shelf"
    )
    axes[0].set_xlabel("Easting, EPSG:3031 (m)")
    axes[0].set_ylabel("Northing, EPSG:3031 (m)")

    colorbar = figure.colorbar(
        image,
        ax=axes[0],
        shrink=0.82,
    )

    colorbar.set_label("Raster value")

    binary_cmap = mcolors.ListedColormap(
        ["#EEEEEE", "#0072B2"]
    )

    image_binary = axes[1].imshow(
        binary_display,
        extent=extent,
        origin="upper",
        cmap=binary_cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    shelf_geodata.boundary.plot(
        ax=axes[1],
        color="black",
        linewidth=1.0,
    )

    axes[1].set_title("(b) Binary pond mask")
    axes[1].set_xlabel("Easting, EPSG:3031 (m)")
    axes[1].set_ylabel("Northing, EPSG:3031 (m)")

    colorbar_binary = figure.colorbar(
        image_binary,
        ax=axes[1],
        shrink=0.82,
    )

    colorbar_binary.set_ticks(
        [0.25, 0.75]
    )

    colorbar_binary.set_ticklabels(
        ["Non-ponded", "Ponded"]
    )

    for axis in axes:
        axis.set_xlim(bounds.left, bounds.right)
        axis.set_ylim(bounds.bottom, bounds.top)
        axis.set_aspect("equal")
        style_axis(axis)

    n_inside = int(inside.sum())
    n_ponded = int(pond.sum())
    area_km2 = (
        n_ponded
        * abs(
            record.get(
                "pixel_area_km2",
                PIXEL_AREA_KM2_NOMINAL,
            )
        )
    )

    figure.text(
        0.5,
        -0.01,
        (
            f"{record['shelf']}, {record['year']}; "
            f"inside-shelf pixels = {n_inside:,}; "
            f"ponded pixels = {n_ponded:,}; "
            f"ponded area = {area_km2:.2f} km²; "
            f"ponded fraction = "
            f"{100.0 * n_ponded / n_inside:.4f}%"
        ),
        ha="center",
        va="top",
        fontsize=9,
    )

    save_figure(
        figure,
        "Fig_S01_raster_preprocessing_example",
    )


#FIGURE 2

def make_figure_s2(inventory):
    shelves = [
        "Cook",
        "Fimbul",
        "Mertz",
        "Shackleton",
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 10.0),
        constrained_layout=True,
    )

    axes = axes.ravel()
    summary_rows = []

    for panel_index, (
        axis,
        shelf,
    ) in enumerate(
        zip(axes, shelves)
    ):
        subset = inventory.loc[
            inventory["shelf"] == shelf
        ].sort_values("year")

        if subset.empty:
            axis.text(
                0.5,
                0.5,
                f"{shelf}: unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue

        record = subset.loc[
            subset["year"]
            == EXAMPLE_YEAR
        ]

        if record.empty:
            record = subset.tail(1)

        record = record.iloc[0]

        with rasterio.open(
            record["raster_path"]
        ) as source:
            raster_crs = source.crs
            raster_footprint = box(
                source.bounds.left,
                source.bounds.bottom,
                source.bounds.right,
                source.bounds.top,
            )

        geometry = load_shelf_geometry(
            record["shapefile_path"],
            raster_crs,
        )

        common_geometry = geometry.intersection(
            raster_footprint
        )

        shelf_geodata = gpd.GeoDataFrame(
            {"name": [shelf]},
            geometry=[geometry],
            crs=raster_crs,
        )

        common_geodata = gpd.GeoDataFrame(
            {"name": [shelf]},
            geometry=[common_geometry],
            crs=raster_crs,
        )

        footprint_geodata = gpd.GeoDataFrame(
            {"name": ["footprint"]},
            geometry=[raster_footprint],
            crs=raster_crs,
        )

        common_geodata.plot(
            ax=axis,
            color="#FDD0A2",
            edgecolor="#E6550D",
            linewidth=0.5,
            alpha=0.65,
        )

        footprint_geodata.boundary.plot(
            ax=axis,
            color="#CC79A7",
            linestyle="--",
            linewidth=1.0,
        )

        shelf_geodata.boundary.plot(
            ax=axis,
            color="black",
            linewidth=1.3,
        )

        bounds = expand_bounds(
            geometry.union(
                raster_footprint
            ).bounds
        )

        axis.set_xlim(bounds[0], bounds[2])
        axis.set_ylim(bounds[1], bounds[3])
        axis.set_aspect("equal")
        axis.set_title(
            f"{chr(97 + panel_index)}. {shelf}",
            loc="left",
            fontweight="bold",
        )

        axis.set_xlabel("Projected x")
        axis.set_ylabel("Projected y")
        style_axis(axis)

        shelf_area = float(geometry.area)
        common_area = float(common_geometry.area)

        coverage = (
            common_area / shelf_area
            if shelf_area > 0
            else np.nan
        )

        axis.text(
            0.02,
            0.02,
            (
                f"Common support / shelf: "
                f"{100.0 * coverage:.1f}%"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.9,
            },
        )

        summary_rows.append(
            {
                "shelf": shelf,
                "year": int(record["year"]),
                "shelf_area_m2": shelf_area,
                "common_support_area_m2": common_area,
                "common_support_fraction": coverage,
                "raster_path": record["raster_path"],
                "shapefile_path": record["shapefile_path"],
            }
        )

    figure.legend(
        handles=[
            Patch(
                facecolor="#FDD0A2",
                edgecolor="#E6550D",
                label="Common spatial support",
            ),
            Line2D(
                [0],
                [0],
                color="#CC79A7",
                linestyle="--",
                label="Melt-raster footprint",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                label="Shelf outline",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    atomic_csv_write(
        pd.DataFrame(summary_rows),
        TABLE_DIR
        / "Fig_S02_common_support_summary.csv",
    )

    save_figure(
        figure,
        "Fig_S02_common_support_partial_shelves",
    )

# FIGURE 3

def binned_median_iqr(
    data,
    x_column,
    y_column,
    number_bins=16,
):
    subset = data.loc[
        np.isfinite(data[x_column])
        & np.isfinite(data[y_column])
        & (data[x_column] > 0)
    ].copy()

    if len(subset) < 10:
        return pd.DataFrame()

    subset["_bin"] = pd.qcut(
        subset[x_column].rank(method="first"),
        q=min(number_bins, len(subset)),
        duplicates="drop",
    )

    return (
        subset.groupby(
            "_bin",
            observed=True,
        )
        .agg(
            x_median=(x_column, "median"),
            y_median=(y_column, "median"),
            y_q25=(
                y_column,
                lambda values: np.nanquantile(
                    values,
                    0.25,
                ),
            ),
            y_q75=(
                y_column,
                lambda values: np.nanquantile(
                    values,
                    0.75,
                ),
            ),
        )
        .reset_index(drop=True)
    )


def make_figure_s3(primary):
    data = primary.loc[
        np.isfinite(
            primary[
                "ponded_ponded_neighbor_pairs"
            ]
        )
        & np.isfinite(
            primary[
                "expected_ponded_ponded_pairs"
            ]
        )
        & np.isfinite(primary["r_pp"])
        & (
            primary[
                "ponded_ponded_neighbor_pairs"
            ]
            > 0
        )
        & (
            primary[
                "expected_ponded_ponded_pairs"
            ]
            > 0
        )
        & (primary["r_pp"] > 0)
        & (primary["ponded_fraction"] > 0)
    ].copy()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.2),
        constrained_layout=True,
    )

    axes[0].scatter(
        data[
            "expected_ponded_ponded_pairs"
        ],
        data[
            "ponded_ponded_neighbor_pairs"
        ],
        s=28,
        alpha=0.45,
        color=COLOR_POND,
        edgecolor="white",
        linewidth=0.35,
        rasterized=True,
    )

    minimum = min(
        data[
            "expected_ponded_ponded_pairs"
        ].min(),
        data[
            "ponded_ponded_neighbor_pairs"
        ].min(),
    )

    maximum = max(
        data[
            "expected_ponded_ponded_pairs"
        ].max(),
        data[
            "ponded_ponded_neighbor_pairs"
        ].max(),
    )

    axes[0].plot(
        [minimum, maximum],
        [minimum, maximum],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="1:1 random-mixing expectation",
    )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(
        "Expected ponded–ponded neighbor pairs"
    )
    axes[0].set_ylabel(
        "Observed ponded–ponded neighbor pairs"
    )
    axes[0].set_title(
        "A. Observed vs expected adjacency"
    )
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].scatter(
        data["ponded_fraction"],
        data["r_pp"],
        s=28,
        alpha=0.40,
        color=COLOR_POND,
        edgecolor="white",
        linewidth=0.35,
        rasterized=True,
    )

    binned = binned_median_iqr(
        data,
        "ponded_fraction",
        "r_pp",
    )

    if not binned.empty:
        axes[1].plot(
            binned["x_median"],
            binned["y_median"],
            color="black",
            linewidth=2.0,
            label="Binned median",
        )

        axes[1].fill_between(
            binned["x_median"],
            binned["y_q25"],
            binned["y_q75"],
            color="black",
            alpha=0.12,
            label="Interquartile range",
        )

    axes[1].axhline(
        1,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$R_{PP}=1$ random mixing",
    )

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].xaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axes[1].set_xlabel(
        "Observed ponded fraction"
    )
    axes[1].set_ylabel(
        r"Ponded–ponded enrichment ratio, $R_{PP}$"
    )
    axes[1].set_title(
        "B. Local clustering enrichment"
    )
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    save_figure(
        figure,
        "Fig_S03_ponded_pixel_adjacency_enrichment",
    )

#FIGURE S5

def make_figure_s4(primary):
    data = primary.loc[
        np.isfinite(primary["moran_i_rook"])
        & (primary["ponded_fraction"] > 0)
    ].copy()

    binned = binned_median_iqr(
        data,
        "ponded_fraction",
        "moran_i_rook",
    )

    figure, axis = plt.subplots(
        figsize=(9.0, 6.2),
        constrained_layout=True,
    )

    axis.scatter(
        data["ponded_fraction"],
        data["moran_i_rook"],
        s=30,
        alpha=0.65,
        color=COLOR_POND,
        edgecolor="none",
        rasterized=True,
        label="Shelf-year observations",
    )

    if not binned.empty:
        axis.fill_between(
            binned["x_median"],
            binned["y_q25"],
            binned["y_q75"],
            color=COLOR_YELLOW,
            alpha=0.22,
            label="Binned interquartile range",
        )

        axis.plot(
            binned["x_median"],
            binned["y_median"],
            color=COLOR_ORANGE,
            marker="o",
            markersize=3,
            linewidth=2.2,
            label="Binned median Moran's I",
        )

    axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="No spatial autocorrelation",
    )

    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axis.set_xlabel("Observed ponded fraction")
    axis.set_ylabel("Moran's I")
    axis.set_title(
        "Surface meltwater ponds are spatially clustered"
    )
    axis.legend(frameon=True, loc="lower right")
    style_axis(axis)

    positive = int(
        (data["moran_i_rook"] > 0).sum()
    )

    total = int(len(data))

    median = float(
        data["moran_i_rook"].median()
    )

    axis.text(
        0.02,
        0.97,
        (
            f"{positive:,} of {total:,} shelf-years "
            f"({100.0 * positive / total:.1f}%) have Moran's I > 0\n"
            f"Median Moran's I = {median:.2f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )

    save_figure(
        figure,
        "Fig_S04_morans_i",
    )


#FIGURE S5

def make_figure_s5(
    primary,
    correlogram,
):
    scene_classes = primary[
        [
            "shelf",
            "year",
            "raster_path",
            "ponded_fraction",
        ]
    ].drop_duplicates()

    scene_classes["ponding_class"] = pd.qcut(
        scene_classes[
            "ponded_fraction"
        ].rank(method="first"),
        q=3,
        labels=[
            "Low ponding",
            "Medium ponding",
            "High ponding",
        ],
    )

    data = correlogram.merge(
        scene_classes,
        on=[
            "shelf",
            "year",
            "raster_path",
        ],
        how="inner",
    )

    data = data.loc[
        np.isfinite(data["distance_m"])
        & np.isfinite(data["lag_r_pp"])
        & (data["distance_m"] > 0)
        & (data["lag_r_pp"] > 0)
    ].copy()

    colors = {
        "Low ponding": "#56B4E9",
        "Medium ponding": "#E69F00",
        "High ponding": "#D55E00",
    }

    summary = (
        data.groupby(
            [
                "ponding_class",
                "distance_m",
            ],
            observed=True,
        )
        .agg(
            rpp_median=("lag_r_pp", "median"),
            rpp_q25=(
                "lag_r_pp",
                lambda values: np.nanquantile(
                    values,
                    0.25,
                ),
            ),
            rpp_q75=(
                "lag_r_pp",
                lambda values: np.nanquantile(
                    values,
                    0.75,
                ),
            ),
            n=("lag_r_pp", "size"),
        )
        .reset_index()
        .sort_values(
            [
                "ponding_class",
                "distance_m",
            ]
        )
    )

    atomic_csv_write(
        summary,
        TABLE_DIR
        / "Fig_S05_lagged_enrichment_summary.csv",
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 6.2),
        constrained_layout=True,
    )

    for ponding_class, group in summary.groupby(
        "ponding_class",
        observed=True,
    ):
        color = colors[str(ponding_class)]
        distance_km = group["distance_m"] / 1000.0

        axis.plot(
            distance_km,
            group["rpp_median"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=str(ponding_class),
        )

        axis.fill_between(
            distance_km,
            group["rpp_q25"],
            group["rpp_q75"],
            color=color,
            alpha=0.18,
        )

    axis.axhline(
        1,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$R_{PP}(h)=1$ random mixing",
    )

    axis.set_yscale("log")
    axis.set_xlabel("Lag distance (km)")
    axis.set_ylabel(
        r"Lagged ponded–ponded enrichment, $R_{PP}(h)$"
    )
    axis.legend(frameon=False)
    style_axis(axis)

    save_figure(
        figure,
        "Fig_S05_lagged_ponded_pixel_enrichment",
    )


#FIGURE S6

def local_quantile_curve(
    data,
    x_column,
    y_column,
    quantile,
    grid=None,
    shelf_balanced=False,
    shelf_column="shelf",
):
    subset = data.loc[
        np.isfinite(data[x_column])
        & np.isfinite(data[y_column])
        & (data[x_column] > 0)
        & (data[y_column] >= 0)
    ].copy()

    if len(subset) < KNEE_MIN_LOCAL_POINTS:
        return pd.DataFrame()

    log_x = np.log10(
        subset[x_column].to_numpy(float)
    )

    y = subset[y_column].to_numpy(float)
    shelves = subset[shelf_column].astype(str).to_numpy()

    if grid is None:
        grid = np.linspace(
            log_x.min(),
            log_x.max(),
            KNEE_GRID_SIZE,
        )

    rows = []
    half_width = KNEE_LOCAL_WINDOW_DEX / 2.0

    for grid_value in grid:
        local_mask = (
            np.abs(
                log_x - grid_value
            )
            <= half_width
        )

        if local_mask.sum() < KNEE_MIN_LOCAL_POINTS:
            indices = np.argsort(
                np.abs(
                    log_x - grid_value
                )
            )[
                :KNEE_MIN_LOCAL_POINTS
            ]
        else:
            indices = np.flatnonzero(
                local_mask
            )

        if len(indices) < KNEE_MIN_LOCAL_POINTS:
            continue

        local_y = y[indices]

        if shelf_balanced:
            local_shelves = shelves[indices]

            (
                unique_shelves,
                inverse,
                counts,
            ) = np.unique(
                local_shelves,
                return_inverse=True,
                return_counts=True,
            )

            if (
                len(unique_shelves)
                < KNEE_MIN_LOCAL_SHELVES
            ):
                continue

            weights = 1.0 / counts[inverse]

            y_quantile = weighted_quantile(
                local_y,
                [quantile],
                weights,
            )[0]

        else:
            y_quantile = float(
                np.nanquantile(
                    local_y,
                    quantile,
                )
            )

        rows.append(
            {
                "log_x": float(grid_value),
                "x": float(10**grid_value),
                "y": float(y_quantile),
                "quantile": float(quantile),
                "n_local": int(len(indices)),
            }
        )

    return pd.DataFrame(rows)


def make_figure_s6(primary):
    data = primary.loc[
        np.isfinite(
            primary[
                "largest_component_area_km2"
            ]
        )
        & np.isfinite(
            primary[
                "largest_component_fraction_of_ponded"
            ]
        )
        & (primary["ponded_fraction"] > 0)
    ].copy()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.2),
        constrained_layout=True,
    )

    specifications = [
        (
            "largest_component_area_km2",
            "Largest connected pond cluster area (km²)",
            "A. Largest connected cluster area",
            COLOR_POND,
        ),
        (
            "largest_component_fraction_of_ponded",
            "Largest cluster / ponded pixels",
            "B. Dominance of largest connected component",
            COLOR_GREEN,
        ),
    ]

    quantiles = [
        (0.50, "-", "Local median"),
        (0.75, "--", "Local 75th percentile"),
        (0.90, ":", "Local 90th percentile"),
    ]

    for axis, (
        response,
        ylabel,
        title,
        point_color,
    ) in zip(
        axes,
        specifications,
    ):
        axis.scatter(
            data["ponded_fraction"],
            data[response],
            s=28,
            alpha=0.35,
            color=point_color,
            edgecolor="white",
            linewidth=0.35,
            rasterized=True,
            label="Shelf-year observations",
        )

        for quantile, linestyle, label in quantiles:
            curve = local_quantile_curve(
                data,
                "ponded_fraction",
                response,
                quantile,
                shelf_balanced=False,
            )

            if curve.empty:
                continue

            axis.plot(
                curve["x"],
                curve["y"],
                color=COLOR_ORANGE,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
            )

        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                percent_formatter
            )
        )
        axis.set_xlabel("Observed ponded fraction")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(frameon=False)
        style_axis(axis)

    axes[1].set_ylim(-0.02, 1.02)
    axes[1].yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0)
    )

    save_figure(
        figure,
        "Fig_S06_connected_component_transition",
    )

# Empirical knee calculation for S7 and S8

def geometric_knee(curve):
    x = curve["x"].to_numpy(float)
    y = curve["y"].to_numpy(float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 10:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    log_x = np.log10(x)
    y_monotonic = np.maximum.accumulate(y)

    x_range = np.ptp(log_x)
    y_range = np.ptp(y_monotonic)

    if x_range <= 0 or y_range <= 0:
        return None

    x_normalized = (
        log_x - log_x.min()
    ) / x_range

    y_normalized = (
        y_monotonic - y_monotonic.min()
    ) / y_range

    denominator = np.hypot(
        y_normalized[-1] - y_normalized[0],
        x_normalized[-1] - x_normalized[0],
    )

    if denominator <= 0:
        return None

    distances = np.abs(
        (
            y_normalized[-1]
            - y_normalized[0]
        )
        * x_normalized
        - (
            x_normalized[-1]
            - x_normalized[0]
        )
        * y_normalized
        + x_normalized[-1]
        * y_normalized[0]
        - y_normalized[-1]
        * x_normalized[0]
    ) / denominator

    edge_count = max(
        1,
        int(
            np.ceil(
                0.05 * len(distances)
            )
        ),
    )

    candidates = np.arange(
        edge_count,
        len(distances) - edge_count,
    )

    if len(candidates) == 0:
        return None

    knee_index = int(
        candidates[
            np.argmax(
                distances[candidates]
            )
        ]
    )

    return {
        "knee_x": float(x[knee_index]),
        "knee_y": float(y[knee_index]),
        "knee_distance": float(
            distances[knee_index]
        ),
    }


def estimate_knee(
    data,
    shelf_column="shelf",
    fixed_grid=None,
):
    subset = data.loc[
        np.isfinite(
            data["ponded_fraction"]
        )
        & np.isfinite(
            data[
                "largest_component_fraction_of_valid"
            ]
        )
        & (data["ponded_fraction"] > 0)
        & data[shelf_column].notna()
    ].copy()

    if len(subset) < KNEE_MIN_LOCAL_POINTS:
        return None

    if fixed_grid is None:
        fixed_grid = np.linspace(
            np.log10(
                subset["ponded_fraction"]
            ).min(),
            np.log10(
                subset["ponded_fraction"]
            ).max(),
            KNEE_GRID_SIZE,
        )

    curve_tables = []
    knee_rows = []

    for quantile in KNEE_QUANTILES:
        curve = local_quantile_curve(
            subset,
            "ponded_fraction",
            "largest_component_fraction_of_valid",
            quantile,
            grid=fixed_grid,
            shelf_balanced=True,
            shelf_column=shelf_column,
        )

        if curve.empty:
            continue

        knee = geometric_knee(curve)

        if knee is None:
            continue

        curve_tables.append(curve)

        knee_rows.append(
            {
                "quantile": quantile,
                **knee,
            }
        )

    knees = pd.DataFrame(knee_rows)

    if len(knees) < 3:
        return None

    knees = knees.sort_values(
        "knee_x"
    ).reset_index(drop=True)

    log_knees = np.log10(
        knees["knee_x"]
    )

    gaps = np.diff(log_knees)

    split_index = int(
        np.argmax(gaps)
    ) + 1

    initial = knees.iloc[
        :split_index
    ].copy()

    extensive = knees.iloc[
        split_index:
    ].copy()

    if initial.empty or extensive.empty:
        return None

    initial["group"] = "initial"
    extensive["group"] = "extensive"

    knees = pd.concat(
        [initial, extensive],
        ignore_index=True,
    )

    return {
        "empirical_knee": float(
            np.median(
                extensive["knee_x"]
            )
        ),
        "initial_zone_low": float(
            initial["knee_x"].min()
        ),
        "initial_zone_high": float(
            initial["knee_x"].max()
        ),
        "extensive_zone_low": float(
            extensive["knee_x"].min()
        ),
        "extensive_zone_high": float(
            extensive["knee_x"].max()
        ),
        "curves": pd.concat(
            curve_tables,
            ignore_index=True,
        ),
        "knees": knees,
        "fixed_grid": np.asarray(
            fixed_grid,
            dtype=float,
        ),
    }


def bootstrap_one(
    data,
    seed,
    fixed_grid,
):
    generator = np.random.default_rng(
        seed
    )

    shelves = np.asarray(
        sorted(
            data["shelf"].dropna().unique()
        )
    )

    groups = {
        shelf: group.copy()
        for shelf, group in data.groupby(
            "shelf",
            observed=True,
        )
    }

    sampled = generator.choice(
        shelves,
        size=len(shelves),
        replace=True,
    )

    pieces = []

    for index, shelf in enumerate(sampled):
        piece = groups[shelf].copy()
        piece["_bootstrap_shelf"] = (
            f"{index}_{shelf}"
        )
        pieces.append(piece)

    bootstrap_data = pd.concat(
        pieces,
        ignore_index=True,
    )

    estimate = estimate_knee(
        bootstrap_data,
        shelf_column="_bootstrap_shelf",
        fixed_grid=fixed_grid,
    )

    if estimate is None:
        return np.nan

    return estimate["empirical_knee"]


def run_knee_bootstrap(
    data,
    fixed_grid,
):
    seeds = [
        stable_seed(
            RANDOM_SEED,
            "knee_bootstrap",
            index,
        )
        for index in range(
            N_KNEE_BOOTSTRAP
        )
    ]

    values = []

    if N_WORKERS <= 1:
        for seed in tqdm(
            seeds,
            desc="Knee bootstrap",
        ):
            values.append(
                bootstrap_one(
                    data,
                    seed,
                    fixed_grid,
                )
            )

    else:
        context = mp.get_context(
            MP_CONTEXT
        )

        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    bootstrap_one,
                    data,
                    seed,
                    fixed_grid,
                )
                for seed in seeds
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Knee bootstrap",
            ):
                try:
                    values.append(
                        future.result()
                    )
                except Exception:
                    values.append(np.nan)

    output = pd.DataFrame(
        {
            "bootstrap_index": np.arange(
                len(values)
            ),
            "empirical_knee": values,
        }
    )

    return output.loc[
        np.isfinite(
            output["empirical_knee"]
        )
    ].reset_index(drop=True)


def cutoff_sensitivity(scenes):
    rows = []

    for cutoff in CUTOFFS_TO_TEST:
        subset = scenes.loc[
            np.isfinite(
                scenes["ponded_fraction"]
            )
            & np.isfinite(
                scenes[
                    "largest_component_fraction_of_valid"
                ]
            )
            & (
                scenes["n_ponded_pixels"]
                >= cutoff
            )
            & (
                scenes["ponded_fraction"]
                > 0
            )
        ].copy()

        estimate = estimate_knee(subset)

        rows.append(
            {
                "minimum_ponded_pixels": cutoff,
                "minimum_ponded_area_km2": (
                    cutoff
                    * PIXEL_AREA_KM2_NOMINAL
                ),
                "n_retained_scenes": len(subset),
                "n_shelves": (
                    subset["shelf"].nunique()
                ),
                "empirical_knee": (
                    estimate["empirical_knee"]
                    if estimate is not None
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def leave_one_shelf_out(
    primary,
    full_knee,
):
    rows = []

    for shelf in sorted(
        primary["shelf"].dropna().unique()
    ):
        subset = primary.loc[
            primary["shelf"] != shelf
        ].copy()

        estimate = estimate_knee(subset)

        knee = (
            estimate["empirical_knee"]
            if estimate is not None
            else np.nan
        )

        rows.append(
            {
                "excluded_shelf": shelf,
                "full_empirical_knee": full_knee,
                "leave_one_out_empirical_knee": knee,
                "knee_change": (
                    knee - full_knee
                    if np.isfinite(knee)
                    else np.nan
                ),
                "knee_change_percentage_points": (
                    100.0
                    * (knee - full_knee)
                    if np.isfinite(knee)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


#FIGURE S7

def make_figure_s7(knee_estimate):
    curves = knee_estimate["curves"]
    knees = knee_estimate["knees"]

    initial = knees.loc[
        knees["group"] == "initial"
    ]

    extensive = knees.loc[
        knees["group"] == "extensive"
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        constrained_layout=True,
    )

    quantiles = sorted(
        curves["quantile"].unique()
    )

    color_map = plt.get_cmap(
        "viridis"
    )

    for index, quantile in enumerate(
        quantiles
    ):
        curve = curves.loc[
            np.isclose(
                curves["quantile"],
                quantile,
            )
        ].sort_values("x")

        color = color_map(
            index
            / max(
                len(quantiles) - 1,
                1,
            )
        )

        axes[0].plot(
            curve["x"],
            curve["y"],
            color=color,
            linewidth=1.6,
            label=f"q={quantile:.2f}",
        )

        knee = knees.loc[
            np.isclose(
                knees["quantile"],
                quantile,
            )
        ]

        if not knee.empty:
            marker = (
                "s"
                if knee["group"].iloc[0]
                == "extensive"
                else "o"
            )

            axes[0].scatter(
                knee["knee_x"].iloc[0],
                knee["knee_y"].iloc[0],
                marker=marker,
                s=55,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
            )

    axes[0].axvspan(
        knee_estimate["initial_zone_low"],
        knee_estimate["initial_zone_high"],
        color=COLOR_POND,
        alpha=0.10,
        label="Initial-growth knee zone",
    )

    axes[0].axvspan(
        knee_estimate["extensive_zone_low"],
        knee_estimate["extensive_zone_high"],
        color=COLOR_YELLOW,
        alpha=0.10,
        label="Extensive-growth knee zone",
    )

    axes[0].axvline(
        knee_estimate["empirical_knee"],
        color="black",
        linewidth=1.8,
        label=(
            "Median extensive-growth knee\n"
            f"{100.0 * knee_estimate['empirical_knee']:.3f}%"
        ),
    )

    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axes[0].set_xlabel("Observed ponded fraction")
    axes[0].set_ylabel(
        "Largest cluster / valid shelf pixels"
    )
    axes[0].set_title(
        "A. Shelf-balanced quantile curves and knees"
    )
    axes[0].legend(
        frameon=False,
        fontsize=7,
        ncol=2,
    )
    style_axis(axes[0])

    axes[1].axvspan(
        knee_estimate["initial_zone_low"],
        knee_estimate["initial_zone_high"],
        color=COLOR_POND,
        alpha=0.10,
    )

    axes[1].axvspan(
        knee_estimate["extensive_zone_low"],
        knee_estimate["extensive_zone_high"],
        color=COLOR_YELLOW,
        alpha=0.10,
    )

    axes[1].scatter(
        initial["knee_x"],
        initial["quantile"],
        color=COLOR_POND,
        marker="o",
        s=65,
        label="Initial growth",
    )

    axes[1].scatter(
        extensive["knee_x"],
        extensive["quantile"],
        color=COLOR_ORANGE,
        marker="s",
        s=65,
        label="Extensive growth",
    )

    axes[1].axvline(
        knee_estimate["empirical_knee"],
        color="black",
        linewidth=1.8,
        label="Median extensive knee",
    )

    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axes[1].set_xlabel(
        "Quantile-specific knee ponded fraction"
    )
    axes[1].set_ylabel("Local response quantile")
    axes[1].set_title(
        "B. Initial- and extensive-growth groups"
    )
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    save_figure(
        figure,
        "Fig_S07_quantile_specific_empirical_knee",
    )

#Figure S8

def make_figure_s8(
    bootstrap,
    cutoff,
    leave_one_out,
    point_estimate,
):
    values = bootstrap[
        "empirical_knee"
    ].dropna().to_numpy(float)

    interval_68 = np.quantile(
        values,
        [0.158655, 0.841345],
    )

    interval_95 = np.quantile(
        values,
        [0.025, 0.975],
    )

    figure_height = max(
        10.5,
        0.25 * len(leave_one_out),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, figure_height),
        constrained_layout=True,
    )

    axes[0, 0].hist(
        values,
        bins=35,
        color="#357C8E",
        alpha=0.82,
        edgecolor="white",
    )

    axes[0, 0].axvspan(
        interval_95[0],
        interval_95[1],
        color="0.7",
        alpha=0.18,
        label="Central 95% interval",
    )

    axes[0, 0].axvspan(
        interval_68[0],
        interval_68[1],
        color=COLOR_YELLOW,
        alpha=0.22,
        label="Central 68% interval",
    )

    axes[0, 0].axvline(
        point_estimate,
        color="black",
        linewidth=1.8,
        label="Full-data knee",
    )

    axes[0, 0].xaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axes[0, 0].set_xlabel(
        "Shelf-bootstrap extensive-growth knee"
    )
    axes[0, 0].set_ylabel("Bootstrap replicates")
    axes[0, 0].set_title("A. Whole-shelf bootstrap")
    axes[0, 0].legend(frameon=False)
    style_axis(axes[0, 0])

    valid_cutoff = cutoff.loc[
        np.isfinite(cutoff["empirical_knee"])
    ]

    axes[0, 1].plot(
        valid_cutoff[
            "minimum_ponded_pixels"
        ],
        valid_cutoff["empirical_knee"],
        color=COLOR_POND,
        marker="o",
        linewidth=2.0,
    )

    axes[0, 1].axvline(
        PRIMARY_MIN_PONDED_PIXELS,
        color=COLOR_ORANGE,
        linestyle="--",
        linewidth=1.4,
    )

    axes[0, 1].axhline(
        point_estimate,
        color="black",
        linestyle=":",
        linewidth=1.4,
    )

    axes[0, 1].yaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axes[0, 1].set_xlabel(
        "Minimum retained ponded-pixel count"
    )
    axes[0, 1].set_ylabel("Extensive-growth knee")
    axes[0, 1].set_title(
        "B. Minimum-pixel sensitivity"
    )
    style_axis(axes[0, 1])

    leave_plot = (
        leave_one_out.dropna(
            subset=[
                "leave_one_out_empirical_knee"
            ]
        )
        .sort_values(
            "leave_one_out_empirical_knee"
        )
        .reset_index(drop=True)
    )

    positions = np.arange(len(leave_plot))

    axes[1, 0].hlines(
        positions,
        xmin=np.minimum(
            leave_plot[
                "leave_one_out_empirical_knee"
            ],
            point_estimate,
        ),
        xmax=np.maximum(
            leave_plot[
                "leave_one_out_empirical_knee"
            ],
            point_estimate,
        ),
        color="0.75",
        linewidth=1.0,
    )

    axes[1, 0].scatter(
        leave_plot[
            "leave_one_out_empirical_knee"
        ],
        positions,
        color=COLOR_GREEN,
        s=38,
        zorder=3,
    )

    axes[1, 0].axvline(
        point_estimate,
        color="black",
        linestyle="--",
        linewidth=1.3,
    )

    axes[1, 0].set_yticks(positions)
    axes[1, 0].set_yticklabels(
        leave_plot["excluded_shelf"],
        fontsize=7,
    )
    axes[1, 0].xaxis.set_major_formatter(
        mticker.FuncFormatter(
            percent_formatter
        )
    )
    axes[1, 0].set_xlabel(
        "Knee after excluding indicated shelf"
    )
    axes[1, 0].set_ylabel("Excluded shelf")
    axes[1, 0].set_title(
        "C. Leave-one-shelf-out estimates"
    )
    style_axis(axes[1, 0])

    influence = (
        leave_one_out.dropna(
            subset=[
                "knee_change_percentage_points"
            ]
        )
        .sort_values(
            "knee_change_percentage_points"
        )
        .reset_index(drop=True)
    )

    positions = np.arange(len(influence))

    colors = np.where(
        influence[
            "knee_change_percentage_points"
        ]
        >= 0,
        COLOR_ORANGE,
        COLOR_POND,
    )

    axes[1, 1].barh(
        positions,
        influence[
            "knee_change_percentage_points"
        ],
        color=colors,
        alpha=0.85,
    )

    axes[1, 1].axvline(
        0,
        color="black",
        linewidth=1.1,
    )

    axes[1, 1].set_yticks(positions)
    axes[1, 1].set_yticklabels(
        influence["excluded_shelf"],
        fontsize=7,
    )
    axes[1, 1].set_xlabel(
        "Change in knee (percentage points)"
    )
    axes[1, 1].set_ylabel("Excluded shelf")
    axes[1, 1].set_title(
        "D. Signed shelf influence"
    )
    style_axis(axes[1, 1])

    save_figure(
        figure,
        "Fig_S08_empirical_knee_robustness",
    )

    return {
        "bootstrap_68_low": float(
            interval_68[0]
        ),
        "bootstrap_68_high": float(
            interval_68[1]
        ),
        "bootstrap_95_low": float(
            interval_95[0]
        ),
        "bootstrap_95_high": float(
            interval_95[1]
        ),
    }

# Figure S9

def effective_number(weights):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights[
        np.isfinite(weights)
        & (weights > 0)
    ]

    if len(weights) == 0:
        return np.nan

    proportions = (
        weights / weights.sum()
    )

    denominator = np.sum(
        proportions**2
    )

    return (
        1.0 / denominator
        if denominator > 0
        else np.nan
    )


def make_figure_s9():
    raw = read_table(
        COMMON_SUPPORT_ERA5_TABLE
    )

    required = [
        "shelf",
        "era5_t2m_djf_c",
        "n_pixels_30m",
        "y_ponded_pixels",
    ]

    missing = [
        column
        for column in required
        if column not in raw.columns
    ]

    if missing:
        raise KeyError(
            f"Common-support table is missing {missing}"
        )

    data = pd.DataFrame(
        {
            "shelf": raw["shelf"].map(
                canonical_shelf_name
            ),
            "temperature_c": pd.to_numeric(
                raw["era5_t2m_djf_c"],
                errors="coerce",
            ),
            "trials": pd.to_numeric(
                raw["n_pixels_30m"],
                errors="coerce",
            ),
            "successes": pd.to_numeric(
                raw["y_ponded_pixels"],
                errors="coerce",
            ),
        }
    )

    data = data.loc[
        np.isfinite(data["temperature_c"])
        & np.isfinite(data["trials"])
        & (data["trials"] > 0)
        & np.isfinite(data["successes"])
        & (data["successes"] >= 0)
        & (data["successes"] <= data["trials"])
    ].copy()

    minimum = (
        np.floor(
            data["temperature_c"].min()
            / TEMPERATURE_BIN_WIDTH_C
        )
        * TEMPERATURE_BIN_WIDTH_C
    )

    maximum = (
        np.ceil(
            data["temperature_c"].max()
            / TEMPERATURE_BIN_WIDTH_C
        )
        * TEMPERATURE_BIN_WIDTH_C
    )

    edges = np.arange(
        minimum,
        maximum
        + TEMPERATURE_BIN_WIDTH_C,
        TEMPERATURE_BIN_WIDTH_C,
    )

    data["temperature_bin"] = pd.cut(
        data["temperature_c"],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    rows = []

    for temperature_bin, group in data.groupby(
        "temperature_bin",
        observed=True,
    ):
        total_trials = float(
            group["trials"].sum()
        )

        total_successes = float(
            group["successes"].sum()
        )

        shelf_weights = (
            group.groupby(
                "shelf",
                observed=True,
            )["trials"]
            .sum()
            .to_numpy(float)
        )

        rows.append(
            {
                "temperature_bin": str(
                    temperature_bin
                ),
                "temperature_midpoint_c": float(
                    temperature_bin.mid
                ),
                "n_rows": int(len(group)),
                "n_zero_rows": int(
                    (group["successes"] == 0).sum()
                ),
                "n_positive_rows": int(
                    (group["successes"] > 0).sum()
                ),
                "positive_row_percent": float(
                    100.0
                    * (
                        group["successes"]
                        > 0
                    ).mean()
                ),
                "pixel_pooled_fraction": (
                    total_successes / total_trials
                ),
                "n_shelves": int(
                    group["shelf"].nunique()
                ),
                "effective_number_of_shelves": (
                    effective_number(
                        shelf_weights
                    )
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        "temperature_midpoint_c"
    )

    atomic_csv_write(
        summary,
        TABLE_DIR
        / "Fig_S09_temperature_bin_summary.csv",
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    width = (
        TEMPERATURE_BIN_WIDTH_C
        * 0.90
    )

    axes[0].bar(
        summary["temperature_midpoint_c"],
        summary["n_zero_rows"],
        width=width,
        color="#D9D9D9",
        edgecolor="0.55",
        linewidth=0.3,
        label="Exactly 0% visible ponding",
    )

    axes[0].bar(
        summary["temperature_midpoint_c"],
        summary["n_positive_rows"],
        bottom=summary["n_zero_rows"],
        width=width,
        color=COLOR_YELLOW,
        edgecolor="0.55",
        linewidth=0.3,
        label="Positive visible ponding",
    )

    axes[0].set_ylabel(
        "Shelf-cell-year observations"
    )
    axes[0].set_title(
        "A. Counts of zero and positive visible-ponding observations"
    )
    axes[0].legend(
        frameon=False,
        loc="upper left",
    )
    style_axis(axes[0])

    axes[1].bar(
        summary["temperature_midpoint_c"],
        summary["positive_row_percent"],
        width=width,
        color=COLOR_ORANGE,
        alpha=0.75,
        label=(
            "Observations with visible ponding"
        ),
    )

    axes[1].plot(
        summary["temperature_midpoint_c"],
        100.0
        * summary["pixel_pooled_fraction"],
        color=COLOR_POND,
        marker="o",
        linewidth=1.8,
        markersize=3,
        label=(
            "Pixel-pooled ponding fraction"
        ),
    )

    axes[1].set_ylabel(
        "Visible-ponding statistic (%)"
    )
    axes[1].set_xlabel(
        "ERA5 DJF 2 m temperature (°C)"
    )

    support_axis = axes[1].twinx()

    support_axis.plot(
        summary["temperature_midpoint_c"],
        summary["n_shelves"],
        color=COLOR_GREEN,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Observed shelves",
    )

    support_axis.plot(
        summary["temperature_midpoint_c"],
        summary[
            "effective_number_of_shelves"
        ],
        color=COLOR_PURPLE,
        marker="s",
        markersize=3,
        linestyle="--",
        linewidth=1.8,
        label=(
            "Effective shelves (pixel-weighted)"
        ),
    )

    support_axis.set_ylabel("Shelf support")

    axes[1].set_title(
        "B. Ponding prevalence, pooled ponding, and shelf support"
    )

    left_handles, left_labels = (
        axes[1].get_legend_handles_labels()
    )

    right_handles, right_labels = (
        support_axis.get_legend_handles_labels()
    )

    axes[1].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    style_axis(axes[1])
    support_axis.spines["top"].set_visible(
        False
    )

    save_figure(
        figure,
        "Fig_S09_temperature_distribution_and_geographic_support",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 88)
    print("RECOMPUTE SI FIGURES S1–S9")
    print("=" * 88)

    print(
        "Common-support table:",
        COMMON_SUPPORT_ERA5_TABLE,
    )

    print(
        "Raster root:",
        CROPPED_RASTER_ROOT,
    )

    print(
        "Shapefile directory:",
        SHAPEFILE_DIR,
    )

    print(
        "Output directory:",
        OUTPUT_DIR.resolve(),
    )

    inventory = build_inventory()

    atomic_csv_write(
        inventory,
        TABLE_DIR / "shelf_raster_inventory.csv",
    )

    make_figure_s1(inventory)
    make_figure_s2(inventory)

    scenes, correlogram, errors = (
        run_spatial_analysis(inventory)
    )

    if scenes.empty:
        raise RuntimeError(
            "No spatial scene metrics were generated."
        )

    primary = scenes.loc[
        np.isfinite(scenes["ponded_fraction"])
        & np.isfinite(
            scenes[
                "largest_component_fraction_of_valid"
            ]
        )
        & (
            scenes["n_ponded_pixels"]
            >= PRIMARY_MIN_PONDED_PIXELS
        )
        & (
            scenes["ponded_fraction"]
            > 0
        )
    ].copy()

    atomic_csv_write(
        primary,
        TABLE_DIR
        / "primary_spatial_scene_metrics.csv",
    )

    info(
        f"All scene rows: {len(scenes):,}"
    )

    info(
        f"Primary rows: {len(primary):,}"
    )

    info(
        f"Primary shelves: "
        f"{primary['shelf'].nunique():,}"
    )

    make_figure_s3(primary)
    make_figure_s4(primary)
    make_figure_s5(
        primary,
        correlogram,
    )
    make_figure_s6(primary)

    knee_estimate = estimate_knee(
        primary
    )

    if knee_estimate is None:
        raise RuntimeError(
            "Could not estimate the empirical knee."
        )

    atomic_csv_write(
        knee_estimate["curves"],
        TABLE_DIR
        / "shelf_balanced_local_quantile_curves.csv",
    )

    atomic_csv_write(
        knee_estimate["knees"],
        TABLE_DIR
        / "quantile_specific_knee_estimates.csv",
    )

    make_figure_s7(knee_estimate)

    bootstrap = run_knee_bootstrap(
        primary,
        knee_estimate["fixed_grid"],
    )

    cutoff = cutoff_sensitivity(scenes)

    leave_one_out = leave_one_shelf_out(
        primary,
        knee_estimate["empirical_knee"],
    )

    atomic_csv_write(
        bootstrap,
        TABLE_DIR
        / "empirical_knee_whole_shelf_bootstrap.csv",
    )

    atomic_csv_write(
        cutoff,
        TABLE_DIR
        / "minimum_pixel_sensitivity.csv",
    )

    atomic_csv_write(
        leave_one_out,
        TABLE_DIR
        / "leave_one_shelf_out_sensitivity.csv",
    )

    bootstrap_intervals = make_figure_s8(
        bootstrap,
        cutoff,
        leave_one_out,
        knee_estimate["empirical_knee"],
    )

    knee_summary = {
        "empirical_knee_fraction": (
            knee_estimate["empirical_knee"]
        ),
        "empirical_knee_percent": (
            100.0
            * knee_estimate["empirical_knee"]
        ),
        **bootstrap_intervals,
        "primary_minimum_ponded_pixels": (
            PRIMARY_MIN_PONDED_PIXELS
        ),
        "n_primary_scenes": int(
            len(primary)
        ),
        "n_primary_shelves": int(
            primary["shelf"].nunique()
        ),
    }

    atomic_json_write(
        knee_summary,
        TABLE_DIR
        / "empirical_knee_summary.json",
    )

    make_figure_s9()

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)

    print(
        "Figures:",
        FIGURE_DIR.resolve(),
    )

    print(
        "Tables:",
        TABLE_DIR.resolve(),
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()