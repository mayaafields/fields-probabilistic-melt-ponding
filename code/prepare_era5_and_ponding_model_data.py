#!/usr/bin/env python
# coding: utf-8

"""
Prepare ERA5 DJF temperature and Landsat visible-ponding observations.

This script creates the common-support dataset used by the
temperature-response models.

Observation: 

Each output row represents one:

    ice shelf × ERA5 grid cell × DJF year

For each row:

    n_pixels_30m
        Number of raster-grid pixels inside the rasterized common-support
        geometry.

    y_ponded_pixels
        Number of those pixels whose melt-raster value is exactly 1.

    observed_fraction
        y_ponded_pixels / n_pixels_30m.

    era5_t2m_djf_c
        ERA5 mean December–January–February 2 m air temperature in Celsius.

DJF convention: 

DJF year YYYY is:

    December YYYY-1 + January YYYY + February YYYY

Spatial support: 

For each shelf, ERA5 cell, and year, the geometry is:

    ERA5 cell
        ∩ ice-shelf polygon
        ∩ melt-raster rectangular footprint
        
Primary outputs: 

PREPROCESS_OUTPUT_DIR/
    era5/
        era5_annual_djf_t2m_celsius.nc
        era5_antarctic_grid.gpkg
        shelf_crops/
            <shelf>_era5_djf.nc
            <shelf>_era5_cells.gpkg

    observations/
        by_shelf/
            <shelf>_common_support_2006_2020.parquet
        observations_era5_ponding_common_support_2006_2020.parquet
        observations_era5_ponding_common_support_2006_2020.csv
        shelf_processing_results.csv
        failed_shelves.csv
        common_support_coverage_by_shelf_year.csv
        shelf_metadata_summary.csv

    metadata/
        preprocessing_metadata.json
        software_versions.json

Required variables: 

ERA5_MONTHLY_PATH
    Monthly ERA5 NetCDF containing 2 m temperature.

MELTWATER_RASTER_DIR
    Root directory containing corrected shelf/year Landsat rasters.

Optional variables: 

PREPROCESS_OUTPUT_DIR
    Default: data/processed/common_support

SHELF_METRICS_TABLE
    Table used to identify shelves and shelf-years.

OBS_YEAR_MIN
    Default: 2006

OBS_YEAR_MAX
    Default: 2020

ERA5_YEAR_MIN
    Default: 1979

ERA5_YEAR_MAX
    Default: 2025

N_WORKERS
    Default: 1

ALL_TOUCHED
    Rasterization option. Default: 0

OVERWRITE
    Recreate existing outputs. Default: 0

MAKE_DIAGNOSTIC_FIGURES
    Create the four-shelf common-support diagnostic. Default: 0

DIAGNOSTIC_YEAR
    Default: 2020
"""

import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import time
import warnings

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import CRS
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

import ponding_native_grid_utils as pngu

warnings.filterwarnings("ignore")

def env_bool(
    name,
    default="0",
):
    """Read a Boolean environment variable represented as 0 or 1."""

    value = os.environ.get(
        name,
        default,
    ).strip()

    if value not in {
        "0",
        "1",
    }:
        raise ValueError(
            f"{name} must be 0 or 1; received {value!r}."
        )

    return value == "1"


def require_environment_path(
    name,
    *,
    must_exist=True,
):
    """Read a required path environment variable."""

    value = os.environ.get(
        name,
        "",
    ).strip()

    if not value:
        raise ValueError(
            f"Required environment variable {name} is not set."
        )

    path = Path(value)

    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"{name} does not exist:\n{path}"
        )

    return path

# Configuration. Settings for successful run.

ERA5_MONTHLY_PATH = require_environment_path(
    "ERA5_MONTHLY_PATH"
)

MELTWATER_RASTER_DIR = require_environment_path(
    "MELTWATER_RASTER_DIR"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PREPROCESS_OUTPUT_DIR",
        "data/processed/common_support",
    )
)

SHELF_METRICS_TABLE_VALUE = os.environ.get(
    "SHELF_METRICS_TABLE",
    "",
).strip()

SHELF_METRICS_TABLE = (
    Path(SHELF_METRICS_TABLE_VALUE)
    if SHELF_METRICS_TABLE_VALUE
    else None
)

OBS_YEAR_MIN = int(
    os.environ.get(
        "OBS_YEAR_MIN",
        "2006",
    )
)

OBS_YEAR_MAX = int(
    os.environ.get(
        "OBS_YEAR_MAX",
        "2020",
    )
)

ERA5_YEAR_MIN = int(
    os.environ.get(
        "ERA5_YEAR_MIN",
        "1979",
    )
)

ERA5_YEAR_MAX = int(
    os.environ.get(
        "ERA5_YEAR_MAX",
        "2025",
    )
)

N_WORKERS = int(
    os.environ.get(
        "N_WORKERS",
        "1",
    )
)

ALL_TOUCHED = env_bool(
    "ALL_TOUCHED",
    "0",
)

OVERWRITE = env_bool(
    "OVERWRITE",
    "0",
)

MAKE_DIAGNOSTIC_FIGURES = env_bool(
    "MAKE_DIAGNOSTIC_FIGURES",
    "0",
)

DIAGNOSTIC_YEAR = int(
    os.environ.get(
        "DIAGNOSTIC_YEAR",
        "2020",
    )
)

ERA5_GRID_CRS = os.environ.get(
    "ERA5_GRID_CRS",
    "EPSG:3031",
)

ERA5_ANTARCTIC_LATITUDE_MAXIMUM = float(
    os.environ.get(
        "ERA5_ANTARCTIC_LATITUDE_MAXIMUM",
        "-45",
    )
)

RASTER_FILENAME_PATTERNS = [
    "ANT_WIDE_{year}_{next_year}_30m_raster.tif",
    "{year}_masked.tif",
]

ERA5_DIRECTORY = OUTPUT_DIR / "era5"

ERA5_SHELF_CROP_DIRECTORY = (
    ERA5_DIRECTORY
    / "shelf_crops"
)

OBSERVATION_DIRECTORY = (
    OUTPUT_DIR
    / "observations"
)

OBSERVATION_BY_SHELF_DIRECTORY = (
    OBSERVATION_DIRECTORY
    / "by_shelf"
)

METADATA_DIRECTORY = (
    OUTPUT_DIR
    / "metadata"
)

DIAGNOSTIC_DIRECTORY = (
    OUTPUT_DIR
    / "diagnostics"
)

for directory in [
    OUTPUT_DIR,
    ERA5_DIRECTORY,
    ERA5_SHELF_CROP_DIRECTORY,
    OBSERVATION_DIRECTORY,
    OBSERVATION_BY_SHELF_DIRECTORY,
    METADATA_DIRECTORY,
    DIAGNOSTIC_DIRECTORY,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

ERA5_DJF_PATH = (
    ERA5_DIRECTORY
    / "era5_annual_djf_t2m_celsius.nc"
)

ERA5_GRID_PATH = (
    ERA5_DIRECTORY
    / "era5_antarctic_grid.gpkg"
)

COMBINED_OBSERVATION_PARQUET = (
    OBSERVATION_DIRECTORY
    / (
        "observations_era5_ponding_"
        f"common_support_{OBS_YEAR_MIN}_{OBS_YEAR_MAX}.parquet"
    )
)

COMBINED_OBSERVATION_CSV = (
    COMBINED_OBSERVATION_PARQUET
    .with_suffix(".csv")
)

PROCESSING_RESULTS_PATH = (
    OBSERVATION_DIRECTORY
    / "shelf_processing_results.csv"
)

FAILED_SHELVES_PATH = (
    OBSERVATION_DIRECTORY
    / "failed_shelves.csv"
)

COVERAGE_SUMMARY_PATH = (
    OBSERVATION_DIRECTORY
    / "common_support_coverage_by_shelf_year.csv"
)

SHELF_METADATA_PATH = (
    OBSERVATION_DIRECTORY
    / "shelf_metadata_summary.csv"
)

PREPROCESSING_METADATA_PATH = (
    METADATA_DIRECTORY
    / "preprocessing_metadata.json"
)

SOFTWARE_VERSIONS_PATH = (
    METADATA_DIRECTORY
    / "software_versions.json"
)

# General code function helpers

def safe_filename(value):
    """Create a filesystem-safe name."""

    value = str(value).strip()

    value = value.replace(
        "–",
        "-",
    )

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return value.strip("_")


def file_sha256(
    path,
    block_size=1024 * 1024,
):
    """Calculate a file SHA-256 checksum."""

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        while True:
            block = file.read(
                block_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def atomic_csv_write(
    dataframe,
    path,
    *,
    index=False,
):
    """Write CSV atomically."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary,
        index=index,
    )

    temporary.replace(
        path
    )

    print(
        "[SAVED]",
        path,
        flush=True,
    )


def atomic_json_write(
    payload,
    path,
):
    """Write JSON atomically."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

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

    temporary.replace(
        path
    )

    print(
        "[SAVED]",
        path,
        flush=True,
    )


def save_netcdf(
    dataset,
    path,
    *,
    encoding=None,
):
    """Save NetCDF atomically, with a compression fallback."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    if encoding is None:
        encoding = {}

    try:
        dataset.to_netcdf(
            temporary,
            encoding=encoding,
        )

    except Exception as error:
        print(
            "[WARNING] Compressed NetCDF save failed:",
            repr(error),
            flush=True,
        )

        if temporary.exists():
            temporary.unlink()

        dataset.to_netcdf(
            temporary
        )

    if path.exists():
        path.unlink()

    temporary.replace(
        path
    )

    print(
        "[SAVED]",
        path,
        flush=True,
    )


def crs_equal(
    first,
    second,
):
    """Test coordinate-reference-system equivalence."""

    return (
        CRS.from_user_input(
            first
        )
        == CRS.from_user_input(
            second
        )
    )

# Shelf mapping and metadata

def read_shelf_names():
    """Identify shelves from a metrics table or utility mapping."""

    if SHELF_METRICS_TABLE is not None:
        if not SHELF_METRICS_TABLE.is_file():
            raise FileNotFoundError(
                f"SHELF_METRICS_TABLE not found:\n"
                f"{SHELF_METRICS_TABLE}"
            )

        if (
            SHELF_METRICS_TABLE
            .suffix
            .lower()
            == ".parquet"
        ):
            metrics = pd.read_parquet(
                SHELF_METRICS_TABLE
            )

        else:
            metrics = pd.read_csv(
                SHELF_METRICS_TABLE,
                low_memory=False,
            )

        if "shelf" not in metrics.columns:
            raise KeyError(
                "SHELF_METRICS_TABLE has no 'shelf' column."
            )

        if "status" in metrics.columns:
            status = (
                metrics[
                    "status"
                ]
                .astype(str)
                .str.upper()
            )

            metrics = metrics[
                status.isin(
                    [
                        "OK",
                        "SUCCESS",
                        "COMPLETE",
                    ]
                )
            ].copy()

        if "year" in metrics.columns:
            metrics[
                "year"
            ] = pd.to_numeric(
                metrics[
                    "year"
                ],
                errors="coerce",
            )

            metrics = metrics[
                metrics[
                    "year"
                ].between(
                    OBS_YEAR_MIN,
                    OBS_YEAR_MAX,
                    inclusive="both",
                )
            ].copy()

        shelves = sorted(
            metrics[
                "shelf"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        if not shelves:
            raise RuntimeError(
                "No shelves were found in SHELF_METRICS_TABLE."
            )

        return shelves

    if hasattr(
        pngu,
        "SHELF_REGION_MAP",
    ):
        shelves = sorted(
            str(value)
            for value
            in pngu.SHELF_REGION_MAP
        )

        if shelves:
            return shelves

    raise ValueError(
        "Set SHELF_METRICS_TABLE or provide "
        "SHELF_REGION_MAP in ponding_native_grid_utils.py."
    )


def get_region_for_shelf(
    shelf,
):
    """Return the configured broad region."""

    if hasattr(
        pngu,
        "SHELF_REGION_MAP",
    ):
        return pngu.SHELF_REGION_MAP.get(
            shelf,
            "Unknown region",
        )

    return "Unknown region"


def get_quadrant_for_shelf(
    shelf,
):
    """Return the configured quadrant."""

    if hasattr(
        pngu,
        "SHELF_REGION_MAP_QUADRANTS",
    ):
        return (
            pngu.SHELF_REGION_MAP_QUADRANTS.get(
                shelf,
                "Unknown quadrant",
            )
        )

    return "Unknown quadrant"

# ERA5 monthly-to-DJF processing

def find_coordinate_name(
    dataset,
    candidates,
    label,
):
    """Find a coordinate or dimension name."""

    for candidate in candidates:
        if (
            candidate in dataset.coords
            or candidate in dataset.dims
        ):
            return candidate

    raise ValueError(
        f"Could not find {label}. "
        f"Coordinates: {list(dataset.coords)}; "
        f"dimensions: {list(dataset.dims)}"
    )


def find_t2m_name(
    dataset,
):
    """Find the 2 m temperature variable."""

    candidates = [
        "t2m",
        "T2M",
        "2t",
        "temperature_2m",
        "air_temperature",
    ]

    for candidate in candidates:
        if candidate in dataset.data_vars:
            return candidate

    raise ValueError(
        "Could not find the 2 m temperature variable. "
        f"Available variables: {list(dataset.data_vars)}"
    )


def collapse_expver(
    dataset,
):
    """Collapse ERA5 expver when present."""

    if "expver" not in dataset.dims:
        return dataset

    if dataset.sizes[
        "expver"
    ] == 1:
        return dataset.isel(
            expver=0,
            drop=True,
        )

    return (
        dataset.bfill(
            "expver"
        )
        .isel(
            expver=0,
            drop=True,
        )
    )


def normalize_longitudes(
    dataset,
):
    """Normalize longitude coordinates to -180 through 180 degrees."""

    longitude = dataset[
        "longitude"
    ]

    if float(
        longitude.max()
    ) > 180:
        normalized = (
            (
                longitude
                + 180
            )
            % 360
        ) - 180

        dataset = dataset.assign_coords(
            longitude=normalized
        )

        dataset = dataset.sortby(
            "longitude"
        )

    return dataset


def convert_temperature_to_celsius(
    data_array,
):
    """Convert temperature to Celsius if stored in Kelvin."""

    units = str(
        data_array.attrs.get(
            "units",
            "",
        )
    ).strip().lower()

    if units in {
        "k",
        "kelvin",
        "degrees_kelvin",
        "degree_kelvin",
    }:
        output = (
            data_array
            - 273.15
        )

    else:
        representative = float(
            data_array.mean(
                skipna=True
            ).values
        )

        if representative > 100:
            output = (
                data_array
                - 273.15
            )
        else:
            output = data_array

    output.attrs.update(
        data_array.attrs
    )

    output.attrs[
        "units"
    ] = "degree_Celsius"

    return output


def prepare_era5_annual_djf(
    monthly_path,
    output_path,
    *,
    year_min,
    year_max,
    overwrite,
    day_weighted=True,
):
    """
    Convert monthly ERA5 temperature to annual DJF Celsius.

    DJF YYYY = Dec YYYY-1 + Jan YYYY + Feb YYYY.
    """

    if (
        output_path.exists()
        and not overwrite
    ):
        print(
            "[EXISTS]",
            output_path,
            flush=True,
        )

        return output_path

    dataset = xr.open_dataset(
        monthly_path
    )

    try:
        dataset = collapse_expver(
            dataset
        )

        time_name = find_coordinate_name(
            dataset,
            [
                "time",
                "valid_time",
                "date",
            ],
            "time coordinate",
        )

        longitude_name = find_coordinate_name(
            dataset,
            [
                "longitude",
                "lon",
                "x",
            ],
            "longitude coordinate",
        )

        latitude_name = find_coordinate_name(
            dataset,
            [
                "latitude",
                "lat",
                "y",
            ],
            "latitude coordinate",
        )

        temperature_name = find_t2m_name(
            dataset
        )

        rename = {}

        if time_name != "time":
            rename[
                time_name
            ] = "time"

        if longitude_name != "longitude":
            rename[
                longitude_name
            ] = "longitude"

        if latitude_name != "latitude":
            rename[
                latitude_name
            ] = "latitude"

        if rename:
            dataset = dataset.rename(
                rename
            )

        dataset = normalize_longitudes(
            dataset
        )

        dataset = dataset.sortby(
            "latitude"
        )

        temperature = convert_temperature_to_celsius(
            dataset[
                temperature_name
            ]
        )

        annual_arrays = []
        complete_years = []
        incomplete_rows = []

        for year in range(
            int(year_min),
            int(year_max) + 1,
        ):
            start = np.datetime64(
                f"{year - 1}-12-01"
            )

            stop = np.datetime64(
                f"{year}-03-01"
            )

            season = temperature.sel(
                time=(
                    (
                        temperature[
                            "time"
                        ]
                        >= start
                    )
                    & (
                        temperature[
                            "time"
                        ]
                        < stop
                    )
                )
            )

            season = season.sel(
                time=season[
                    "time"
                ].dt.month.isin(
                    [
                        12,
                        1,
                        2,
                    ]
                )
            )

            timestamps = pd.to_datetime(
                season[
                    "time"
                ].values
            )

            expected_month_year = {
                (
                    12,
                    year - 1,
                ),
                (
                    1,
                    year,
                ),
                (
                    2,
                    year,
                ),
            }

            found_month_year = {
                (
                    int(timestamp.month),
                    int(timestamp.year),
                )
                for timestamp
                in timestamps
            }

            if (
                found_month_year
                != expected_month_year
            ):
                incomplete_rows.append(
                    {
                        "djf_year": (
                            year
                        ),
                        "expected": str(
                            sorted(
                                expected_month_year
                            )
                        ),
                        "found": str(
                            sorted(
                                found_month_year
                            )
                        ),
                        "n_records": int(
                            len(
                                timestamps
                            )
                        ),
                    }
                )

                continue

            if day_weighted:
                days = (
                    season[
                        "time"
                    ]
                    .dt.days_in_month
                    .astype(
                        "float64"
                    )
                )

                weights = (
                    days
                    / days.sum()
                )

                seasonal_mean = (
                    season
                    * weights
                ).sum(
                    "time",
                    skipna=True,
                )

            else:
                seasonal_mean = (
                    season.mean(
                        "time",
                        skipna=True,
                    )
                )

            seasonal_mean = (
                seasonal_mean.expand_dims(
                    year=[
                        year
                    ]
                )
            )

            annual_arrays.append(
                seasonal_mean
            )

            complete_years.append(
                year
            )

        if not annual_arrays:
            raise RuntimeError(
                "No complete DJF seasons were found."
            )

        annual = xr.concat(
            annual_arrays,
            dim="year",
        )

        annual = annual.rename(
            "era5_t2m_djf_c"
        )

        annual.attrs.update(
            temperature.attrs
        )

        annual.attrs.update(
            {
                "units": (
                    "degree_Celsius"
                ),
                "description": (
                    "ERA5 annual DJF mean 2 m air temperature"
                ),
                "djf_definition": (
                    "DJF YYYY = Dec YYYY-1 + Jan YYYY + Feb YYYY"
                ),
                "day_weighted": str(
                    day_weighted
                ),
                "source_file": str(
                    monthly_path
                ),
            }
        )

        output_dataset = (
            annual.to_dataset()
        )

        output_dataset.attrs.update(
            {
                "source_file": str(
                    monthly_path
                ),
                "source_sha256": file_sha256(
                    monthly_path
                ),
                "temperature_units": (
                    "degree_Celsius"
                ),
                "djf_definition": (
                    "DJF YYYY = Dec YYYY-1 + Jan YYYY + Feb YYYY"
                ),
                "complete_djf_years": (
                    ",".join(
                        map(
                            str,
                            complete_years,
                        )
                    )
                ),
            }
        )

        save_netcdf(
            output_dataset,
            output_path,
            encoding={
                "era5_t2m_djf_c": {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                }
            },
        )

        if incomplete_rows:
            atomic_csv_write(
                pd.DataFrame(
                    incomplete_rows
                ),
                ERA5_DIRECTORY
                / "incomplete_djf_seasons.csv",
            )

        return output_path

    finally:
        dataset.close()


def open_era5_djf(
    path,
):
    """Open the annual ERA5 DJF dataset."""

    dataset = xr.open_dataset(
        path
    )

    rename = {}

    if "lon" in dataset.coords:
        rename[
            "lon"
        ] = "longitude"

    if "lat" in dataset.coords:
        rename[
            "lat"
        ] = "latitude"

    if "season_year" in dataset.coords:
        rename[
            "season_year"
        ] = "year"

    if rename:
        dataset = dataset.rename(
            rename
        )

    if (
        "era5_t2m_djf_c"
        not in dataset.data_vars
    ):
        raise KeyError(
            "Annual ERA5 file has no "
            "'era5_t2m_djf_c' variable."
        )

    dataset = normalize_longitudes(
        dataset
    )

    dataset = dataset.sortby(
        "latitude"
    )

    return (
        dataset,
        dataset[
            "era5_t2m_djf_c"
        ],
    )
# ERA5 grid

def regular_coordinate_bounds(
    values,
    *,
    periodic=False,
):
    """Calculate cell bounds from regular coordinate centers."""

    centers = np.asarray(
        values,
        dtype="float64",
    )

    centers = np.sort(
        centers
    )

    differences = np.diff(
        centers
    )

    differences = differences[
        np.isfinite(
            differences
        )
        & (
            differences
            > 0
        )
    ]

    if not differences.size:
        raise ValueError(
            "Could not infer coordinate spacing."
        )

    spacing = float(
        np.median(
            differences
        )
    )

    lower = (
        centers
        - spacing / 2.0
    )

    upper = (
        centers
        + spacing / 2.0
    )

    if periodic:
        lower = np.maximum(
            lower,
            -180.0,
        )

        upper = np.minimum(
            upper,
            180.0,
        )

    return (
        centers,
        lower,
        upper,
        spacing,
    )


def build_projected_era5_grid(
    temperature,
    *,
    latitude_maximum,
    output_crs,
):
    """Build projected ERA5 cell polygons with stable native-grid IDs."""

    source_longitude = np.asarray(
        temperature[
            "longitude"
        ].values,
        dtype="float64",
    )

    source_longitude = (
        (
            (
                source_longitude
                + 180.0
            )
            % 360.0
        )
        - 180.0
    )

    source_latitude = np.asarray(
        temperature[
            "latitude"
        ].values,
        dtype="float64",
    )

    (
        longitude_centers,
        longitude_lower,
        longitude_upper,
        _,
    ) = regular_coordinate_bounds(
        source_longitude,
        periodic=True,
    )

    (
        latitude_centers,
        latitude_lower,
        latitude_upper,
        _,
    ) = regular_coordinate_bounds(
        source_latitude,
        periodic=False,
    )

    latitude_lower = np.maximum(
        latitude_lower,
        -90.0,
    )

    latitude_upper = np.minimum(
        latitude_upper,
        90.0,
    )

    keep_latitude = (
        latitude_centers
        <= latitude_maximum
    )

    latitude_centers = latitude_centers[
        keep_latitude
    ]

    latitude_lower = latitude_lower[
        keep_latitude
    ]

    latitude_upper = latitude_upper[
        keep_latitude
    ]

    rows = []

    for latitude_position in tqdm(
        range(
            len(
                latitude_centers
            )
        ),
        desc="Building ERA5 grid",
    ):
        latitude = float(
            latitude_centers[
                latitude_position
            ]
        )

        south = float(
            latitude_lower[
                latitude_position
            ]
        )

        north = float(
            latitude_upper[
                latitude_position
            ]
        )

        source_lat_index = int(
            np.argmin(
                np.abs(
                    source_latitude
                    - latitude
                )
            )
        )

        for longitude_position in range(
            len(
                longitude_centers
            )
        ):
            longitude = float(
                longitude_centers[
                    longitude_position
                ]
            )

            west = float(
                longitude_lower[
                    longitude_position
                ]
            )

            east = float(
                longitude_upper[
                    longitude_position
                ]
            )

            if (
                east <= west
                or north <= south
            ):
                continue

            wrapped_difference = (
                (
                    (
                        source_longitude
                        - longitude
                        + 180.0
                    )
                    % 360.0
                )
                - 180.0
            )

            source_lon_index = int(
                np.argmin(
                    np.abs(
                        wrapped_difference
                    )
                )
            )

            cell_id = (
                source_lat_index
                * len(
                    source_longitude
                )
                + source_lon_index
            )

            rows.append(
                {
                    "era5_cell_id": int(
                        cell_id
                    ),
                    "era5_lat_index": (
                        source_lat_index
                    ),
                    "era5_lon_index": (
                        source_lon_index
                    ),
                    "longitude": (
                        longitude
                    ),
                    "latitude": (
                        latitude
                    ),
                    "geometry": box(
                        west,
                        south,
                        east,
                        north,
                    ),
                }
            )

    geographic = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326",
    )

    if geographic[
        "era5_cell_id"
    ].duplicated().any():
        raise RuntimeError(
            "ERA5 cell IDs are not unique."
        )

    return geographic.to_crs(
        output_crs
    )


def save_era5_grid(
    grid,
    path,
):
    """Save the projected ERA5 grid."""

    path = Path(path)

    if path.exists():
        path.unlink()

    grid.to_file(
        path,
        layer="era5_grid",
        driver="GPKG",
    )

    print(
        "[SAVED]",
        path,
        flush=True,
    )

# Shelf-specific ERA5 crops. Cropping to MEaSURES boundaries

def subset_grid_to_shelf(
    grid,
    shelf_geometry,
):
    """Select ERA5 cells with positive shelf-overlap area."""

    try:
        indices = list(
            grid.sindex.query(
                shelf_geometry,
                predicate="intersects",
            )
        )

        selected = grid.iloc[
            indices
        ].copy()

    except Exception:
        selected = grid[
            grid.intersects(
                shelf_geometry
            )
        ].copy()

    selected = selected[
        selected.geometry.notna()
        & ~selected.geometry.is_empty
    ].copy()

    selected[
        "shelf_overlap_area_m2"
    ] = (
        selected.geometry
        .intersection(
            shelf_geometry
        )
        .area
    )

    return selected[
        selected[
            "shelf_overlap_area_m2"
        ]
        > 0
    ].copy()


def shelf_crop_paths(
    shelf,
):
    """Return one shelf's ERA5 crop paths."""

    name = safe_filename(
        shelf
    )

    netcdf_path = (
        ERA5_SHELF_CROP_DIRECTORY
        / f"{name}_era5_djf.nc"
    )

    geometry_path = (
        ERA5_SHELF_CROP_DIRECTORY
        / f"{name}_era5_cells.gpkg"
    )

    return (
        netcdf_path,
        geometry_path,
    )


def extract_shelf_era5_dataset(
    temperature,
    selected_cells,
):
    """Extract ERA5 by stable source latitude/longitude indices."""

    selected_cells = (
        selected_cells[
            [
                "era5_cell_id",
                "era5_lat_index",
                "era5_lon_index",
                "longitude",
                "latitude",
            ]
        ]
        .drop_duplicates(
            "era5_cell_id"
        )
        .sort_values(
            "era5_cell_id"
        )
        .reset_index(
            drop=True
        )
    )

    latitude_index = xr.DataArray(
        selected_cells[
            "era5_lat_index"
        ].to_numpy(
            dtype=int
        ),
        dims="era5_cell",
    )

    longitude_index = xr.DataArray(
        selected_cells[
            "era5_lon_index"
        ].to_numpy(
            dtype=int
        ),
        dims="era5_cell",
    )

    values = temperature.isel(
        latitude=latitude_index,
        longitude=longitude_index,
    )

    values = values.transpose(
        "year",
        "era5_cell",
    )

    values = values.rename(
        "era5_t2m_djf_c"
    )

    output = values.to_dataset()

    output = output.assign_coords(
        era5_cell=np.arange(
            len(
                selected_cells
            ),
            dtype=int,
        )
    )

    for column in [
        "era5_cell_id",
        "era5_lat_index",
        "era5_lon_index",
        "longitude",
        "latitude",
    ]:
        output[
            column
        ] = xr.DataArray(
            selected_cells[
                column
            ].to_numpy(),
            dims="era5_cell",
        )

    output[
        "era5_t2m_djf_c"
    ].attrs.update(
        {
            "units": (
                "degree_Celsius"
            ),
            "description": (
                "ERA5 annual DJF mean 2 m air temperature"
            ),
            "djf_definition": (
                "DJF YYYY = Dec YYYY-1 + Jan YYYY + Feb YYYY"
            ),
        }
    )

    return output


def create_one_shelf_era5_crop(
    shelf,
    shelf_geometry,
    *,
    grid,
    temperature,
    overwrite,
):
    """Create one shelf's ERA5 temperature and geometry files."""

    (
        netcdf_path,
        geometry_path,
    ) = shelf_crop_paths(
        shelf
    )

    if (
        netcdf_path.exists()
        and geometry_path.exists()
        and not overwrite
    ):
        return {
            "shelf": shelf,
            "status": "exists",
            "n_cells": np.nan,
            "netcdf_path": str(
                netcdf_path
            ),
            "geometry_path": str(
                geometry_path
            ),
            "error": "",
        }

    try:
        selected = subset_grid_to_shelf(
            grid,
            shelf_geometry,
        )

        if selected.empty:
            raise RuntimeError(
                "No positive-area ERA5 cell overlaps."
            )

        shelf_dataset = (
            extract_shelf_era5_dataset(
                temperature,
                selected,
            )
        )

        shelf_dataset.attrs.update(
            {
                "shelf": str(
                    shelf
                ),
                "source_era5_file": str(
                    ERA5_DJF_PATH
                ),
                "crop_type": (
                    "native ERA5 cells with positive-area shelf overlap"
                ),
                "selection_rule": (
                    "ERA5 cell intersection with shelf polygon has positive area"
                ),
                "djf_definition": (
                    "DJF YYYY = Dec YYYY-1 + Jan YYYY + Feb YYYY"
                ),
            }
        )

        save_netcdf(
            shelf_dataset,
            netcdf_path,
            encoding={
                "era5_t2m_djf_c": {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                }
            },
        )

        if geometry_path.exists():
            geometry_path.unlink()

        selected.to_file(
            geometry_path,
            layer="era5_shelf_cells",
            driver="GPKG",
        )

        return {
            "shelf": shelf,
            "status": "OK",
            "n_cells": int(
                len(
                    selected
                )
            ),
            "netcdf_path": str(
                netcdf_path
            ),
            "geometry_path": str(
                geometry_path
            ),
            "error": "",
        }

    except Exception as error:
        return {
            "shelf": shelf,
            "status": "ERROR",
            "n_cells": 0,
            "netcdf_path": str(
                netcdf_path
            ),
            "geometry_path": str(
                geometry_path
            ),
            "error": repr(
                error
            ),
        }


def create_all_shelf_era5_crops(
    shelves,
    shelf_geometries,
    *,
    grid,
    temperature,
    overwrite,
):
    """Create shelf-specific ERA5 files."""

    rows = []

    for shelf in tqdm(
        shelves,
        desc="Saving ERA5 shelf crops",
    ):
        if shelf not in shelf_geometries:
            rows.append(
                {
                    "shelf": shelf,
                    "status": (
                        "geometry_missing"
                    ),
                    "n_cells": 0,
                    "netcdf_path": "",
                    "geometry_path": "",
                    "error": (
                        "Shelf geometry was not loaded."
                    ),
                }
            )

            continue

        rows.append(
            create_one_shelf_era5_crop(
                shelf,
                shelf_geometries[
                    shelf
                ],
                grid=grid,
                temperature=temperature,
                overwrite=overwrite,
            )
        )

    results = pd.DataFrame(
        rows
    )

    atomic_csv_write(
        results,
        ERA5_DIRECTORY
        / "shelf_crop_results.csv",
    )

    return results

# Melt-raster function helpers

def find_melt_raster(
    shelf,
    year,
):
    """Find one shelf/year corrected melt raster."""

    shelf_names = [
        str(
            shelf
        ),
        safe_filename(
            shelf
        ),
        str(
            shelf
        ).replace(
            " ",
            "_",
        ),
    ]

    for shelf_name in shelf_names:
        directory = (
            MELTWATER_RASTER_DIR
            / shelf_name
        )

        for pattern in (
            RASTER_FILENAME_PATTERNS
        ):
            filename = pattern.format(
                year=int(
                    year
                ),
                next_year=(
                    int(
                        year
                    )
                    + 1
                ),
            )

            path = (
                directory
                / filename
            )

            if path.is_file():
                return path

    return None


def classify_raster_pixels(
    data,
    *,
    nodata=None,
):
    """
    Classify corrected ponding raster pixels.

    Returns
    -------
    array_float

    original_valid
        True where the source raster contained a valid value. 

    ponded
        True where the original valid raster value equals 1.
    """

    if np.ma.isMaskedArray(
        data
    ):
        raw = np.asarray(
            data.data
        )

        mask = np.ma.getmaskarray(
            data
        ).copy()

    else:
        raw = np.asarray(
            data
        )

        mask = np.zeros(
            raw.shape,
            dtype=bool,
        )

    if nodata is not None:
        try:
            if np.isfinite(
                nodata
            ):
                mask |= (
                    raw
                    == nodata
                )

        except Exception:
            pass

    if np.issubdtype(
        raw.dtype,
        np.floating,
    ):
        mask |= (
            ~np.isfinite(
                raw
            )
        )

    original_valid = (
        ~mask
    )

    array_float = raw.astype(
        "float64",
        copy=True,
    )

    array_float[
        mask
    ] = np.nan

    ponded = (
        original_valid
        & (
            raw
            == 1
        )
    )

    return (
        array_float,
        original_valid,
        ponded,
    )


def reproject_gdf(
    geodataframe,
    target_crs,
):
    """Reproject a GeoDataFrame with strict CRS validation."""

    if target_crs is None:
        raise ValueError(
            "Target raster CRS is missing."
        )

    if geodataframe.crs is None:
        raise ValueError(
            "Input GeoDataFrame CRS is missing."
        )

    if crs_equal(
        geodataframe.crs,
        target_crs,
    ):
        return geodataframe.copy()

    return geodataframe.to_crs(
        target_crs
    )

# common-support landsat melt aggregation

def aggregate_one_shelf_year(
    shelf,
    year,
    *,
    shelf_clipped_cells,
    raster_path,
    cell_id_column="era5_cell_id",
    all_touched=False,
):
    """
    Aggregate one shelf-year raster using the original denominator convention.

    Geometry:
        ERA5 cell ∩ shelf polygon ∩ raster rectangular footprint

    Denominator:
        Every raster-grid pixel whose rasterized cell label is greater than
        zero

    Numerator:
        Rasterized common-support pixels whose source raster value equals 1.
    """

    with rasterio.open(
        raster_path
    ) as source:
        raster_data = source.read(
            1,
            masked=True,
        )

        (
            _,
            original_valid,
            ponded,
        ) = classify_raster_pixels(
            raster_data,
            nodata=source.nodata,
        )

        cells = reproject_gdf(
            shelf_clipped_cells,
            source.crs,
        )

        raster_footprint = box(
            *source.bounds
        )

        cells = cells[
            cells.intersects(
                raster_footprint
            )
        ].copy()

        if cells.empty:
            return pd.DataFrame()

        cells[
            "geometry"
        ] = (
            cells.geometry.intersection(
                raster_footprint
            )
        )

        cells = cells[
            cells.geometry.notna()
            & ~cells.geometry.is_empty
        ].copy()

        cells[
            "effective_overlap_area_m2"
        ] = cells.geometry.area

        cells = cells[
            cells[
                "effective_overlap_area_m2"
            ]
            > 0
        ].copy()

        if cells.empty:
            return pd.DataFrame()

        cells[
            "coverage_fraction_of_shelf_cell"
        ] = (
            cells[
                "effective_overlap_area_m2"
            ]
            / cells[
                "shelf_cell_overlap_area_m2"
            ]
        )

        cells = cells.reset_index(
            drop=True
        )

        shapes = []
        label_lookup = {}

        for label, row in enumerate(
            cells.itertuples(
                index=False
            ),
            start=1,
        ):
            shapes.append(
                (
                    row.geometry,
                    label,
                )
            )

            label_lookup[
                label
            ] = {
                "era5_cell_id": int(
                    getattr(
                        row,
                        cell_id_column,
                    )
                ),
                "shelf_cell_overlap_area_m2": float(
                    row.shelf_cell_overlap_area_m2
                ),
                "effective_overlap_area_m2": float(
                    row.effective_overlap_area_m2
                ),
                "coverage_fraction_of_shelf_cell": float(
                    row.coverage_fraction_of_shelf_cell
                ),
            }

        label_raster = rasterize(
            shapes,
            out_shape=source.shape,
            transform=source.transform,
            fill=0,
            dtype="int32",
            all_touched=all_touched,
        )

        inside_common_support = (
            label_raster
            > 0
        )

        labels_inside = label_raster[
            inside_common_support
        ]

        labels_ponded = label_raster[
            inside_common_support
            & ponded
        ]

        valid_inside_count = int(
            np.sum(
                inside_common_support
                & original_valid
            )
        )

        total_inside_count = int(
            np.sum(
                inside_common_support
            )
        )

        if labels_inside.size == 0:
            return pd.DataFrame()

        maximum_label = int(
            label_raster.max()
        )

        pixel_counts = np.bincount(
            labels_inside,
            minlength=maximum_label + 1,
        )

        ponded_counts = np.bincount(
            labels_ponded,
            minlength=maximum_label + 1,
        )

        valid_source_counts = np.bincount(
            label_raster[
                inside_common_support
                & original_valid
            ],
            minlength=maximum_label + 1,
        )

        rows = []

        for label in range(
            1,
            maximum_label + 1,
        ):
            number_inside = int(
                pixel_counts[
                    label
                ]
            )

            if number_inside <= 0:
                continue

            number_ponded = int(
                ponded_counts[
                    label
                ]
            )

            number_source_valid = int(
                valid_source_counts[
                    label
                ]
            )

            metadata = label_lookup[
                label
            ]

            rows.append(
                {
                    "shelf": shelf,
                    "year": int(
                        year
                    ),
                    "era5_cell_id": (
                        metadata[
                            "era5_cell_id"
                        ]
                    ),
                    "shelf_cell_overlap_area_m2": (
                        metadata[
                            "shelf_cell_overlap_area_m2"
                        ]
                    ),
                    "effective_overlap_area_m2": (
                        metadata[
                            "effective_overlap_area_m2"
                        ]
                    ),
                    "coverage_fraction_of_shelf_cell": (
                        metadata[
                            "coverage_fraction_of_shelf_cell"
                        ]
                    ),
                    "n_pixels_30m": (
                        number_inside
                    ),
                    "y_ponded_pixels": (
                        number_ponded
                    ),
                    "ponding_frac": (
                        number_ponded
                        / number_inside
                    ),
                    "n_source_valid_pixels_qa": (
                        number_source_valid
                    ),
                    "n_source_masked_or_nodata_pixels_qa": (
                        number_inside
                        - number_source_valid
                    ),
                    "source_valid_fraction_inside_geometry_qa": (
                        number_source_valid
                        / number_inside
                    ),
                    "n_rasters": 1,
                    "raster_path": str(
                        raster_path
                    ),
                    "raster_crs": str(
                        source.crs
                    ),
                    "raster_nodata": (
                        source.nodata
                    ),
                    "all_touched": bool(
                        all_touched
                    ),
                    "denominator_convention": (
                        "all raster-grid pixels inside rasterized "
                        "ERA5-cell/shelf/raster-footprint geometry; "
                        "masked and nodata source pixels count as nonponded"
                    ),
                }
            )

        print(
            f"[AGGREGATE] {shelf} {year}: "
            f"inside={total_inside_count:,}; "
            f"source_valid={valid_inside_count:,}; "
            f"source_masked_or_nodata="
            f"{total_inside_count - valid_inside_count:,}",
            flush=True,
        )

        return pd.DataFrame(
            rows
        )


def read_shelf_era5_long_table(
    shelf,
    *,
    years,
):
    """Read one shelf's ERA5 crop as a year-cell table."""

    netcdf_path, _ = shelf_crop_paths(
        shelf
    )

    if not netcdf_path.is_file():
        raise FileNotFoundError(
            f"Missing shelf ERA5 crop:\n"
            f"{netcdf_path}"
        )

    dataset = xr.open_dataset(
        netcdf_path
    )

    try:
        available_years = set(
            dataset[
                "year"
            ].values.astype(int)
        )

        requested_years = set(
            int(
                year
            )
            for year
            in years
        )

        missing_years = (
            requested_years
            - available_years
        )

        if missing_years:
            raise RuntimeError(
                f"{shelf} ERA5 crop is missing years: "
                f"{sorted(missing_years)}"
            )

        temperature = (
            dataset[
                "era5_t2m_djf_c"
            ]
            .sel(
                year=sorted(
                    requested_years
                )
            )
        )

        table = (
            temperature.to_dataframe(
                name="era5_t2m_djf_c"
            )
            .reset_index()
        )

        cell_lookup = pd.DataFrame(
            {
                "era5_cell": (
                    dataset[
                        "era5_cell"
                    ].values
                ),
                "era5_cell_id": (
                    dataset[
                        "era5_cell_id"
                    ]
                    .values
                    .astype(int)
                ),
                "era5_lat_index": (
                    dataset[
                        "era5_lat_index"
                    ]
                    .values
                    .astype(int)
                ),
                "era5_lon_index": (
                    dataset[
                        "era5_lon_index"
                    ]
                    .values
                    .astype(int)
                ),
                "longitude": (
                    dataset[
                        "longitude"
                    ]
                    .values
                    .astype(float)
                ),
                "latitude": (
                    dataset[
                        "latitude"
                    ]
                    .values
                    .astype(float)
                ),
            }
        )

        table = table.merge(
            cell_lookup,
            on="era5_cell",
            how="left",
            validate="many_to_one",
        )

        table[
            "year"
        ] = table[
            "year"
        ].astype(int)

        return table[
            [
                "year",
                "era5_cell_id",
                "era5_lat_index",
                "era5_lon_index",
                "longitude",
                "latitude",
                "era5_t2m_djf_c",
            ]
        ].copy()

    finally:
        dataset.close()


def process_one_shelf(
    shelf,
    shelf_geometry,
    *,
    grid,
    year_min,
    year_max,
    all_touched,
):
    """Create one shelf's common-support observation table."""

    selected_cells = subset_grid_to_shelf(
        grid,
        shelf_geometry,
    )

    if selected_cells.empty:
        raise RuntimeError(
            "No positive-area ERA5 cell overlaps."
        )

    shelf_clipped = (
        selected_cells.copy()
    )

    shelf_clipped[
        "geometry"
    ] = (
        shelf_clipped.geometry.intersection(
            shelf_geometry
        )
    )

    shelf_clipped = shelf_clipped[
        shelf_clipped.geometry.notna()
        & ~shelf_clipped.geometry.is_empty
    ].copy()

    shelf_clipped[
        "shelf_cell_overlap_area_m2"
    ] = (
        shelf_clipped.geometry.area
    )

    shelf_clipped = shelf_clipped[
        shelf_clipped[
            "shelf_cell_overlap_area_m2"
        ]
        > 0
    ].copy()

    observation_parts = []
    missing_rasters = []

    for year in range(
        int(
            year_min
        ),
        int(
            year_max
        )
        + 1,
    ):
        raster_path = find_melt_raster(
            shelf,
            year,
        )

        if raster_path is None:
            missing_rasters.append(
                year
            )

            continue

        part = aggregate_one_shelf_year(
            shelf,
            year,
            shelf_clipped_cells=(
                shelf_clipped
            ),
            raster_path=(
                raster_path
            ),
            cell_id_column=(
                "era5_cell_id"
            ),
            all_touched=(
                all_touched
            ),
        )

        if not part.empty:
            observation_parts.append(
                part
            )

    if missing_rasters:
        raise RuntimeError(
            f"Missing melt rasters for years: "
            f"{missing_rasters}"
        )

    if not observation_parts:
        raise RuntimeError(
            "No common-support observation rows were created."
        )

    observations = pd.concat(
        observation_parts,
        ignore_index=True,
    )

    duplicate = observations.duplicated(
        [
            "shelf",
            "year",
            "era5_cell_id",
        ],
        keep=False,
    )

    if duplicate.any():
        raise RuntimeError(
            "Duplicate shelf-year-cell observation rows were created."
        )

    years = list(
        range(
            int(
                year_min
            ),
            int(
                year_max
            )
            + 1,
        )
    )

    temperature = read_shelf_era5_long_table(
        shelf,
        years=years,
    )

    output = observations.merge(
        temperature,
        on=[
            "year",
            "era5_cell_id",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_temperature = (
        output[
            "era5_t2m_djf_c"
        ].isna()
    )

    if missing_temperature.any():
        examples = output.loc[
            missing_temperature,
            [
                "shelf",
                "year",
                "era5_cell_id",
            ],
        ].head(
            20
        )

        raise RuntimeError(
            "Missing ERA5 temperature after merge. "
            f"Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    output[
        "region"
    ] = get_region_for_shelf(
        shelf
    )

    output[
        "quadrant"
    ] = get_quadrant_for_shelf(
        shelf
    )

    output[
        "n_pixels_30m"
    ] = output[
        "n_pixels_30m"
    ].astype(
        "int64"
    )

    output[
        "y_ponded_pixels"
    ] = output[
        "y_ponded_pixels"
    ].astype(
        "int64"
    )

    preferred_columns = [
        "shelf",
        "region",
        "quadrant",
        "year",
        "era5_cell_id",
        "era5_lat_index",
        "era5_lon_index",
        "longitude",
        "latitude",
        "shelf_cell_overlap_area_m2",
        "effective_overlap_area_m2",
        "coverage_fraction_of_shelf_cell",
        "n_pixels_30m",
        "y_ponded_pixels",
        "ponding_frac",
        "n_source_valid_pixels_qa",
        "n_source_masked_or_nodata_pixels_qa",
        "source_valid_fraction_inside_geometry_qa",
        "era5_t2m_djf_c",
        "n_rasters",
        "raster_path",
        "raster_crs",
        "raster_nodata",
        "all_touched",
        "denominator_convention",
    ]

    output = output[
        [
            column
            for column
            in preferred_columns
            if column in output.columns
        ]
    ].sort_values(
        [
            "shelf",
            "year",
            "era5_cell_id",
        ]
    ).reset_index(
        drop=True
    )

    return output
    
# Parallel shelf processing

_WORKER_GRID = None
_WORKER_SHELF_GEOMETRIES = None


def initialize_worker(
    grid,
    shelf_geometries,
):
    """Initialize process-level read-only objects."""

    global _WORKER_GRID
    global _WORKER_SHELF_GEOMETRIES

    _WORKER_GRID = grid

    _WORKER_SHELF_GEOMETRIES = (
        shelf_geometries
    )

    pngu.CROPPED_ANTWIDE_ROOT = str(
        MELTWATER_RASTER_DIR
    )

    pngu.RASTER_GLOB = (
        "ANT_WIDE_*_*_30m_raster.tif"
    )


def shelf_observation_paths(
    shelf,
):
    """Return per-shelf output paths."""

    name = safe_filename(
        shelf
    )

    parquet_path = (
        OBSERVATION_BY_SHELF_DIRECTORY
        / (
            f"{name}_common_support_"
            f"{OBS_YEAR_MIN}_{OBS_YEAR_MAX}.parquet"
        )
    )

    csv_path = (
        parquet_path.with_suffix(
            ".csv"
        )
    )

    return (
        parquet_path,
        csv_path,
    )


def process_shelf_worker(
    shelf,
):
    """Process and save one shelf."""

    (
        parquet_path,
        csv_path,
    ) = shelf_observation_paths(
        shelf
    )

    if (
        parquet_path.exists()
        and not OVERWRITE
    ):
        return {
            "shelf": shelf,
            "status": "exists",
            "n_rows": np.nan,
            "parquet_path": str(
                parquet_path
            ),
            "csv_path": str(
                csv_path
            ),
            "error": "",
        }

    try:
        if (
            shelf
            not in _WORKER_SHELF_GEOMETRIES
        ):
            raise KeyError(
                "Shelf geometry was not loaded."
            )

        table = process_one_shelf(
            shelf,
            _WORKER_SHELF_GEOMETRIES[
                shelf
            ],
            grid=_WORKER_GRID,
            year_min=(
                OBS_YEAR_MIN
            ),
            year_max=(
                OBS_YEAR_MAX
            ),
            all_touched=(
                ALL_TOUCHED
            ),
        )

        table.to_parquet(
            parquet_path,
            index=False,
        )

        table.to_csv(
            csv_path,
            index=False,
        )

        return {
            "shelf": shelf,
            "status": "OK",
            "n_rows": int(
                len(
                    table
                )
            ),
            "parquet_path": str(
                parquet_path
            ),
            "csv_path": str(
                csv_path
            ),
            "error": "",
        }

    except Exception as error:
        return {
            "shelf": shelf,
            "status": "ERROR",
            "n_rows": 0,
            "parquet_path": str(
                parquet_path
            ),
            "csv_path": str(
                csv_path
            ),
            "error": repr(
                error
            ),
        }


def build_all_observations(
    shelves,
    shelf_geometries,
    grid,
):
    """Build per-shelf tables and combine successful results."""

    if N_WORKERS <= 1:
        initialize_worker(
            grid,
            shelf_geometries,
        )

        results = [
            process_shelf_worker(
                shelf
            )
            for shelf
            in tqdm(
                shelves,
                desc="Processing shelves",
            )
        ]

    else:
        context_name = (
            "fork"
            if "fork"
            in mp.get_all_start_methods()
            else "spawn"
        )

        context = mp.get_context(
            context_name
        )

        results = []

        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(
                grid,
                shelf_geometries,
            ),
        ) as executor:
            future_lookup = {
                executor.submit(
                    process_shelf_worker,
                    shelf,
                ): shelf
                for shelf
                in shelves
            }

            for future in tqdm(
                as_completed(
                    future_lookup
                ),
                total=len(
                    future_lookup
                ),
                desc="Processing shelves",
            ):
                shelf = future_lookup[
                    future
                ]

                try:
                    result = future.result()

                except Exception as error:
                    result = {
                        "shelf": shelf,
                        "status": (
                            "FUTURE_ERROR"
                        ),
                        "n_rows": 0,
                        "parquet_path": "",
                        "csv_path": "",
                        "error": repr(
                            error
                        ),
                    }

                results.append(
                    result
                )

                print(
                    f"[{result['status']}] "
                    f"{shelf}: "
                    f"rows={result.get('n_rows')}; "
                    f"{result.get('error', '')}",
                    flush=True,
                )

    results_table = pd.DataFrame(
        results
    ).sort_values(
        "shelf"
    ).reset_index(
        drop=True
    )

    atomic_csv_write(
        results_table,
        PROCESSING_RESULTS_PATH,
    )

    failures = results_table[
        ~results_table[
            "status"
        ].isin(
            [
                "OK",
                "exists",
            ]
        )
    ].copy()

    atomic_csv_write(
        failures,
        FAILED_SHELVES_PATH,
    )

    successful_paths = results_table.loc[
        results_table[
            "status"
        ].isin(
            [
                "OK",
                "exists",
            ]
        ),
        "parquet_path",
    ].tolist()

    parts = []

    for path in successful_paths:
        path = Path(
            path
        )

        if path.is_file():
            parts.append(
                pd.read_parquet(
                    path
                )
            )

    if not parts:
        raise RuntimeError(
            "No successful shelf observation tables were found."
        )

    combined = pd.concat(
        parts,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "shelf",
            "year",
            "era5_cell_id",
        ]
    ).reset_index(
        drop=True
    )

    duplicate = combined.duplicated(
        [
            "shelf",
            "year",
            "era5_cell_id",
        ],
        keep=False,
    )

    if duplicate.any():
        raise RuntimeError(
            "Combined table contains duplicate shelf-year-cell rows."
        )

    combined.to_parquet(
        COMBINED_OBSERVATION_PARQUET,
        index=False,
    )

    combined.to_csv(
        COMBINED_OBSERVATION_CSV,
        index=False,
    )

    print(
        "[SAVED]",
        COMBINED_OBSERVATION_PARQUET,
        flush=True,
    )

    print(
        "[SAVED]",
        COMBINED_OBSERVATION_CSV,
        flush=True,
    )

    return (
        combined,
        failures,
        results_table,
    )

# Summary tables and quality checking data

def summarize_coverage(
    observations,
):
    """Summarize geometric and source-valid support by shelf and year."""

    summary = (
        observations.groupby(
            [
                "shelf",
                "year",
            ],
            observed=True,
        )
        .agg(
            n_era5_cells=(
                "era5_cell_id",
                "nunique",
            ),
            selected_shelf_cell_area_m2=(
                "shelf_cell_overlap_area_m2",
                "sum",
            ),
            effective_common_support_area_m2=(
                "effective_overlap_area_m2",
                "sum",
            ),
            n_pixels_30m=(
                "n_pixels_30m",
                "sum",
            ),
            y_ponded_pixels=(
                "y_ponded_pixels",
                "sum",
            ),
            n_source_valid_pixels_qa=(
                "n_source_valid_pixels_qa",
                "sum",
            ),
            n_source_masked_or_nodata_pixels_qa=(
                "n_source_masked_or_nodata_pixels_qa",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "geometric_common_support_fraction"
    ] = (
        summary[
            "effective_common_support_area_m2"
        ]
        / summary[
            "selected_shelf_cell_area_m2"
        ]
    )

    summary[
        "source_valid_fraction_inside_geometry_qa"
    ] = (
        summary[
            "n_source_valid_pixels_qa"
        ]
        / summary[
            "n_pixels_30m"
        ]
    )

    summary[
        "ponding_frac_common_support"
    ] = (
        summary[
            "y_ponded_pixels"
        ]
        / summary[
            "n_pixels_30m"
        ]
    )

    return summary


def summarize_shelf_metadata(
    observations,
):
    """Create the source table for the shelf metadata supplement."""

    source = observations.copy()

    source[
        "temperature_pixel_weighted_numerator"
    ] = (
        source[
            "era5_t2m_djf_c"
        ]
        * source[
            "n_pixels_30m"
        ]
    )

    summary = (
        source.groupby(
            [
                "shelf",
                "region",
            ],
            observed=True,
        )
        .agg(
            n_years=(
                "year",
                "nunique",
            ),
            year_minimum=(
                "year",
                "min",
            ),
            year_maximum=(
                "year",
                "max",
            ),
            n_era5_cells=(
                "era5_cell_id",
                "nunique",
            ),
            n_rows=(
                "era5_cell_id",
                "size",
            ),
            total_n_pixels_30m=(
                "n_pixels_30m",
                "sum",
            ),
            total_y_ponded_pixels=(
                "y_ponded_pixels",
                "sum",
            ),
            era5_temperature_mean_c=(
                "era5_t2m_djf_c",
                "mean",
            ),
            temperature_pixel_weighted_numerator=(
                "temperature_pixel_weighted_numerator",
                "sum",
            ),
            coverage_fraction_mean=(
                "coverage_fraction_of_shelf_cell",
                "mean",
            ),
            coverage_fraction_minimum=(
                "coverage_fraction_of_shelf_cell",
                "min",
            ),
            coverage_fraction_maximum=(
                "coverage_fraction_of_shelf_cell",
                "max",
            ),
            total_effective_overlap_area_m2=(
                "effective_overlap_area_m2",
                "sum",
            ),
            total_source_valid_pixels_qa=(
                "n_source_valid_pixels_qa",
                "sum",
            ),
            total_source_masked_or_nodata_pixels_qa=(
                "n_source_masked_or_nodata_pixels_qa",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "observed_ponded_percent"
    ] = (
        100.0
        * summary[
            "total_y_ponded_pixels"
        ]
        / summary[
            "total_n_pixels_30m"
        ]
    )

    summary[
        "era5_temperature_pixel_weighted_mean_c"
    ] = (
        summary[
            "temperature_pixel_weighted_numerator"
        ]
        / summary[
            "total_n_pixels_30m"
        ]
    )

    summary[
        "total_effective_overlap_area_km2"
    ] = (
        summary[
            "total_effective_overlap_area_m2"
        ]
        / 1.0e6
    )

    summary[
        "source_valid_fraction_inside_geometry_qa"
    ] = (
        summary[
            "total_source_valid_pixels_qa"
        ]
        / summary[
            "total_n_pixels_30m"
        ]
    )

    return summary.drop(
        columns=[
            "temperature_pixel_weighted_numerator",
        ]
    ).sort_values(
        "shelf"
    ).reset_index(
        drop=True
    )


def validate_final_observations(
    observations,
    shelves,
):
    """Apply strict final-dataset assertions."""

    required = [
        "shelf",
        "year",
        "era5_cell_id",
        "longitude",
        "latitude",
        "n_pixels_30m",
        "y_ponded_pixels",
        "ponding_frac",
        "era5_t2m_djf_c",
        "effective_overlap_area_m2",
        "coverage_fraction_of_shelf_cell",
        "denominator_convention",
    ]

    missing = [
        column
        for column in required
        if column not in observations.columns
    ]

    if missing:
        raise RuntimeError(
            f"Final observation table is missing: {missing}"
        )

    expected_years = set(
        range(
            OBS_YEAR_MIN,
            OBS_YEAR_MAX + 1,
        )
    )

    found_years = set(
        observations[
            "year"
        ].astype(
            int
        )
    )

    if found_years != expected_years:
        raise RuntimeError(
            f"Expected years {sorted(expected_years)}; "
            f"found {sorted(found_years)}."
        )

    found_shelves = set(
        observations[
            "shelf"
        ].astype(
            str
        )
    )

    missing_shelves = (
        set(
            shelves
        )
        - found_shelves
    )

    if missing_shelves:
        raise RuntimeError(
            f"Final table is missing shelves: "
            f"{sorted(missing_shelves)}"
        )

    conditions = {
        "nonpositive n_pixels_30m": (
            observations[
                "n_pixels_30m"
            ]
            <= 0
        ),
        "negative y_ponded_pixels": (
            observations[
                "y_ponded_pixels"
            ]
            < 0
        ),
        "y_ponded_pixels exceeds n_pixels_30m": (
            observations[
                "y_ponded_pixels"
            ]
            > observations[
                "n_pixels_30m"
            ]
        ),
        "ponding_frac outside zero to one": (
            (
                observations[
                    "ponding_frac"
                ]
                < 0
            )
            | (
                observations[
                    "ponding_frac"
                ]
                > 1
            )
        ),
        "missing or nonfinite temperature": (
            ~np.isfinite(
                observations[
                    "era5_t2m_djf_c"
                ]
            )
        ),
        "nonpositive effective overlap area": (
            observations[
                "effective_overlap_area_m2"
            ]
            <= 0
        ),
    }

    failures = {
        name: int(
            mask.sum()
        )
        for name, mask
        in conditions.items()
        if mask.any()
    }

    if failures:
        raise RuntimeError(
            f"Final dataset QA failed: {failures}"
        )

    duplicate = observations.duplicated(
        [
            "shelf",
            "year",
            "era5_cell_id",
        ],
        keep=False,
    )

    if duplicate.any():
        raise RuntimeError(
            "Duplicate shelf-year-cell rows exist."
        )

# four-shelf diagnostic plot. cropped to common support

def expand_bounds(
    bounds,
    fraction=0.07,
):
    """Expand map bounds."""

    minimum_x, minimum_y, maximum_x, maximum_y = (
        bounds
    )

    width = max(
        maximum_x
        - minimum_x,
        1.0,
    )

    height = max(
        maximum_y
        - minimum_y,
        1.0,
    )

    return (
        minimum_x
        - fraction
        * width,
        minimum_y
        - fraction
        * height,
        maximum_x
        + fraction
        * width,
        maximum_y
        + fraction
        * height,
    )


def create_partial_coverage_figure(
    shelves,
    shelf_geometries,
    grid,
    *,
    year,
):
    """Create the Cook/Fimbul/Mertz/Shackleton support figure."""

    selected_shelves = [
        shelf
        for shelf
        in [
            "Cook",
            "Fimbul",
            "Mertz",
            "Shackleton",
        ]
        if (
            shelf in shelves
            and shelf
            in shelf_geometries
        )
    ]

    if not selected_shelves:
        return

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            13,
            10,
        ),
        constrained_layout=True,
    )

    axes = axes.ravel()

    rows = []

    for axis, shelf in zip(
        axes,
        selected_shelves,
    ):
        shelf_geometry = (
            shelf_geometries[
                shelf
            ]
        )

        raster_path = find_melt_raster(
            shelf,
            year,
        )

        if raster_path is None:
            axis.set_title(
                f"{shelf}: raster missing"
            )

            axis.set_axis_off()

            continue

        selected = subset_grid_to_shelf(
            grid,
            shelf_geometry,
        )

        clipped = selected.copy()

        clipped[
            "geometry"
        ] = (
            clipped.geometry.intersection(
                shelf_geometry
            )
        )

        clipped = clipped[
            clipped.geometry.notna()
            & ~clipped.geometry.is_empty
        ].copy()

        with rasterio.open(
            raster_path
        ) as source:
            footprint = gpd.GeoDataFrame(
                {
                    "id": [
                        1
                    ]
                },
                geometry=[
                    box(
                        *source.bounds
                    )
                ],
                crs=source.crs,
            )

        if not crs_equal(
            footprint.crs,
            grid.crs,
        ):
            footprint = footprint.to_crs(
                grid.crs
            )

        footprint_geometry = (
            footprint.geometry.iloc[
                0
            ]
        )

        common = clipped.copy()

        common[
            "geometry"
        ] = (
            common.geometry.intersection(
                footprint_geometry
            )
        )

        common = common[
            common.geometry.notna()
            & ~common.geometry.is_empty
        ].copy()

        common_area = float(
            common.geometry.area.sum()
        )

        shelf_area = float(
            shelf_geometry.area
        )

        coverage = (
            common_area
            / shelf_area
            if shelf_area > 0
            else np.nan
        )

        common.plot(
            ax=axis,
            color="#FDD0A2",
            edgecolor="#E6550D",
            linewidth=0.4,
            alpha=0.65,
        )

        clipped.boundary.plot(
            ax=axis,
            color="#6BAED6",
            linewidth=0.45,
        )

        footprint.boundary.plot(
            ax=axis,
            color="#CC79A7",
            linewidth=1.0,
            linestyle="--",
        )

        gpd.GeoSeries(
            [
                shelf_geometry
            ],
            crs=grid.crs,
        ).boundary.plot(
            ax=axis,
            color="black",
            linewidth=1.3,
        )

        bounds = expand_bounds(
            unary_union(
                [
                    shelf_geometry,
                    footprint_geometry,
                ]
            ).bounds
        )

        axis.set_xlim(
            bounds[
                0
            ],
            bounds[
                2
            ],
        )

        axis.set_ylim(
            bounds[
                1
            ],
            bounds[
                3
            ],
        )

        axis.set_aspect(
            "equal"
        )

        axis.set_title(
            shelf
        )

        axis.text(
            0.02,
            0.02,
            (
                f"ERA5 cells: {len(selected):,}\n"
                f"Geometric common support / shelf: "
                f"{100.0 * coverage:.1f}%"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={
                "facecolor": (
                    "white"
                ),
                "edgecolor": (
                    "0.7"
                ),
                "alpha": 0.9,
            },
        )

        rows.append(
            {
                "shelf": shelf,
                "diagnostic_year": (
                    year
                ),
                "n_era5_cells": int(
                    len(
                        selected
                    )
                ),
                "shelf_area_m2": (
                    shelf_area
                ),
                "common_support_area_m2": (
                    common_area
                ),
                "common_support_fraction_of_shelf": (
                    coverage
                ),
                "raster_path": str(
                    raster_path
                ),
            }
        )

    for axis in axes[
        len(
            selected_shelves
        ):
    ]:
        axis.set_axis_off()

    figure.legend(
        handles=[
            Patch(
                facecolor="#FDD0A2",
                edgecolor="#E6550D",
                label="Geometric common support",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color="#6BAED6",
                label="ERA5 cells clipped to shelf",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color="#CC79A7",
                linestyle="--",
                label="Melt-raster footprint",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color="black",
                label="Shelf outline",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    figure.suptitle(
        "Common spatial support for partially covered ice shelves",
        fontsize=14,
    )

    output_basename = (
        DIAGNOSTIC_DIRECTORY
        / (
            "partial_coverage_shelves_"
            f"{year}"
        )
    )

    figure.savefig(
        Path(
            str(
                output_basename
            )
            + ".png"
        ),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        Path(
            str(
                output_basename
            )
            + ".pdf"
        ),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        figure
    )

    atomic_csv_write(
        pd.DataFrame(
            rows
        ),
        DIAGNOSTIC_DIRECTORY
        / (
            "partial_coverage_shelves_"
            f"{year}.csv"
        ),
    )

# Software metadata

def software_versions():
    """Return important software versions."""

    output = {
        "python": (
            platform.python_version()
        ),
        "platform": (
            platform.platform()
        ),
        "numpy": (
            np.__version__
        ),
        "pandas": (
            pd.__version__
        ),
        "geopandas": (
            gpd.__version__
        ),
        "xarray": (
            xr.__version__
        ),
        "rasterio": (
            rasterio.__version__
        ),
    }

    try:
        import shapely

        output[
            "shapely"
        ] = shapely.__version__

    except Exception:
        pass

    try:
        import pyproj

        output[
            "pyproj"
        ] = pyproj.__version__

    except Exception:
        pass

    return output


# Main Functions

def main():
    """Run ERA5 and melt-data preparation."""

    start_time = time.time()

    print("\n" + "=" * 100)
    print(
        "ERA5 AND MELT COMMON-SUPPORT PREPARATION"
    )
    print("=" * 100)

    print(
        "ERA5 monthly:",
        ERA5_MONTHLY_PATH,
        flush=True,
    )

    print(
        "Melt rasters:",
        MELTWATER_RASTER_DIR,
        flush=True,
    )

    print(
        "Output:",
        OUTPUT_DIR.resolve(),
        flush=True,
    )

    print(
        "Observation period:",
        f"{OBS_YEAR_MIN}–{OBS_YEAR_MAX}",
        flush=True,
    )

    print(
        "Denominator convention:",
        (
            "all raster-grid pixels inside rasterized geometric "
            "common support; masked/nodata count as nonponded"
        ),
        flush=True,
    )

    print("=" * 100 + "\n")

    shelves = read_shelf_names()

    pngu.CROPPED_ANTWIDE_ROOT = str(
        MELTWATER_RASTER_DIR
    )

    pngu.RASTER_GLOB = (
        "ANT_WIDE_*_*_30m_raster.tif"
    )

    (
        shelf_geometries,
        shelf_shapefiles,
        failed_geometry,
    ) = pngu.load_shelf_geometries(
        shelves
    )

    if failed_geometry:
        print(
            "[WARNING] Geometry failures:",
            failed_geometry,
            flush=True,
        )

    prepare_era5_annual_djf(
        ERA5_MONTHLY_PATH,
        ERA5_DJF_PATH,
        year_min=(
            ERA5_YEAR_MIN
        ),
        year_max=(
            ERA5_YEAR_MAX
        ),
        overwrite=(
            OVERWRITE
        ),
        day_weighted=True,
    )

    (
        era5_dataset,
        era5_temperature,
    ) = open_era5_djf(
        ERA5_DJF_PATH
    )

    try:
        if (
            ERA5_GRID_PATH.exists()
            and not OVERWRITE
        ):
            era5_grid = gpd.read_file(
                ERA5_GRID_PATH,
                layer="era5_grid",
            )

        else:
            era5_grid = (
                build_projected_era5_grid(
                    era5_temperature,
                    latitude_maximum=(
                        ERA5_ANTARCTIC_LATITUDE_MAXIMUM
                    ),
                    output_crs=(
                        ERA5_GRID_CRS
                    ),
                )
            )

            save_era5_grid(
                era5_grid,
                ERA5_GRID_PATH,
            )

        crop_results = (
            create_all_shelf_era5_crops(
                shelves,
                shelf_geometries,
                grid=era5_grid,
                temperature=era5_temperature,
                overwrite=OVERWRITE,
            )
        )

        failed_crops = crop_results[
            ~crop_results[
                "status"
            ].isin(
                [
                    "OK",
                    "exists",
                ]
            )
        ]

        if not failed_crops.empty:
            raise RuntimeError(
                "ERA5 shelf crops failed:\n"
                f"{failed_crops.to_string(index=False)}"
            )

        (
            observations,
            failed_shelves,
            processing_results,
        ) = build_all_observations(
            shelves,
            shelf_geometries,
            era5_grid,
        )

        if not failed_shelves.empty:
            raise RuntimeError(
                "Shelf observation processing failed. "
                f"See {FAILED_SHELVES_PATH}."
            )

        validate_final_observations(
            observations,
            shelves,
        )

        coverage = summarize_coverage(
            observations
        )

        shelf_metadata = (
            summarize_shelf_metadata(
                observations
            )
        )

        atomic_csv_write(
            coverage,
            COVERAGE_SUMMARY_PATH,
        )

        atomic_csv_write(
            shelf_metadata,
            SHELF_METADATA_PATH,
        )

        if MAKE_DIAGNOSTIC_FIGURES:
            create_partial_coverage_figure(
                shelves,
                shelf_geometries,
                era5_grid,
                year=DIAGNOSTIC_YEAR,
            )

        elapsed_seconds = (
            time.time()
            - start_time
        )

        metadata = {
            "analysis": (
                "ERA5 and Landsat visible-ponding "
                "common-support preprocessing"
            ),
            "era5_monthly_path": str(
                ERA5_MONTHLY_PATH.resolve()
            ),
            "era5_monthly_sha256": (
                file_sha256(
                    ERA5_MONTHLY_PATH
                )
            ),
            "meltwater_raster_directory": str(
                MELTWATER_RASTER_DIR.resolve()
            ),
            "shelf_metrics_table": (
                str(
                    SHELF_METRICS_TABLE.resolve()
                )
                if SHELF_METRICS_TABLE
                is not None
                else None
            ),
            "observation_period": [
                OBS_YEAR_MIN,
                OBS_YEAR_MAX,
            ],
            "era5_requested_period": [
                ERA5_YEAR_MIN,
                ERA5_YEAR_MAX,
            ],
            "djf_definition": (
                "DJF YYYY = Dec YYYY-1 + Jan YYYY + Feb YYYY"
            ),
            "djf_day_weighted": True,
            "era5_grid_crs": (
                ERA5_GRID_CRS
            ),
            "rasterization_all_touched": (
                ALL_TOUCHED
            ),
            "geometric_common_support_definition": (
                "ERA5 cell intersection with shelf polygon "
                "and melt-raster rectangular footprint"
            ),
            "denominator_convention": (
                "Every raster-grid pixel with a rasterized common-support "
                "label greater than zero contributes to n_pixels_30m. "
                "Masked, nodata, and nonfinite source pixels inside the "
                "geometry are treated as nonponded."
            ),
            "numerator_convention": (
                "A pixel contributes to y_ponded_pixels only when the "
                "source raster value is exactly 1 and originally valid."
            ),
            "source_valid_pixel_fields": (
                "n_source_valid_pixels_qa and "
                "n_source_masked_or_nodata_pixels_qa are QA fields only "
                "and do not alter n_pixels_30m."
            ),
            "n_requested_shelves": int(
                len(
                    shelves
                )
            ),
            "n_output_shelves": int(
                observations[
                    "shelf"
                ].nunique()
            ),
            "n_output_rows": int(
                len(
                    observations
                )
            ),
            "n_positive_rows": int(
                (
                    observations[
                        "y_ponded_pixels"
                    ]
                    > 0
                ).sum()
            ),
            "n_zero_rows": int(
                (
                    observations[
                        "y_ponded_pixels"
                    ]
                    == 0
                ).sum()
            ),
            "observed_pixel_pooled_fraction": float(
                observations[
                    "y_ponded_pixels"
                ].sum()
                / observations[
                    "n_pixels_30m"
                ].sum()
            ),
            "combined_observation_parquet": str(
                COMBINED_OBSERVATION_PARQUET.resolve()
            ),
            "combined_observation_sha256": (
                file_sha256(
                    COMBINED_OBSERVATION_PARQUET
                )
            ),
            "combined_observation_csv": str(
                COMBINED_OBSERVATION_CSV.resolve()
            ),
            "era5_djf_file": str(
                ERA5_DJF_PATH.resolve()
            ),
            "era5_grid_file": str(
                ERA5_GRID_PATH.resolve()
            ),
            "number_of_workers": (
                N_WORKERS
            ),
            "elapsed_seconds": float(
                elapsed_seconds
            ),
        }

        atomic_json_write(
            metadata,
            PREPROCESSING_METADATA_PATH,
        )

        atomic_json_write(
            software_versions(),
            SOFTWARE_VERSIONS_PATH,
        )

        print("\n" + "=" * 100)
        print(
            "FINAL DATASET SUMMARY"
        )
        print("=" * 100)

        print(
            "Rows:",
            f"{len(observations):,}",
        )

        print(
            "Shelves:",
            observations[
                "shelf"
            ].nunique(),
        )

        print(
            "Years:",
            (
                f"{observations['year'].min()}–"
                f"{observations['year'].max()}"
            ),
        )

        print(
            "Positive rows:",
            f"{(observations['y_ponded_pixels'] > 0).sum():,}",
        )

        print(
            "Zero rows:",
            f"{(observations['y_ponded_pixels'] == 0).sum():,}",
        )

        print(
            "Observed pixel-pooled fraction:",
            (
                observations[
                    "y_ponded_pixels"
                ].sum()
                / observations[
                    "n_pixels_30m"
                ].sum()
            ),
        )

        print(
            "Output:",
            COMBINED_OBSERVATION_PARQUET.resolve(),
        )

        print(
            "\nDone.",
            flush=True,
        )

    finally:
        era5_dataset.close()


# Run Main Functions

if __name__ == "__main__":
    mp.freeze_support()
    main()