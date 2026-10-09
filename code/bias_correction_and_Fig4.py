#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bias-Correction Script and Figure 4.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import hashlib
import pickle
import warnings

import arviz as az
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio

from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

PROJECT_DIR = "/raid01/mafields/project_two"

MONTHLY_CMIP_PATH = os.path.join(
    PROJECT_DIR,
    "cropped_monthly_both_scenarios_contains.pkl",
)

MONTHLY_ERA5_PATH = os.path.join(
    PROJECT_DIR,
    "era5_monthly_dict_contains.pkl",
)

MODEL_ROOT = "row_binomial_vs_beta_binomial_scenario_models"

PENINSULA_MODEL_DIR = os.path.join(
    MODEL_ROOT,
    "era5_antarctic_peninsula_row_2006_2020",
)

PENINSULA_TRACE_PATH = os.path.join(
    PENINSULA_MODEL_DIR,
    "era5_antarctic_peninsula_row_binomial_trace.nc",
)

PENINSULA_TRAINING_PATH = os.path.join(
    PENINSULA_MODEL_DIR,
    "era5_antarctic_peninsula_row_model_dataframe.parquet",
)

SHAPEFILE_DIR = (
    "/raid01/mafields/tas/MODELS_filtered/ssp585/shape_files"
)

LANDSAT_TIF_PATH = os.environ.get(
    "LANDSAT_TIF_PATH",
    os.path.join(
        PROJECT_DIR,
        "00000-20080319-092059124.tif",
    ),
)

OUT_DIR = os.path.join(
    PROJECT_DIR,
    "figure4_fast_validation",
)

os.makedirs(OUT_DIR, exist_ok=True)

BIASCORRECTED_DJF_PATH = os.path.join(
    OUT_DIR,
    "biascorrected_cmip6_djf_cells_2021_2300.parquet",
)

PROJECTED_SHELF_YEAR_PATH = os.path.join(
    OUT_DIR,
    "peninsula_binomial_projected_shelf_year_2021_2300.parquet",
)

FIGURE4_PATH = os.path.join(
    OUT_DIR,
    "figure4_fast_validation.png",
)

FIGURE4_PDF_PATH = os.path.join(
    OUT_DIR,
    "figure4_fast_validation.pdf",
)

IDW_CACHE_DIR = os.path.join(
    OUT_DIR,
    "idw_cache",
)

os.makedirs(IDW_CACHE_DIR, exist_ok=True)


MAX_POSTERIOR_DRAWS = 0

REUSE_PROJECTED_SHELF_YEAR = False
REUSE_BIASCORRECTED_DJF = True

SCENARIOS = ("ssp245", "ssp585")

OVERLAP_START = 1979
OVERLAP_END = 2025

FUTURE_START_YEAR = 2021
FUTURE_END_YEAR = 2300

RANDOM_SEED = 42

FIGURE_DPI = 600
SAVE_PDF = True

TARGET_CRS = "EPSG:3031"

IDW_K = 4
IDW_POWER = 2.0
MAX_NEAREST_DISTANCE_M = 250_000

EXCLUDE_SHELVES = {
    "LarsenB pre2002",
}

MAP_SPECS = (
    ("ssp245", 2100, "A"),
    ("ssp585", 2100, "B"),
    ("ssp585", 2300, "C"),
)

SCENARIO_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
}

SCENARIO_COLORS = {
    "ssp245": "#0072B2",
    "ssp585": "#D55E00",
}

THRESHOLD_CENTRAL_PERCENT = 0.87
THRESHOLD_LOW_PERCENT = 0.72
THRESHOLD_HIGH_PERCENT = 1.26

THRESHOLD_COLOR = "#C51B7D"

PROJECTION_MODEL_CASE = "antarctic_peninsula_binomial"
PROJECTION_MODEL_LABEL = "Peninsula-trained Binomial"

SHELF_NAME_MAP = {
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
    "lbisgoneoutline": "LarsenB pre2002",
    "larsenbpre2002": "LarsenB pre2002",
    "larsenbprecollapse": "LarsenB pre2002",
    "larsenc": "LarsenC",
    "larsend": "LarsenD",
    "lazarev": "Lazarev",
    "mariner": "Mariner",
    "mertz": "Mertz",
    "moscowuniversity": "Moscow University",
    "nansen": "Nansen",
    "nickerson": "Nickerson",
    "nivl": "Nivl",
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
    "tucker": "Tucker",
    "venable": "Venable",
    "vigrid": "Vigrid",
    "west": "West",
    "withrow": "Withrow",
}


def normalize_shelf_name(value):
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
    if pd.isna(value):
        return np.nan

    normalized = normalize_shelf_name(value)

    return SHELF_NAME_MAP.get(
        normalized,
        str(value).strip(),
    )


def canonicalize_shelf_dictionary(data):
    output = {}

    for key, value in data.items():
        output[canonical_shelf_name(key)] = value

    return output


def canonicalize_cmip_dictionary(data):
    return {
        scenario: canonicalize_shelf_dictionary(shelves)
        for scenario, shelves in data.items()
    }

def check_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def load_pickle(path):
    check_file(path)

    with open(path, "rb") as file:
        return pickle.load(file)


def get_value(obj, key, index):
    if isinstance(obj, pd.DataFrame):
        return obj[key].iloc[index]

    return obj[key][index]


def number_of_models(obj):
    if isinstance(obj, pd.DataFrame):
        return len(obj)

    return len(obj["Model"])


def to_1d(value, dtype=float):
    return (
        np.asarray(value, dtype=dtype)
        .squeeze()
        .reshape(-1)
    )


def to_temperature_2d(value, expected_time):
    array = np.asarray(value)

    while array.dtype == object and array.size == 1:
        array = np.asarray(array.item())

    if array.dtype == object:
        try:
            array = array.astype(float)
        except (TypeError, ValueError):
            array = np.stack(
                [
                    np.asarray(item, dtype=float).squeeze()
                    for item in array.ravel()
                ],
                axis=0,
            )
    else:
        array = array.astype(float)

    array = np.squeeze(array)

    if array.ndim == 1:
        if array.size % expected_time != 0:
            raise ValueError(
                "Temperature cannot be reshaped to time × cell."
            )

        array = array.reshape(expected_time, -1)

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D temperature, received {array.shape}."
        )

    if (
        array.shape[0] != expected_time
        and array.shape[1] == expected_time
    ):
        array = array.T

    if array.shape[0] != expected_time:
        raise ValueError(
            "Temperature and time dimensions differ."
        )

    if np.nanmedian(array) > 100:
        array = array - 273.15

    return array.astype(np.float32, copy=False)


def longitude_to_180(longitude):
    longitude = np.asarray(longitude, dtype=float)

    return (
        (longitude + 180.0) % 360.0
    ) - 180.0


def weighted_mean_by_latitude(values, latitude):
    values = np.asarray(values, dtype=float)
    latitude = np.asarray(latitude, dtype=float)

    weights = np.cos(
        np.deg2rad(latitude)
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


COORDINATE_TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    TARGET_CRS,
    always_xy=True,
)

MEMORY_IDW_CACHE = {}


def projected_xy(longitude, latitude):
    x, y = COORDINATE_TRANSFORMER.transform(
        longitude_to_180(longitude),
        np.asarray(latitude, dtype=float),
    )

    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


def coordinate_digest(*arrays):
    digest = hashlib.sha1()

    for array in arrays:
        normalized = np.asarray(
            array,
            dtype=np.float64,
        ).reshape(-1)

        digest.update(
            np.asarray(
                normalized.shape,
                dtype=np.int64,
            ).tobytes()
        )
        digest.update(normalized.tobytes())

    return digest.hexdigest()


def idw_cache_path(
    source_latitude,
    source_longitude,
    target_latitude,
    target_longitude,
):
    digest = coordinate_digest(
        source_latitude,
        longitude_to_180(source_longitude),
        target_latitude,
        longitude_to_180(target_longitude),
    )

    return os.path.join(
        IDW_CACHE_DIR,
        f"idw_{digest}.npz",
    )


def build_idw_weights(
    source_latitude,
    source_longitude,
    target_latitude,
    target_longitude,
):
    source_latitude = to_1d(source_latitude)
    source_longitude = longitude_to_180(
        to_1d(source_longitude)
    )
    target_latitude = to_1d(target_latitude)
    target_longitude = longitude_to_180(
        to_1d(target_longitude)
    )

    valid_source = (
        np.isfinite(source_latitude)
        & np.isfinite(source_longitude)
    )

    if not valid_source.any():
        raise ValueError(
            "No finite source coordinates."
        )

    source_x, source_y = projected_xy(
        source_longitude[valid_source],
        source_latitude[valid_source],
    )

    target_x, target_y = projected_xy(
        target_longitude,
        target_latitude,
    )

    tree = cKDTree(
        np.column_stack([
            source_x,
            source_y,
        ])
    )

    k = min(
        IDW_K,
        int(valid_source.sum()),
    )

    distance, local_index = tree.query(
        np.column_stack([
            target_x,
            target_y,
        ]),
        k=k,
        workers=-1,
    )

    if k == 1:
        distance = distance[:, None]
        local_index = local_index[:, None]

    source_index = (
        np.where(valid_source)[0][local_index]
    )

    exact = distance <= 1e-12
    exact_rows = exact.any(axis=1)

    weights = np.zeros_like(
        distance,
        dtype=np.float64,
    )

    nonexact_rows = ~exact_rows

    if nonexact_rows.any():
        raw_weights = (
            1.0
            / np.maximum(
                distance[nonexact_rows],
                1.0,
            ) ** IDW_POWER
        )

        raw_sums = raw_weights.sum(
            axis=1,
            keepdims=True,
        )

        weights[nonexact_rows] = np.divide(
            raw_weights,
            raw_sums,
            out=np.zeros_like(raw_weights),
            where=raw_sums > 0,
        )

    if exact_rows.any():
        exact_count = exact[exact_rows].sum(
            axis=1,
            keepdims=True,
        )

        weights[exact_rows] = (
            exact[exact_rows]
            / exact_count
        )

    too_far = (
        distance[:, 0]
        > MAX_NEAREST_DISTANCE_M
    )

    weights[too_far] = 0.0

    return (
        source_index.astype(np.int32),
        weights.astype(np.float32),
    )


def get_idw_weights(
    source_latitude,
    source_longitude,
    target_latitude,
    target_longitude,
):
    cache_path = idw_cache_path(
        source_latitude,
        source_longitude,
        target_latitude,
        target_longitude,
    )

    if cache_path in MEMORY_IDW_CACHE:
        return MEMORY_IDW_CACHE[cache_path]

    if os.path.isfile(cache_path):
        with np.load(cache_path) as cached:
            result = (
                cached["indices"],
                cached["weights"],
            )

        MEMORY_IDW_CACHE[cache_path] = result

        return result

    result = build_idw_weights(
        source_latitude,
        source_longitude,
        target_latitude,
        target_longitude,
    )

    np.savez_compressed(
        cache_path,
        indices=result[0],
        weights=result[1],
    )

    MEMORY_IDW_CACHE[cache_path] = result

    return result


def apply_idw(
    temperature,
    source_indices,
    weights,
    chunk_size=240,
):
    temperature = np.asarray(
        temperature,
        dtype=np.float32,
    )

    number_of_times = temperature.shape[0]
    number_of_targets = source_indices.shape[0]

    output = np.full(
        (number_of_times, number_of_targets),
        np.nan,
        dtype=np.float32,
    )

    for start in range(
        0,
        number_of_times,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            number_of_times,
        )

        selected = temperature[
            start:stop,
            source_indices,
        ]

        valid = np.isfinite(selected)

        effective_weights = (
            weights[None, :, :]
            * valid
        )

        denominator = effective_weights.sum(
            axis=2
        )

        numerator = np.nansum(
            selected * effective_weights,
            axis=2,
        )

        output[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.full_like(
                numerator,
                np.nan,
                dtype=np.float32,
            ),
            where=denominator > 0,
        )

    return output


def regrid_to_reference(
    temperature,
    source_latitude,
    source_longitude,
    target_latitude,
    target_longitude,
):
    source_latitude = to_1d(source_latitude)
    source_longitude = longitude_to_180(
        to_1d(source_longitude)
    )
    target_latitude = to_1d(target_latitude)
    target_longitude = longitude_to_180(
        to_1d(target_longitude)
    )

    same_grid = (
        len(source_latitude)
        == len(target_latitude)
        and np.allclose(
            source_latitude,
            target_latitude,
            equal_nan=True,
        )
        and np.allclose(
            source_longitude,
            target_longitude,
            equal_nan=True,
        )
    )

    if same_grid:
        return np.asarray(
            temperature,
            dtype=np.float32,
        ).copy()

    source_indices, weights = get_idw_weights(
        source_latitude,
        source_longitude,
        target_latitude,
        target_longitude,
    )

    return apply_idw(
        temperature,
        source_indices,
        weights,
    )


def reference_arrays(reference, shelf):
    year = to_1d(
        get_value(reference, "year", 0),
        dtype=np.int32,
    )

    month = to_1d(
        get_value(reference, "month", 0),
        dtype=np.int16,
    )

    temperature = to_temperature_2d(
        get_value(reference, "cropped temp", 0),
        len(year),
    )

    latitude = to_1d(
        get_value(reference, "cropped lat", 0),
        dtype=np.float32,
    )

    longitude = longitude_to_180(
        to_1d(
            get_value(
                reference,
                "cropped lon",
                0,
            ),
            dtype=np.float32,
        )
    ).astype(np.float32)

    if temperature.shape[1] != len(latitude):
        raise ValueError(
            f"{shelf}: ERA5 temperature/grid mismatch."
        )

    return {
        "year": year,
        "month": month,
        "temperature": temperature,
        "latitude": latitude,
        "longitude": longitude,
    }


def bias_correct_monthly(
    *,
    cmip_year,
    cmip_month,
    cmip_temperature,
    cmip_latitude,
    cmip_longitude,
    reference,
):
    cmip_on_reference = regrid_to_reference(
        cmip_temperature,
        cmip_latitude,
        cmip_longitude,
        reference["latitude"],
        reference["longitude"],
    )

    cmip_lookup = {
        (int(year), int(month)): index
        for index, (year, month) in enumerate(
            zip(cmip_year, cmip_month)
        )
    }

    reference_lookup = {
        (int(year), int(month)): index
        for index, (year, month) in enumerate(
            zip(
                reference["year"],
                reference["month"],
            )
        )
    }

    common_times = [
        key
        for key in (
            set(cmip_lookup)
            & set(reference_lookup)
        )
        if (
            OVERLAP_START
            <= key[0]
            <= OVERLAP_END
        )
    ]

    if not common_times:
        raise ValueError(
            "No CMIP6–ERA5 monthly overlap."
        )

    monthly_bias = np.zeros(
        (
            12,
            len(reference["latitude"]),
        ),
        dtype=np.float32,
    )

    for target_month in range(1, 13):
        keys = [
            key
            for key in common_times
            if key[1] == target_month
        ]

        if not keys:
            continue

        cmip_indices = [
            cmip_lookup[key]
            for key in keys
        ]

        reference_indices = [
            reference_lookup[key]
            for key in keys
        ]

        monthly_bias[target_month - 1] = (
            np.nanmean(
                cmip_on_reference[cmip_indices]
                - reference["temperature"][
                    reference_indices
                ],
                axis=0,
            )
        )

    monthly_bias = np.nan_to_num(
        monthly_bias,
        nan=0.0,
    )

    corrected = cmip_on_reference.copy()

    for target_month in range(1, 13):
        selected = (
            cmip_month == target_month
        )

        corrected[selected] -= (
            monthly_bias[target_month - 1]
        )

    return corrected


def aggregate_djf(
    year,
    month,
    temperature,
):
    lookup = {
        (int(y), int(m)): index
        for index, (y, m) in enumerate(
            zip(year, month)
        )
    }

    minimum_year = max(
        FUTURE_START_YEAR,
        int(np.min(year)),
    )

    maximum_year = min(
        FUTURE_END_YEAR,
        int(np.max(year)),
    )

    output_year = []
    output_temperature = []

    for target_year in range(
        minimum_year,
        maximum_year + 1,
    ):
        keys = (
            (target_year - 1, 12),
            (target_year, 1),
            (target_year, 2),
        )

        if not all(key in lookup for key in keys):
            continue

        indices = [
            lookup[key]
            for key in keys
        ]

        output_year.append(target_year)

        output_temperature.append(
            np.nanmean(
                temperature[indices],
                axis=0,
            )
        )

    if not output_year:
        return (
            np.empty(0, dtype=np.int16),
            np.empty(
                (0, temperature.shape[1]),
                dtype=np.float32,
            ),
        )

    return (
        np.asarray(
            output_year,
            dtype=np.int16,
        ),
        np.asarray(
            output_temperature,
            dtype=np.float32,
        ),
    )


def create_biascorrected_djf_cells(
    monthly_cmip,
    monthly_era5,
):
    frames = []
    total_success = 0
    total_skipped = 0

    for scenario in SCENARIOS:
        common_shelves = sorted(
            (
                set(monthly_cmip[scenario])
                & set(monthly_era5)
            )
            - EXCLUDE_SHELVES
        )

        scenario_success = 0

        for shelf in common_shelves:
            try:
                reference = reference_arrays(
                    monthly_era5[shelf],
                    shelf,
                )
            except Exception:
                total_skipped += 1
                continue

            cmip_shelf = (
                monthly_cmip[scenario][shelf]
            )

            for model_index in range(
                number_of_models(cmip_shelf)
            ):
                model_name = str(
                    get_value(
                        cmip_shelf,
                        "Model",
                        model_index,
                    )
                )

                try:
                    year = to_1d(
                        get_value(
                            cmip_shelf,
                            "year",
                            model_index,
                        ),
                        dtype=np.int32,
                    )

                    month = to_1d(
                        get_value(
                            cmip_shelf,
                            "month",
                            model_index,
                        ),
                        dtype=np.int16,
                    )

                    temperature = to_temperature_2d(
                        get_value(
                            cmip_shelf,
                            "cropped temp",
                            model_index,
                        ),
                        len(year),
                    )

                    latitude = to_1d(
                        get_value(
                            cmip_shelf,
                            "cropped lat",
                            model_index,
                        ),
                        dtype=np.float32,
                    )

                    longitude = longitude_to_180(
                        to_1d(
                            get_value(
                                cmip_shelf,
                                "cropped lon",
                                model_index,
                            ),
                            dtype=np.float32,
                        )
                    ).astype(np.float32)

                    corrected = bias_correct_monthly(
                        cmip_year=year,
                        cmip_month=month,
                        cmip_temperature=temperature,
                        cmip_latitude=latitude,
                        cmip_longitude=longitude,
                        reference=reference,
                    )

                    (
                        djf_year,
                        djf_temperature,
                    ) = aggregate_djf(
                        year,
                        month,
                        corrected,
                    )

                    if len(djf_year) == 0:
                        continue

                    number_of_years = len(djf_year)
                    number_of_cells = len(
                        reference["latitude"]
                    )

                    frame = pd.DataFrame({
                        "scenario": scenario,
                        "shelf": shelf,
                        "cmip_model": model_name,
                        "year": np.repeat(
                            djf_year,
                            number_of_cells,
                        ),
                        "cell_index": np.tile(
                            np.arange(
                                number_of_cells,
                                dtype=np.int32,
                            ),
                            number_of_years,
                        ),
                        "lat": np.tile(
                            reference[
                                "latitude"
                            ],
                            number_of_years,
                        ),
                        "lon": np.tile(
                            reference[
                                "longitude"
                            ],
                            number_of_years,
                        ),
                        "temperature_c": (
                            djf_temperature.reshape(-1)
                        ),
                    })

                    frame = frame.loc[
                        np.isfinite(
                            frame["temperature_c"]
                        )
                    ]

                    if not frame.empty:
                        frames.append(frame)
                        scenario_success += 1
                        total_success += 1

                except Exception:
                    total_skipped += 1

        print(
            f"{scenario}: {scenario_success} "
            "shelf-model series corrected"
        )

    if not frames:
        raise RuntimeError(
            "Bias correction produced no valid output."
        )

    output = pd.concat(
        frames,
        ignore_index=True,
    )

    output["scenario"] = output[
        "scenario"
    ].astype("category")

    output["shelf"] = output[
        "shelf"
    ].astype("category")

    output["cmip_model"] = output[
        "cmip_model"
    ].astype("category")

    output["year"] = output[
        "year"
    ].astype(np.int16)

    output["lat"] = output[
        "lat"
    ].astype(np.float32)

    output["lon"] = output[
        "lon"
    ].astype(np.float32)

    output["temperature_c"] = output[
        "temperature_c"
    ].astype(np.float32)

    output.to_parquet(
        BIASCORRECTED_DJF_PATH,
        index=False,
        compression="zstd",
    )

    print(
        f"Bias correction complete: "
        f"{total_success} successful, "
        f"{total_skipped} skipped"
    )

    return output

#AP Peninsula temperature-response binomial model
#Applying as a sensitivity test exploring how ice shelves
#change if they responded like shelves in the AP

def first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"None of these columns were found: {candidates}"
    )


def posterior_variable(trace, candidates):
    available = list(
        trace.posterior.data_vars
    )

    return first_present(
        available,
        candidates,
    )


def load_peninsula_model():
    check_file(PENINSULA_TRACE_PATH)
    check_file(PENINSULA_TRAINING_PATH)

    trace = az.from_netcdf(
        PENINSULA_TRACE_PATH
    )

    training = pd.read_parquet(
        PENINSULA_TRAINING_PATH
    )

    temperature_column = first_present(
        training.columns,
        (
            "era5_t2m_djf_c",
            "era5_t2m_djf",
            "temperature_c",
            "temperature",
            "temp_c",
        ),
    )

    training_temperature = pd.to_numeric(
        training[temperature_column],
        errors="coerce",
    ).to_numpy(float)

    training_temperature = (
        training_temperature[
            np.isfinite(training_temperature)
        ]
    )

    if training_temperature.size == 0:
        raise RuntimeError(
            "No valid model-training temperatures."
        )

    alpha_name = posterior_variable(
        trace,
        (
            "alpha",
            "intercept",
        ),
    )

    beta_name = posterior_variable(
        trace,
        (
            "beta_temp",
            "beta_temperature",
            "temperature_slope",
            "beta",
        ),
    )

    alpha = (
        trace.posterior[alpha_name]
        .values
        .reshape(-1)
        .astype(np.float32)
    )

    beta = (
        trace.posterior[beta_name]
        .values
        .reshape(-1)
        .astype(np.float32)
    )

    if len(alpha) != len(beta):
        raise RuntimeError(
            "Posterior alpha and beta lengths differ."
        )

    if (
        MAX_POSTERIOR_DRAWS > 0
        and len(alpha) > MAX_POSTERIOR_DRAWS
    ):
        random_generator = (
            np.random.default_rng(
                RANDOM_SEED
            )
        )

        selected = np.sort(
            random_generator.choice(
                len(alpha),
                size=MAX_POSTERIOR_DRAWS,
                replace=False,
            )
        )

        alpha = alpha[selected]
        beta = beta[selected]

    return {
        "case": PROJECTION_MODEL_CASE,
        "label": PROJECTION_MODEL_LABEL,
        "alpha": alpha,
        "beta": beta,
        "temperature_mean": np.float32(
            training_temperature.mean()
        ),
        "temperature_sd": np.float32(
            training_temperature.std(ddof=0)
        ),
    }


def inverse_cloglog(linear_predictor):
    linear_predictor = np.clip(
        linear_predictor,
        -30.0,
        20.0,
    )

    return (
        1.0
        - np.exp(
            -np.exp(linear_predictor)
        )
    )


def apply_peninsula_model(
    biascorrected_cells,
    response_model,
):
    work = biascorrected_cells.copy()

    work["_weight"] = np.cos(
        np.deg2rad(
            work["lat"].to_numpy(
                dtype=np.float32
            )
        )
    ).astype(np.float32)

    group_columns = [
        "scenario",
        "shelf",
        "cmip_model",
        "year",
    ]

    rows = []

    grouped = work.groupby(
        group_columns,
        observed=True,
        sort=False,
    )

    for group_key, group in grouped:
        temperature = group[
            "temperature_c"
        ].to_numpy(np.float32)

        latitude = group[
            "lat"
        ].to_numpy(np.float32)

        weights = group[
            "_weight"
        ].to_numpy(np.float32)

        valid = (
            np.isfinite(temperature)
            & np.isfinite(latitude)
            & np.isfinite(weights)
            & (weights > 0)
        )

        if not valid.any():
            continue

        temperature = temperature[valid]
        latitude = latitude[valid]
        weights = weights[valid]
        weights /= weights.sum()

        standardized_temperature = (
            temperature
            - response_model[
                "temperature_mean"
            ]
        ) / response_model[
            "temperature_sd"
        ]

        linear_predictor = (
            response_model["alpha"][:, None]
            + response_model["beta"][:, None]
            * standardized_temperature[None, :]
        )

        cell_fraction = inverse_cloglog(
            linear_predictor
        )

        shelf_draws = (
            cell_fraction @ weights
        )

        q025, median, q975 = np.quantile(
            shelf_draws,
            [0.025, 0.500, 0.975],
        )

        (
            scenario,
            shelf,
            cmip_model,
            year,
        ) = group_key

        rows.append({
            "response_model_case":
                response_model["case"],
            "response_model_label":
                response_model["label"],
            "scenario": str(scenario),
            "shelf": str(shelf),
            "cmip_model": str(cmip_model),
            "year": int(year),
            "temperature_c":
                weighted_mean_by_latitude(
                    temperature,
                    latitude,
                ),
            "ponding_percent_mean":
                100.0 * float(
                    shelf_draws.mean()
                ),
            "ponding_percent_q025":
                100.0 * float(q025),
            "ponding_percent_median":
                100.0 * float(median),
            "ponding_percent_q975":
                100.0 * float(q975),
        })

    output = pd.DataFrame.from_records(rows)

    if output.empty:
        raise RuntimeError(
            "Response-model projection was empty."
        )

    output.to_parquet(
        PROJECTED_SHELF_YEAR_PATH,
        index=False,
        compression="zstd",
    )

    print(
        f"Response model complete: "
        f"{len(output):,} shelf-model-year rows"
    )

    return output

def load_shelf_shapes():
    vector_paths = sorted(
        glob.glob(
            os.path.join(
                SHAPEFILE_DIR,
                "*.shp",
            )
        )
    )

    frames = []

    for path in vector_paths:
        basename = os.path.splitext(
            os.path.basename(path)
        )[0]

        shelf = canonical_shelf_name(
            basename
        )

        if shelf in EXCLUDE_SHELVES:
            continue

        try:
            frame = gpd.read_file(path)

            if frame.empty:
                continue

            frame = frame.loc[
                frame.geometry.notna()
                & ~frame.geometry.is_empty
            ].copy()

            if frame.empty:
                continue

            if frame.crs is None:
                frame = frame.set_crs(
                    TARGET_CRS,
                    allow_override=True,
                )

            frame = frame.to_crs(
                TARGET_CRS
            )

            frame["shelf"] = shelf

            frames.append(
                frame[
                    ["shelf", "geometry"]
                ]
            )

        except Exception:
            continue

    if not frames:
        raise RuntimeError(
            "No shelf shapes could be loaded."
        )

    shelves = gpd.GeoDataFrame(
        pd.concat(
            frames,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=TARGET_CRS,
    )

    return shelves.dissolve(
        by="shelf",
        as_index=False,
    )


def shelf_map_values(
    projections,
    scenario,
    requested_year,
):
    scenario_data = projections.loc[
        projections["scenario"].eq(
            scenario
        )
    ].copy()

    if scenario_data.empty:
        raise RuntimeError(
            f"No projections for {scenario}."
        )

    available_years = (
        scenario_data["year"]
        .dropna()
        .astype(int)
        .unique()
    )

    year_used = int(
        available_years[
            np.argmin(
                np.abs(
                    available_years
                    - requested_year
                )
            )
        ]
    )

    subset = scenario_data.loc[
        scenario_data["year"].eq(
            year_used
        )
    ]

    values = (
        subset.groupby(
            "shelf",
            observed=True,
        )["ponding_percent_median"]
        .median()
        .rename("ponding_percent")
        .reset_index()
    )

    return values, year_used


def antarctic_time_series(projections):
    per_model = (
        projections.groupby(
            [
                "scenario",
                "year",
                "cmip_model",
            ],
            observed=True,
        )["ponding_percent_median"]
        .mean()
        .rename("ponding_percent")
        .reset_index()
    )

    output = (
        per_model.groupby(
            ["scenario", "year"],
            observed=True,
        )["ponding_percent"]
        .agg(
            median="median",
            q10=lambda values:
                np.quantile(values, 0.10),
            q90=lambda values:
                np.quantile(values, 0.90),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    return output

# Manuscript Figure 4 Settings: Bias Correction using both scenarios
# Focusing on the AP binomial model

CM_PER_INCH = 2.54

FIGURE4_WIDTH_CM = float(
    os.environ.get("FIGURE4_WIDTH_CM", "22.0")
)

FIGURE4_HEIGHT_CM = float(
    os.environ.get("FIGURE4_HEIGHT_CM", "15.5")
)

# PNAS large figure restrictions
if FIGURE4_WIDTH_CM > 22.0 or FIGURE4_HEIGHT_CM > 18.0:
    raise ValueError(
        "Figure 4 must be no larger than 22 cm wide × 18 cm high."
    )

FIGURE4_WIDTH = FIGURE4_WIDTH_CM / CM_PER_INCH
FIGURE4_HEIGHT = FIGURE4_HEIGHT_CM / CM_PER_INCH

FIGURE_DPI = int(
    os.environ.get("FIGURE_DPI", "600")
)

TIME_SERIES_START_YEAR = FUTURE_START_YEAR
TIME_SERIES_END_YEAR = FUTURE_END_YEAR

INSET_START_YEAR = 2021
INSET_END_YEAR = 2100
INSET_Y_MAXIMUM = float(
    os.environ.get("INSET_Y_MAXIMUM", "2.0")
)

TIME_SERIES_Y_MAXIMUM = float(
    os.environ.get("TIME_SERIES_Y_MAXIMUM", "0")
)

RASTER_MAX_SIZE = int(
    os.environ.get("RASTER_MAX_SIZE", "3500")
)

READ_RGB_IF_AVAILABLE = (
    os.environ.get("READ_RGB_IF_AVAILABLE", "1") == "1"
)

RASTER_ALPHA = float(
    os.environ.get("RASTER_ALPHA", "0.20")
)

RASTER_STRETCH_LOW = float(
    os.environ.get("RASTER_STRETCH_LOW", "2.0")
)

RASTER_STRETCH_HIGH = float(
    os.environ.get("RASTER_STRETCH_HIGH", "98.0")
)

MAP_BORDER_COLOR = "0.12"
MAP_BORDER_LINEWIDTH = 0.75
WATER_COLOR = "#D7EEF7"

GRATICULE_COLOR = "0.48"
GRATICULE_ALPHA = 0.22
GRATICULE_LINEWIDTH = 0.35

SHARED_POWER_GAMMA = 0.40
SHARED_COLOR_PERCENTILE = 99.5
MINIMUM_SHARED_COLOR_MAXIMUM = 1.0
COLORBAR_NUMBER_OF_INTERVALS = 6

OUTLINE_STYLES = {
    "none": {
        "color": "0.42",
        "linewidth": 0.35,
        "alpha": 0.62,
    },
    "possible": {
        "color": "#FFBF65",
        "linewidth": 0.65,
        "alpha": 0.80,
    },
    "central": {
        "color": "#CC5300",
        "linewidth": 0.75,
        "alpha": 0.85,
    },
    "robust": {
        "color": "#FF0020",
        "linewidth": 0.85,
        "alpha": 0.90,
    },
}

CMAP_FRACTION = LinearSegmentedColormap.from_list(
    "future_visible_ponding",
    [
        (0.000, "#FFFFFF"),
        (0.025, "#F7FBFF"),
        (0.080, "#DEEBF7"),
        (0.160, "#C6DBEF"),
        (0.290, "#9ECAE1"),
        (0.450, "#6BAED6"),
        (0.630, "#4292C6"),
        (0.790, "#2171B5"),
        (0.910, "#08519C"),
        (1.000, "#08306B"),
    ],
    N=256,
)

CMAP_FRACTION.set_under("#FFFFFF")
CMAP_FRACTION.set_bad(color="#FFFFFF", alpha=0.0)
CMAP_FRACTION.set_over("#02152F")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "figure.dpi": 150,
    "savefig.dpi": FIGURE_DPI,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.25",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def format_tick(value, position=None):
    if not np.isfinite(value):
        return ""

    if np.isclose(value, 0.0, atol=1e-12):
        return "0"

    absolute = abs(float(value))

    if absolute >= 10:
        return f"{value:.0f}"

    if absolute >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")

    if absolute >= 0.1:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    return f"{value:.3f}".rstrip("0").rstrip(".")


def calculate_colorbar_ticks(maximum):
    locator = mticker.MaxNLocator(
        nbins=COLORBAR_NUMBER_OF_INTERVALS,
        min_n_ticks=4,
    )

    ticks = locator.tick_values(0.0, maximum)
    ticks = ticks[
        (ticks >= 0.0)
        & (ticks <= maximum)
    ]

    ticks = np.unique(
        np.concatenate([
            np.array([0.0]),
            ticks,
            np.array([maximum]),
        ])
    )

    return ticks


def save_figure4(figure):

    figure.savefig(
        FIGURE4_PATH,
        dpi=FIGURE_DPI,
        facecolor="white",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )

    if SAVE_PDF:
        figure.savefig(
            FIGURE4_PDF_PATH,
            facecolor="white",
            edgecolor="none",
            bbox_inches=None,
            pad_inches=0,
        )

    plt.close(figure)


def first_crossing_year(
    series,
    scenario,
    start_year,
    end_year,
):
    group = (
        series.loc[
            series["scenario"].eq(scenario)
            & (series["year"] >= start_year)
            & (series["year"] <= end_year),
            ["year", "median"],
        ]
        .dropna()
        .sort_values("year")
        .reset_index(drop=True)
    )

    if group.empty:
        return None

    years = group["year"].to_numpy(int)
    median = group["median"].to_numpy(float)

    above = (
        median
        >= THRESHOLD_CENTRAL_PERCENT
    )

    consecutive = np.ones(
        len(group),
        dtype=bool,
    )

    if len(group) > 1:
        consecutive[:-1] = (
            np.diff(years) == 1
        )

    sustained_from_here = np.logical_and.accumulate(
        above[::-1]
    )[::-1]

    continuous_from_here = np.logical_and.accumulate(
        consecutive[::-1]
    )[::-1]

    valid = (
        sustained_from_here
        & continuous_from_here
    )

    indices = np.flatnonzero(valid)

    if indices.size == 0:
        return None

    return int(years[indices[0]])


def draw_threshold_region(axis):
    axis.axhspan(
        THRESHOLD_LOW_PERCENT,
        THRESHOLD_HIGH_PERCENT,
        color=THRESHOLD_COLOR,
        alpha=0.12,
        linewidth=0,
        zorder=0,
    )

    axis.axhline(
        THRESHOLD_CENTRAL_PERCENT,
        color=THRESHOLD_COLOR,
        linestyle="-.",
        linewidth=1.5,
        zorder=3,
    )


def draw_scenario_time_series(
    axis,
    series,
    scenario,
    *,
    show_full_range=True,
    linewidth=2.3,
):
    group = series.loc[
        series["scenario"].eq(scenario)
    ].sort_values("year")

    if group.empty:
        return

    color = SCENARIO_COLORS[scenario]
    years = group["year"].to_numpy(float)

    if show_full_range:
        axis.fill_between(
            years,
            group["minimum"].to_numpy(float),
            group["maximum"].to_numpy(float),
            color=color,
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )

    axis.fill_between(
        years,
        group["q10"].to_numpy(float),
        group["q90"].to_numpy(float),
        color=color,
        alpha=0.24,
        linewidth=0,
        zorder=2,
    )

    axis.plot(
        years,
        group["median"].to_numpy(float),
        color=color,
        linewidth=linewidth,
        zorder=5,
    )


def draw_crossing_lines(
    axis,
    series,
    start_year,
    end_year,
    *,
    annotate=True,
):
    crossing_years = {}

    for scenario in SCENARIOS:
        crossing_year = first_crossing_year(
            series,
            scenario,
            start_year,
            end_year,
        )

        crossing_years[scenario] = crossing_year

        if crossing_year is None:
            continue

        color = SCENARIO_COLORS[scenario]

        axis.axvline(
            crossing_year,
            color=color,
            linestyle="--",
            linewidth=1.5,
            zorder=6,
        )

        if annotate:
            axis.text(
                crossing_year,
                0.96,
                str(crossing_year),
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                rotation=90,
                fontsize=8,
                color=color,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.0,
                },
                zorder=7,
            )

    return crossing_years


def load_landsat_raster():
    check_file(LANDSAT_TIF_PATH)

    with rasterio.open(
        LANDSAT_TIF_PATH
    ) as source:
        if source.crs is None:
            raise ValueError(
                "The Landsat raster has no coordinate reference system."
            )

        scale = max(
            source.width / RASTER_MAX_SIZE,
            source.height / RASTER_MAX_SIZE,
            1.0,
        )

        width = max(
            1,
            int(round(source.width / scale)),
        )

        height = max(
            1,
            int(round(source.height / scale)),
        )

        bands = (
            [1, 2, 3]
            if (
                READ_RGB_IF_AVAILABLE
                and source.count >= 3
            )
            else [1]
        )

        array = source.read(
            bands,
            out_shape=(
                len(bands),
                height,
                width,
            ),
            masked=True,
            resampling=Resampling.bilinear,
        )

        extent = plotting_extent(source)
        raster_crs = source.crs

    print(
        "[INFO] Landsat display:",
        f"{width} × {height}",
    )

    return array, extent, raster_crs


def prepare_landsat_raster(array):
    array = np.ma.array(array)

    if array.ndim != 3:
        return array

    if array.shape[0] == 1:
        band = np.asarray(
            array[0].filled(np.nan),
            dtype=np.float32,
        )

        mask = np.ma.getmaskarray(
            array[0]
        )

        valid = (
            np.isfinite(band)
            & ~mask
        )

        if not valid.any():
            return np.ma.masked_all(
                band.shape,
                dtype=np.float32,
            )

        lower, upper = np.nanpercentile(
            band[valid],
            [
                RASTER_STRETCH_LOW,
                RASTER_STRETCH_HIGH,
            ],
        )

        if upper <= lower:
            return np.ma.array(
                band,
                mask=~valid,
            )

        output = np.clip(
            (band - lower) / (upper - lower),
            0.0,
            1.0,
        )

        return np.ma.array(
            output,
            mask=~valid,
        )

    rgb = np.moveaxis(
        array[:3],
        0,
        -1,
    )

    rgb_data = np.asarray(
        rgb.filled(np.nan),
        dtype=np.float32,
    )

    mask = np.ma.getmaskarray(rgb)

    output = np.zeros_like(
        rgb_data,
        dtype=np.float32,
    )

    valid_pixel = np.ones(
        rgb_data.shape[:2],
        dtype=bool,
    )

    for index in range(3):
        band = rgb_data[..., index]

        valid = (
            np.isfinite(band)
            & ~mask[..., index]
        )

        valid_pixel &= valid

        if not valid.any():
            continue

        lower, upper = np.nanpercentile(
            band[valid],
            [
                RASTER_STRETCH_LOW,
                RASTER_STRETCH_HIGH,
            ],
        )

        if upper <= lower:
            continue

        output[..., index] = np.clip(
            (band - lower) / (upper - lower),
            0.0,
            1.0,
        )

    return np.dstack([
        output,
        valid_pixel.astype(np.float32),
    ])


def plot_landsat_raster(
    axis,
    raster,
    raster_extent,
):
    axis.set_facecolor(WATER_COLOR)

    if (
        raster.ndim == 3
        and raster.shape[-1] == 4
    ):
        image = raster.copy()
        image[..., 3] *= RASTER_ALPHA

        axis.imshow(
            image,
            extent=raster_extent,
            origin="upper",
            interpolation="bilinear",
            zorder=0,
        )

    else:
        axis.imshow(
            raster,
            extent=raster_extent,
            origin="upper",
            cmap="Greys",
            alpha=RASTER_ALPHA,
            interpolation="bilinear",
            zorder=0,
        )


def map_extent_from_shelves(shelves):
    left, bottom, right, top = shelves.total_bounds

    width = right - left
    height = top - bottom

    x_padding = 0.025 * width
    y_padding = 0.025 * height

    return (
        left - x_padding,
        right + x_padding,
        bottom - y_padding,
        top + y_padding,
    )


def prepare_graticule_segments(target_crs):
    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    segments = []

    latitudes = np.linspace(
        -89.5,
        -55.0,
        300,
    )

    for longitude in np.arange(
        -180,
        180,
        30,
    ):
        x, y = transformer.transform(
            np.full_like(latitudes, longitude),
            latitudes,
        )

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() > 1:
            segments.append(
                np.column_stack([
                    np.asarray(x)[valid],
                    np.asarray(y)[valid],
                ])
            )

    longitudes = np.linspace(
        -180,
        180,
        720,
    )

    for latitude in (-80, -75, -70, -65, -60):
        x, y = transformer.transform(
            longitudes,
            np.full_like(longitudes, latitude),
        )

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() > 1:
            segments.append(
                np.column_stack([
                    np.asarray(x)[valid],
                    np.asarray(y)[valid],
                ])
            )

    return segments


def add_graticule(axis, segments):
    from matplotlib.collections import LineCollection

    axis.add_collection(
        LineCollection(
            segments,
            colors=GRATICULE_COLOR,
            linewidths=GRATICULE_LINEWIDTH,
            alpha=GRATICULE_ALPHA,
            zorder=1,
        )
    )


def add_map_border(axis):
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            fill=False,
            edgecolor=MAP_BORDER_COLOR,
            linewidth=MAP_BORDER_LINEWIDTH,
            zorder=100,
            clip_on=False,
        )
    )


def shelf_map_statistics(
    projections,
    scenario,
    requested_year,
):
    """
    Return ensemble shelf statistics at the available year nearest the request.

    Map fill:
        median across CMIP6 models.

    Threshold classes:
        possible: maximum >= lower threshold
        central:  median >= central threshold
        robust:   minimum >= upper threshold
    """

    scenario_data = projections.loc[
        projections["scenario"].eq(scenario)
    ].copy()

    if scenario_data.empty:
        raise RuntimeError(
            f"No projections for {scenario}."
        )

    available_years = (
        scenario_data["year"]
        .dropna()
        .astype(int)
        .unique()
    )

    year_used = int(
        available_years[
            np.argmin(
                np.abs(
                    available_years
                    - requested_year
                )
            )
        ]
    )

    subset = scenario_data.loc[
        scenario_data["year"].eq(year_used)
    ].copy()

    values = (
        subset.groupby(
            "shelf",
            observed=True,
        )["ponding_percent_median"]
        .agg(
            ponding_percent="median",
            ensemble_minimum_percent="min",
            ensemble_maximum_percent="max",
        )
        .reset_index()
    )

    values["exceedance_class"] = "none"

    values.loc[
        values["ensemble_maximum_percent"]
        >= THRESHOLD_LOW_PERCENT,
        "exceedance_class",
    ] = "possible"

    values.loc[
        values["ponding_percent"]
        >= THRESHOLD_CENTRAL_PERCENT,
        "exceedance_class",
    ] = "central"

    values.loc[
        values["ensemble_minimum_percent"]
        >= THRESHOLD_HIGH_PERCENT,
        "exceedance_class",
    ] = "robust"

    return values, year_used


def plot_map_panel(
    axis,
    shelves,
    panel,
    scenario,
    panel_label,
    color_norm,
    map_extent,
    graticule_segments,
    landsat_raster,
    landsat_extent,
):
    mapped = shelves.merge(
        panel["values"],
        on="shelf",
        how="left",
    )

    plot_landsat_raster(
        axis,
        landsat_raster,
        landsat_extent,
    )

    add_graticule(
        axis,
        graticule_segments,
    )

    # Fill polygons without outlines first.
    mapped.plot(
        ax=axis,
        column="ponding_percent",
        cmap=CMAP_FRACTION,
        norm=color_norm,
        alpha=0.78,
        edgecolor="none",
        missing_kwds={
            "color": (0.93, 0.93, 0.93, 0.55),
            "edgecolor": "none",
        },
        zorder=4,
    )

    # Draw threshold-classified outlines separately.
    for classification in (
        "none",
        "possible",
        "central",
        "robust",
    ):
        subset = mapped.loc[
            mapped["exceedance_class"]
            .fillna("none")
            .eq(classification)
        ]

        if subset.empty:
            continue

        style = OUTLINE_STYLES[classification]

        subset.boundary.plot(
            ax=axis,
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=8,
        )

    axis.set_xlim(
        map_extent[0],
        map_extent[1],
    )

    axis.set_ylim(
        map_extent[2],
        map_extent[3],
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
        anchor="C",
    )

    axis.set_axis_off()

    axis.set_title(
        (
            f"{SCENARIO_LABELS[scenario]}, "
            f"{panel['year_used']}"
        ),
        fontsize=10.5,
        fontweight="bold",
        pad=4,
    )

    axis.text(
        0.018,
        0.975,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.45",
            "linewidth": 0.45,
            "alpha": 0.94,
            "pad": 1.1,
        },
        zorder=110,
    )

    add_map_border(axis)


def plot_projection_time_series_with_inset(
    axis,
    series,
):
    draw_threshold_region(axis)

    for scenario in SCENARIOS:
        draw_scenario_time_series(
            axis,
            series,
            scenario,
            show_full_range=True,
            linewidth=2.3,
        )

    crossing_years = draw_crossing_lines(
        axis,
        series,
        TIME_SERIES_START_YEAR,
        TIME_SERIES_END_YEAR,
        annotate=True,
    )

    if TIME_SERIES_Y_MAXIMUM > 0:
        y_maximum = TIME_SERIES_Y_MAXIMUM
    else:
        y_maximum = max(
            THRESHOLD_HIGH_PERCENT * 1.2,
            float(series["q90"].max()) * 1.10,
        )

    axis.set_xlim(
        TIME_SERIES_START_YEAR,
        TIME_SERIES_END_YEAR,
    )

    axis.set_ylim(0, y_maximum)

    axis.set_xlabel(
        "Year",
        fontsize=10,
    )

    axis.set_ylabel(
        "Median ponding (%)",
        fontsize=9,
        labelpad=4,
    )

    axis.text(
        0.018,
        1.15,
        "D",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.45",
            "linewidth": 0.45,
            "alpha": 0.94,
            "pad": 1.1,
        },
        zorder=110,
    )

    axis.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, position: f"{value:g}%"
        )
    )

    axis.tick_params(
        axis="both",
        labelsize=8.5,
    )

    axis.grid(
        True,
        which="major",
        color="0.86",
        linewidth=0.7,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    inset = inset_axes(
        axis,
        width="43%",
        height="58%",
        loc="upper left",
        bbox_to_anchor=(
            0.045,
            0.02,
            0.93,
            0.92,
        ),
        bbox_transform=axis.transAxes,
        borderpad=0.8,
    )

    draw_threshold_region(inset)

    for scenario in SCENARIOS:
        draw_scenario_time_series(
            inset,
            series,
            scenario,
            show_full_range=False,
            linewidth=1.7,
        )

    draw_crossing_lines(
        inset,
        series,
        INSET_START_YEAR,
        INSET_END_YEAR,
        annotate=True,
    )

    inset.set_xlim(
        INSET_START_YEAR,
        INSET_END_YEAR,
    )

    inset.set_ylim(
        0,
        INSET_Y_MAXIMUM,
    )

    inset.set_title(
        "2021–2100",
        fontsize=9,
        fontweight="bold",
        pad=4,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.94,
            "boxstyle": "square,pad=0.20",
        },
    )

    inset.xaxis.set_major_locator(
        mticker.MultipleLocator(20)
    )

    inset.xaxis.set_minor_locator(
        mticker.MultipleLocator(5)
    )

    inset.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, position: f"{value:g}%"
        )
    )

    inset.tick_params(labelsize=8)

    inset.grid(
        True,
        which="major",
        color="0.86",
        linewidth=0.55,
    )

    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("0.30")
        spine.set_linewidth(0.8)

    mark_inset(
        axis,
        inset,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.40",
        linewidth=0.8,
    )

    return crossing_years

def create_figure4(projections):
    """
    Create Figure 4 using the publication formatting of Figure 2.

    The output is 22.0 cm × 15.5 cm by default and therefore remains within
    the requested maximum of 22 cm × 18 cm.
    """

    shelves = load_shelf_shapes()

    (
        landsat_array,
        landsat_extent,
        landsat_crs,
    ) = load_landsat_raster()

    landsat_raster = prepare_landsat_raster(
        landsat_array
    )

    if shelves.crs != landsat_crs:
        shelves = shelves.to_crs(
            landsat_crs
        )

    map_extent = landsat_extent

    graticule_segments = prepare_graticule_segments(
        landsat_crs
    )

    panel_data = {}
    all_values = []

    for (
        scenario,
        requested_year,
        panel_label,
    ) in MAP_SPECS:
        values, year_used = shelf_map_statistics(
            projections,
            scenario,
            requested_year,
        )

        panel_data[
            (scenario, requested_year)
        ] = {
            "values": values,
            "year_used": year_used,
            "panel_label": panel_label,
        }

        all_values.extend(
            values["ponding_percent"].to_numpy(float)
        )

    finite_map_values = np.asarray(
        all_values,
        dtype=float,
    )

    finite_map_values = finite_map_values[
        np.isfinite(finite_map_values)
    ]

    if finite_map_values.size == 0:
        raise RuntimeError(
            "No finite values for Figure 4 maps."
        )

    color_maximum = max(
        MINIMUM_SHARED_COLOR_MAXIMUM,
        float(
            np.percentile(
                finite_map_values,
                SHARED_COLOR_PERCENTILE,
            )
        ),
    )

    color_norm = PowerNorm(
        gamma=SHARED_POWER_GAMMA,
        vmin=0.0,
        vmax=color_maximum,
        clip=False,
    )

    colorbar_ticks = calculate_colorbar_ticks(
        color_maximum
    )

    series = antarctic_time_series(
        projections
    )

    figure = plt.figure(
        figsize=(
            FIGURE4_WIDTH,
            FIGURE4_HEIGHT,
        ),
        facecolor="white",
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.12, 0.72],
        left=0.06,
        right=0.98,
        bottom=0.19,
        top=0.96,
        hspace=0.36,
        wspace=0.035,
    )

    map_axes = [
        figure.add_subplot(
            grid[0, column]
        )
        for column in range(3)
    ]

    time_axis = figure.add_subplot(
        grid[1, :]
    )

    time_position = time_axis.get_position()

    time_axis.set_position([
        0.105,
        time_position.y0,
        0.98 - 0.105,
        time_position.height,
    ])

    for (
        axis,
        (
            scenario,
            requested_year,
            panel_label,
        ),
    ) in zip(
        map_axes,
        MAP_SPECS,
    ):
        panel = panel_data[
            (scenario, requested_year)
        ]

        plot_map_panel(
            axis=axis,
            shelves=shelves,
            panel=panel,
            scenario=scenario,
            panel_label=panel_label,
            color_norm=color_norm,
            map_extent=map_extent,
            graticule_segments=graticule_segments,
            landsat_raster=landsat_raster,
            landsat_extent=landsat_extent,
        )

    figure.canvas.draw()

    map_positions = [
        axis.get_position()
        for axis in map_axes
    ]

    map_left = min(
        position.x0
        for position in map_positions
    )

    map_right = max(
        position.x1
        for position in map_positions
    )

    map_bottom = min(
        position.y0
        for position in map_positions
    )

    colorbar_axis = figure.add_axes([
        map_left,
        map_bottom - 0.045,
        map_right - map_left,
        0.018,
    ])

    scalar_mappable = ScalarMappable(
        norm=color_norm,
        cmap=CMAP_FRACTION,
    )

    scalar_mappable.set_array([])

    colorbar = figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
        extend="max",
        extendfrac=0.025,
        ticks=colorbar_ticks,
        spacing="uniform",
    )

    colorbar.set_label(
        (
            "CMIP6 ensemble-median expected "
            "visible ponding (%)"
        ),
        fontsize=10,
        labelpad=4,
    )

    colorbar.ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_tick)
    )

    colorbar.ax.tick_params(
        axis="x",
        labelsize=8.5,
        direction="out",
        length=3,
        width=0.7,
        pad=2,
    )

    colorbar.outline.set_linewidth(0.7)

    crossing_years = (
        plot_projection_time_series_with_inset(
            time_axis,
            series,
        )
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=OUTLINE_STYLES["possible"]["color"],
            linewidth=1.2,
            label=(
                "Possible: CMIP6 maximum ≥ "
                f"{THRESHOLD_LOW_PERCENT:.2f}%"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=OUTLINE_STYLES["central"]["color"],
            linewidth=1.6,
            label=(
                "Central: CMIP6 median ≥ "
                f"{THRESHOLD_CENTRAL_PERCENT:.2f}%"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=OUTLINE_STYLES["robust"]["color"],
            linewidth=2.0,
            label=(
                "Robust: CMIP6 minimum ≥ "
                f"{THRESHOLD_HIGH_PERCENT:.2f}%"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=SCENARIO_COLORS["ssp245"],
            linewidth=2.3,
            label="SSP2-4.5 ensemble median",
        ),
        Patch(
            facecolor=SCENARIO_COLORS["ssp245"],
            edgecolor="none",
            alpha=0.24,
            label="SSP2-4.5 10th–90th percentile",
        ),
        Line2D(
            [0],
            [0],
            color=SCENARIO_COLORS["ssp585"],
            linewidth=2.3,
            label="SSP5-8.5 ensemble median",
        ),
        Patch(
            facecolor=SCENARIO_COLORS["ssp585"],
            edgecolor="none",
            alpha=0.24,
            label="SSP5-8.5 10th–90th percentile",
        ),
        Patch(
            facecolor=THRESHOLD_COLOR,
            edgecolor="none",
            alpha=0.12,
            label=(
                "Empirical threshold interval "
                f"{THRESHOLD_LOW_PERCENT:.2f}–"
                f"{THRESHOLD_HIGH_PERCENT:.2f}%"
            ),
        ),
    ]

    legend = figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=True,
        fontsize=8,
        handlelength=2.5,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.65")
    legend.get_frame().set_alpha(0.97)

    save_figure4(figure)

    print(
        "[SAVED]",
        FIGURE4_PATH,
        f"({FIGURE4_WIDTH_CM:.1f} cm × "
        f"{FIGURE4_HEIGHT_CM:.1f} cm, "
        f"{FIGURE_DPI} dpi)",
    )

    if SAVE_PDF:
        print("[SAVED]", FIGURE4_PDF_PATH)

    print(
        "[INFO] Sustained central-threshold crossing years:",
        crossing_years,
    )

# Main Function

def main():
    if (
        REUSE_PROJECTED_SHELF_YEAR
        and os.path.isfile(
            PROJECTED_SHELF_YEAR_PATH
        )
    ):
        print(
            "[INFO] Reusing completed projection analysis:",
            PROJECTED_SHELF_YEAR_PATH,
        )

        projections = pd.read_parquet(
            PROJECTED_SHELF_YEAR_PATH
        )

    else:
        if (
            REUSE_BIASCORRECTED_DJF
            and os.path.isfile(
                BIASCORRECTED_DJF_PATH
            )
        ):
            print(
                "[INFO] Reusing bias-corrected DJF cache:",
                BIASCORRECTED_DJF_PATH,
            )

            biascorrected_cells = (
                pd.read_parquet(
                    BIASCORRECTED_DJF_PATH
                )
            )

        else:
            monthly_cmip = (
                canonicalize_cmip_dictionary(
                    load_pickle(
                        MONTHLY_CMIP_PATH
                    )
                )
            )

            monthly_era5 = (
                canonicalize_shelf_dictionary(
                    load_pickle(
                        MONTHLY_ERA5_PATH
                    )
                )
            )

            biascorrected_cells = (
                create_biascorrected_djf_cells(
                    monthly_cmip,
                    monthly_era5,
                )
            )

        response_model = (
            load_peninsula_model()
        )

        projections = (
            apply_peninsula_model(
                biascorrected_cells,
                response_model,
            )
        )

    print(
        "[INFO] Recreating Figure 4 from projection results."
    )

    create_figure4(projections)

    print(FIGURE4_PATH)


if __name__ == "__main__":
    main()