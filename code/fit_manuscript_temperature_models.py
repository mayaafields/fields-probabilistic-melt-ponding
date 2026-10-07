#!/usr/bin/env python
# coding: utf-8

"""
Fit the two Bayesian temperature-response models reported Fields et al., (In Review).

Models: 

1. All-Shelves temperature-only Beta-Binomial model

   Training data:
       All shelf data × ERA5-cell × year observations from 2006–2020.

   Model:
       y_i ~ BetaBinomial(
           n_i,
           mu_i * kappa,
           (1 - mu_i) * kappa
       )

       cloglog(mu_i) =
           alpha_all
           + beta_temp_all * temperature_all_z_i

       alpha_all ~ Normal(-6, 3)
       beta_temp_all ~ Normal(0, 1.5)
       kappa ~ Gamma(shape=2, rate=0.1)
       shelves: 47 Antarctic ice shelves
       resolution: 0.25° ERA5 cells clipped to the MEaSURES Antarctic ice shelf
       boundaries
       melt data: Pre-processed and aggregated to the ERA5 clipped cells

2. Antarctic Peninsula temperature-only Binomial sensitivity model

   Training shelves:
       George VI
       Larsen B remnant
       Larsen C
       Larsen D
       Stange

   Model:
       y_i ~ Binomial(n_i, mu_i)

       cloglog(mu_i) =
           alpha_ap
           + beta_temp_ap * temperature_ap_z_i

       alpha_ap ~ Normal(-6, 3)
       beta_temp_ap ~ Normal(0, 1.5)

Observation unit
----------------
Each row is one:

    ice shelf × ERA5 grid cell × DJF year

with:

    y_ponded_pixels = number of visibly ponded 30 m pixels
    n_pixels_30m = number of valid 30 m pixels


    - validates and exports the model-input records;
    - fits the two final temperature-response models;
    - saves posterior traces and parameter summaries;
    - creates trace and rank diagnostics;
    - generates posterior temperature-response curves;
    - generates descriptive temperature-bin summaries;
    - calculates fitted-period calibration;
    - calculates row-level and shelf-year residuals;
    - estimates the temperature at which each posterior curve reaches the
      empirical connected-ponding threshold;
    - writes complete run metadata.

Expected input
--------------
A Parquet, CSV, Feather, or GeoPackage-compatible table containing:

    shelf
    year
    era5_cell_id
    era5_t2m_djf_c
    n_pixels_30m
    y_ponded_pixels

Optional filtering columns:

    effective_overlap_area_m2
    coverage_fraction_of_shelf_cell
    region

Default model run settings: 

    observation period: 2006–2020
    draws per chain: 2,000
    tuning iterations per chain: 2,000
    chains: 4
    target_accept: 0.95

Example:

    MODEL_DATAFRAME=data/processed/observations_ERA5temp_ponding_COMMON_SUPPORT_2006_2020.parquet \
    MODEL_OUT_DIR=results/temperature_models \
    DRAWS=2000 \
    TUNE=2000 \
    CHAINS=4 \
    CORES=4 \
    python fit_manuscript_temperature_models.py
"""


# Limit nested numerical-library threading

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

# Imports

import hashlib
import json
import multiprocessing as mp
import re
import time
import warnings
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore")

# Constants

SHELF_COLUMN = "shelf"
YEAR_COLUMN = "year"
CELL_COLUMN = "era5_cell_id"
TEMPERATURE_COLUMN = "era5_t2m_djf_c"
N_COLUMN = "n_pixels_30m"
Y_COLUMN = "y_ponded_pixels"

OBSERVED_FRACTION_COLUMN = "observed_fraction"

ALL_SHELVES_MODEL = "all_shelves_beta_binomial"
PENINSULA_MODEL = "peninsula_binomial"

MODEL_NAMES = [
    ALL_SHELVES_MODEL,
    PENINSULA_MODEL,
]

MODEL_LABELS = {
    ALL_SHELVES_MODEL: (
        "All-Shelves Beta-Binomial"
    ),
    PENINSULA_MODEL: (
        "Antarctic Peninsula Binomial sensitivity"
    ),
}

MODEL_COLORS = {
    ALL_SHELVES_MODEL: "#D55E00",
    PENINSULA_MODEL: "#6A3D9A",
}

MANUSCRIPT_PENINSULA_SHELVES = {
    "georgevi": "George VI",
    "larsenb": "Larsen B",
    "larsenc": "Larsen C",
    "larsend": "Larsen D",
    "stange": "Stange",
}

DEFAULT_START_YEAR = 2006
DEFAULT_END_YEAR = 2020

DEFAULT_DRAWS = 2000
DEFAULT_TUNE = 2000
DEFAULT_CHAINS = 4
DEFAULT_TARGET_ACCEPT = 0.95

ALPHA_PRIOR_MEAN = -6.0
ALPHA_PRIOR_SD = 3.0

BETA_TEMP_PRIOR_MEAN = 0.0
BETA_TEMP_PRIOR_SD = 1.5

DEFAULT_KAPPA_ALPHA = 2.0
DEFAULT_KAPPA_BETA = 0.1

DEFAULT_THRESHOLD_CENTRAL = 0.008718
DEFAULT_THRESHOLD_LOWER_68 = 0.0072
DEFAULT_THRESHOLD_UPPER_68 = 0.0126

PROBABILITY_EPSILON = 1.0e-10

# Universal Plot style. Can be adjusted for preferences.

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# General function helpers

def env_bool(name, default="0"):
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


def normalize_shelf_name(value):
    """Normalize shelf names for stable matching."""

    if pd.isna(value):
        return None

    value = str(value).lower().strip()

    value = re.sub(
        r'<math><mrow><msup><mo form="prefix" stretchy="false">[</mo><mo form="postfix" stretchy="false">)</mo></msup><mo form="postfix" stretchy="false">]</mo><mo>∗</mo></mrow></math>',
        "",
        value,
    )

    for phrase in [
        "ice shelves",
        "ice shelf",
        "iceshelf",
        "shelves",
        "shelf",
        "remnant",
        "glacier",
    ]:
        value = value.replace(
            phrase,
            " ",
        )

    value = "".join(
        character
        for character in value
        if character.isalnum()
    )

    return value

def require_file(path, label):
    """Require one input file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} was not found:\n{path}"
        )


def require_columns(
    dataframe,
    columns,
    label,
):
    """Require table columns."""

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{label} is missing columns: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def read_table(path):
    """Read a supported tabular input format."""

    path = Path(path)

    require_file(
        path,
        "Model input table",
    )

    suffix = path.suffix.lower()

    if suffix in {
        ".parquet",
        ".pq",
    }:
        return pd.read_parquet(path)

    if suffix in {
        ".csv",
        ".txt",
    }:
        return pd.read_csv(
            path,
            low_memory=False,
        )

    if suffix in {
        ".feather",
        ".arrow",
    }:
        return pd.read_feather(path)

    if suffix == ".gpkg":
        import geopandas as gpd

        layer = os.environ.get(
            "MODEL_DATA_LAYER",
            "",
        ).strip()

        if not layer:
            raise ValueError(
                "MODEL_DATA_LAYER must be set when "
                "MODEL_DATAFRAME is a GeoPackage."
            )

        return pd.DataFrame(
            gpd.read_file(
                path,
                layer=layer,
                ignore_geometry=True,
            )
        )

    raise ValueError(
        f"Unsupported input format: {suffix}"
    )


def file_sha256(
    path,
    block_size=1024 * 1024,
):
    """Calculate the SHA-256 checksum of an input file."""

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

            digest.update(block)

    return digest.hexdigest()


def atomic_csv_write(
    dataframe,
    path,
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

    temporary.replace(path)

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

    temporary.replace(path)

    print(
        "[SAVED]",
        path,
        flush=True,
    )


def save_figure(
    figure,
    basename,
    dpi=300,
):
    """Save PNG and PDF."""

    basename = Path(basename)

    basename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = Path(
        str(basename)
        + ".png"
    )

    pdf_path = Path(
        str(basename)
        + ".pdf"
    )

    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    print(
        "[SAVED]",
        png_path,
        flush=True,
    )

    print(
        "[SAVED]",
        pdf_path,
        flush=True,
    )


def save_idata_safely(
    idata,
    path,
    overwrite=False,
):
    """Save InferenceData without silently replacing an existing trace."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.suffix.lower() != ".nc":
        path = path.with_suffix(".nc")

    if path.exists():
        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        if overwrite:
            backup = path.with_name(
                f"{path.stem}_backup_"
                f"{timestamp}{path.suffix}"
            )

            path.replace(backup)

            print(
                "[BACKUP]",
                backup,
                flush=True,
            )

        else:
            path = path.with_name(
                f"{path.stem}_"
                f"{timestamp}{path.suffix}"
            )

    temporary = path.with_name(
        f"{path.stem}_tmp_"
        f"{os.getpid()}{path.suffix}"
    )

    az.to_netcdf(
        idata,
        temporary,
    )

    temporary.replace(path)

    print(
        "[SAVED]",
        path,
        flush=True,
    )

    return path


def flatten_posterior(
    idata,
    variable,
):
    """Flatten posterior chain and draw dimensions."""

    if variable not in idata.posterior:
        raise KeyError(
            f"{variable!r} was not found in the posterior."
        )

    values = idata.posterior[
        variable
    ].values

    if values.ndim == 2:
        return values.reshape(-1)

    return values.reshape(
        (-1,) + values.shape[2:]
    )


def thin_draw_indices(
    number_of_draws,
    maximum_draws,
    seed,
):
    """Select posterior draws reproducibly."""

    if (
        maximum_draws <= 0
        or number_of_draws <= maximum_draws
    ):
        return np.arange(
            number_of_draws
        )

    generator = np.random.default_rng(
        seed
    )

    return np.sort(
        generator.choice(
            number_of_draws,
            size=maximum_draws,
            replace=False,
        )
    )


def inv_cloglog_pm(eta):
    """PyTensor inverse complementary-log-log link."""

    eta = pm.math.clip(
        eta,
        -30.0,
        20.0,
    )

    return pm.math.clip(
        1.0
        - pm.math.exp(
            -pm.math.exp(eta)
        ),
        PROBABILITY_EPSILON,
        1.0
        - PROBABILITY_EPSILON,
    )


def inv_cloglog_np(eta):
    """NumPy inverse complementary-log-log link."""

    eta = np.clip(
        np.asarray(
            eta,
            dtype=float,
        ),
        -30.0,
        20.0,
    )

    return np.clip(
        -np.expm1(
            -np.exp(eta)
        ),
        PROBABILITY_EPSILON,
        1.0
        - PROBABILITY_EPSILON,
    )


def style_axis(axis):
    """Apply common Cartesian styling."""

    axis.spines[
        "top"
    ].set_visible(False)

    axis.spines[
        "right"
    ].set_visible(False)

    axis.grid(
        True,
        color="0.87",
        linewidth=0.7,
        alpha=0.8,
        zorder=0,
    )


# =============================================================================
# 5. Data preparation
# =============================================================================

def validate_and_prepare_data(
    raw,
    *,
    start_year,
    end_year,
    minimum_coverage_fraction,
):
    """Validate and prepare common-support row-level observations."""

    required = [
        SHELF_COLUMN,
        YEAR_COLUMN,
        CELL_COLUMN,
        TEMPERATURE_COLUMN,
        N_COLUMN,
        Y_COLUMN,
    ]

    require_columns(
        raw,
        required,
        "Model input table",
    )

    data = raw.copy()

    numeric_columns = [
        YEAR_COLUMN,
        CELL_COLUMN,
        TEMPERATURE_COLUMN,
        N_COLUMN,
        Y_COLUMN,
    ]

    optional_numeric_columns = [
        "effective_overlap_area_m2",
        "coverage_fraction_of_shelf_cell",
    ]

    for column in (
        numeric_columns
        + [
            column
            for column
            in optional_numeric_columns
            if column in data.columns
        ]
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    keep = (
        data[SHELF_COLUMN].notna()
        & np.isfinite(
            data[YEAR_COLUMN]
        )
        & data[YEAR_COLUMN].between(
            start_year,
            end_year,
            inclusive="both",
        )
        & np.isfinite(
            data[CELL_COLUMN]
        )
        & np.isfinite(
            data[TEMPERATURE_COLUMN]
        )
        & np.isfinite(
            data[N_COLUMN]
        )
        & (
            data[N_COLUMN]
            > 0
        )
        & np.isfinite(
            data[Y_COLUMN]
        )
        & (
            data[Y_COLUMN]
            >= 0
        )
        & (
            data[Y_COLUMN]
            <= data[N_COLUMN]
        )
    )

    if (
        "effective_overlap_area_m2"
        in data.columns
    ):
        keep &= (
            np.isfinite(
                data[
                    "effective_overlap_area_m2"
                ]
            )
            & (
                data[
                    "effective_overlap_area_m2"
                ]
                > 0
            )
        )

    if (
        "coverage_fraction_of_shelf_cell"
        in data.columns
    ):
        keep &= (
            np.isfinite(
                data[
                    "coverage_fraction_of_shelf_cell"
                ]
            )
            & (
                data[
                    "coverage_fraction_of_shelf_cell"
                ]
                >= minimum_coverage_fraction
            )
        )

    data = (
        data.loc[keep]
        .copy()
        .reset_index(drop=True)
    )

    if data.empty:
        raise RuntimeError(
            "No valid observations remain."
        )

    data[YEAR_COLUMN] = (
        data[YEAR_COLUMN]
        .round()
        .astype(int)
    )

    data[CELL_COLUMN] = (
        data[CELL_COLUMN]
        .round()
        .astype("int64")
    )

    data[N_COLUMN] = (
        data[N_COLUMN]
        .round()
        .astype("int64")
    )

    data[Y_COLUMN] = (
        data[Y_COLUMN]
        .round()
        .astype("int64")
    )

    duplicate = data.duplicated(
        [
            SHELF_COLUMN,
            CELL_COLUMN,
            YEAR_COLUMN,
        ],
        keep=False,
    )

    if duplicate.any():
        duplicate_rows = data.loc[
            duplicate,
            [
                SHELF_COLUMN,
                CELL_COLUMN,
                YEAR_COLUMN,
            ],
        ]

        raise RuntimeError(
            "Duplicate shelf-cell-year rows exist. "
            f"First examples:\n"
            f"{duplicate_rows.head(20).to_string(index=False)}"
        )

    data["shelf_normalized"] = (
        data[SHELF_COLUMN]
        .map(
            normalize_shelf_name
        )
    )

    data[OBSERVED_FRACTION_COLUMN] = (
        data[Y_COLUMN]
        / data[N_COLUMN]
    )

    return data


def standardize_temperature(
    data,
    temperature_column,
):
    """Return a copy with a standardized temperature column."""

    output = data.copy()

    mean = float(
        output[
            temperature_column
        ].mean()
    )

    standard_deviation = float(
        output[
            temperature_column
        ].std(
            ddof=0
        )
    )

    if not np.isfinite(mean):
        raise ValueError(
            "Temperature mean is not finite."
        )

    if (
        not np.isfinite(
            standard_deviation
        )
        or standard_deviation <= 0
    ):
        raise ValueError(
            "Temperature standard deviation "
            "must be positive."
        )

    output["temperature_z"] = (
        output[
            temperature_column
        ]
        - mean
    ) / standard_deviation

    return (
        output,
        mean,
        standard_deviation,
    )


def select_peninsula_training_data(
    all_shelves_data,
):
    """Select exactly the five Peninsula shelves reported in the paper."""

    peninsula = all_shelves_data[
        all_shelves_data[
            "shelf_normalized"
        ].isin(
            MANUSCRIPT_PENINSULA_SHELVES
        )
    ].copy()

    found = set(
        peninsula[
            "shelf_normalized"
        ].dropna().unique()
    )

    missing = (
        set(
            MANUSCRIPT_PENINSULA_SHELVES
        )
        - found
    )

    if missing:
        raise RuntimeError(
            "The Peninsula training data are missing: "
            f"{sorted(missing)}"
        )

    unexpected = (
        found
        - set(
            MANUSCRIPT_PENINSULA_SHELVES
        )
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected Peninsula shelves were selected: "
            f"{sorted(unexpected)}"
        )

    return peninsula.reset_index(
        drop=True
    )


def make_temperature_bins(
    data,
    temperature_column,
    *,
    number_of_bins,
):
    """Create quantile bins for descriptive plotting."""

    source = data.copy()

    number_of_bins = min(
        number_of_bins,
        source[
            temperature_column
        ].nunique(),
        len(source),
    )

    if number_of_bins < 2:
        raise ValueError(
            "At least two temperature bins are required."
        )

    source["_temperature_bin"] = pd.qcut(
        source[
            temperature_column
        ],
        q=number_of_bins,
        duplicates="drop",
    )

    source["_bin_code"] = (
        source[
            "_temperature_bin"
        ].cat.codes
    )

    source = source[
        source["_bin_code"]
        >= 0
    ].copy()

    rows = []

    for bin_code, group in source.groupby(
        "_bin_code",
        observed=True,
        sort=True,
    ):
        valid_pixels = float(
            group[N_COLUMN].sum()
        )

        ponded_pixels = float(
            group[Y_COLUMN].sum()
        )

        row_fraction = (
            group[Y_COLUMN]
            / group[N_COLUMN]
        )

        shelf_pixel_totals = (
            group.groupby(
                SHELF_COLUMN,
                observed=True,
            )[N_COLUMN]
            .sum()
            .to_numpy(
                dtype=float
            )
        )

        shelf_weights = (
            shelf_pixel_totals
            / shelf_pixel_totals.sum()
        )

        effective_shelves = (
            1.0
            / np.sum(
                shelf_weights ** 2
            )
        )

        rows.append(
            {
                "bin_index": int(
                    bin_code
                ),
                "bin_label": str(
                    group[
                        "_temperature_bin"
                    ].iloc[0]
                ),
                "temperature_mean_pixel_weighted": float(
                    np.average(
                        group[
                            temperature_column
                        ],
                        weights=group[
                            N_COLUMN
                        ],
                    )
                ),
                "temperature_mean_equal_row": float(
                    group[
                        temperature_column
                    ].mean()
                ),
                "temperature_minimum": float(
                    group[
                        temperature_column
                    ].min()
                ),
                "temperature_maximum": float(
                    group[
                        temperature_column
                    ].max()
                ),
                "n_rows": int(
                    len(group)
                ),
                "n_shelves": int(
                    group[
                        SHELF_COLUMN
                    ].nunique()
                ),
                "effective_shelves_pixel_weighted": float(
                    effective_shelves
                ),
                "valid_pixels": int(
                    round(
                        valid_pixels
                    )
                ),
                "ponded_pixels": int(
                    round(
                        ponded_pixels
                    )
                ),
                "observed_pixel_pooled_fraction": (
                    ponded_pixels
                    / valid_pixels
                ),
                "observed_equal_row_mean_fraction": float(
                    row_fraction.mean()
                ),
                "positive_observation_fraction": float(
                    (
                        group[Y_COLUMN]
                        > 0
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "temperature_mean_pixel_weighted"
    ).reset_index(
        drop=True
    )


def data_summary(
    data,
    *,
    model_name,
    temperature_mean,
    temperature_sd,
):
    """Create one model-training sample summary."""

    return {
        "model_name": model_name,
        "n_rows": int(
            len(data)
        ),
        "n_shelves": int(
            data[
                SHELF_COLUMN
            ].nunique()
        ),
        "n_shelf_cells": int(
            data[
                [
                    SHELF_COLUMN,
                    CELL_COLUMN,
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "n_years": int(
            data[
                YEAR_COLUMN
            ].nunique()
        ),
        "year_minimum": int(
            data[
                YEAR_COLUMN
            ].min()
        ),
        "year_maximum": int(
            data[
                YEAR_COLUMN
            ].max()
        ),
        "zero_rows": int(
            (
                data[Y_COLUMN]
                == 0
            ).sum()
        ),
        "positive_rows": int(
            (
                data[Y_COLUMN]
                > 0
            ).sum()
        ),
        "positive_row_fraction": float(
            (
                data[Y_COLUMN]
                > 0
            ).mean()
        ),
        "total_valid_pixels": int(
            data[
                N_COLUMN
            ].sum()
        ),
        "total_ponded_pixels": int(
            data[
                Y_COLUMN
            ].sum()
        ),
        "observed_pixel_pooled_fraction": float(
            data[
                Y_COLUMN
            ].sum()
            / data[
                N_COLUMN
            ].sum()
        ),
        "temperature_minimum_c": float(
            data[
                TEMPERATURE_COLUMN
            ].min()
        ),
        "temperature_maximum_c": float(
            data[
                TEMPERATURE_COLUMN
            ].max()
        ),
        "temperature_mean_c": (
            temperature_mean
        ),
        "temperature_sd_c": (
            temperature_sd
        ),
        "shelves": sorted(
            data[
                SHELF_COLUMN
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
    }


# =============================================================================
# 6. Model builders and sampling
# =============================================================================

def add_temperature_response(
    temperature_z,
    *,
    dimension_name,
    parameter_suffix,
):
    """Create priors and a complementary-log-log response."""

    temperature_data = pm.Data(
        "temperature_z",
        np.asarray(
            temperature_z,
            dtype=float,
        ),
        dims=dimension_name,
    )

    alpha = pm.Normal(
        f"alpha_{parameter_suffix}",
        mu=ALPHA_PRIOR_MEAN,
        sigma=ALPHA_PRIOR_SD,
    )

    beta_temp = pm.Normal(
        f"beta_temp_{parameter_suffix}",
        mu=BETA_TEMP_PRIOR_MEAN,
        sigma=BETA_TEMP_PRIOR_SD,
    )

    eta = (
        alpha
        + beta_temp
        * temperature_data
    )

    probability = pm.Deterministic(
        "probability",
        inv_cloglog_pm(
            eta
        ),
        dims=dimension_name,
    )

    return probability


def build_all_shelves_beta_binomial(
    data,
    *,
    kappa_alpha,
    kappa_beta,
):
    """Build the manuscript All-Shelves Beta-Binomial model."""

    coordinates = {
        "obs": np.arange(
            len(data)
        ),
    }

    with pm.Model(
        coords=coordinates
    ) as model:
        n_data = pm.Data(
            "n",
            data[
                N_COLUMN
            ].to_numpy(
                dtype="int64"
            ),
            dims="obs",
        )

        probability = (
            add_temperature_response(
                data[
                    "temperature_z"
                ].to_numpy(
                    dtype=float
                ),
                dimension_name="obs",
                parameter_suffix="all",
            )
        )

        kappa = pm.Gamma(
            "kappa",
            alpha=kappa_alpha,
            beta=kappa_beta,
        )

        pm.Deterministic(
            "rho",
            1.0
            / (
                1.0
                + kappa
            ),
        )

        pm.BetaBinomial(
            "y",
            n=n_data,
            alpha=(
                probability
                * kappa
            ),
            beta=(
                (
                    1.0
                    - probability
                )
                * kappa
            ),
            observed=data[
                Y_COLUMN
            ].to_numpy(
                dtype="int64"
            ),
            dims="obs",
        )

    return model


def build_peninsula_binomial(
    data,
):
    """Build the manuscript Peninsula Binomial sensitivity model."""

    coordinates = {
        "obs": np.arange(
            len(data)
        ),
    }

    with pm.Model(
        coords=coordinates
    ) as model:
        n_data = pm.Data(
            "n",
            data[
                N_COLUMN
            ].to_numpy(
                dtype="int64"
            ),
            dims="obs",
        )

        probability = (
            add_temperature_response(
                data[
                    "temperature_z"
                ].to_numpy(
                    dtype=float
                ),
                dimension_name="obs",
                parameter_suffix="ap",
            )
        )

        pm.Binomial(
            "y",
            n=n_data,
            p=probability,
            observed=data[
                Y_COLUMN
            ].to_numpy(
                dtype="int64"
            ),
            dims="obs",
        )

    return model


def sample_model(
    model,
    *,
    draws,
    tune,
    chains,
    cores,
    target_accept,
    maximum_tree_depth,
    random_seed,
    compute_log_likelihood,
):
    """Sample one model with the No-U-Turn Sampler."""

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            nuts={
                "max_treedepth": (
                    maximum_tree_depth
                ),
            },
            idata_kwargs={
                "log_likelihood": (
                    compute_log_likelihood
                ),
            },
        )

    return idata


# =============================================================================
# 7. Posterior calculations
# =============================================================================

def model_parameter_names(
    model_name,
):
    """Return model parameters for summaries."""

    if model_name == ALL_SHELVES_MODEL:
        return [
            "alpha_all",
            "beta_temp_all",
            "kappa",
            "rho",
        ]

    if model_name == PENINSULA_MODEL:
        return [
            "alpha_ap",
            "beta_temp_ap",
        ]

    raise ValueError(
        model_name
    )


def response_parameter_names(
    model_name,
):
    """Return response coefficient names."""

    if model_name == ALL_SHELVES_MODEL:
        return (
            "alpha_all",
            "beta_temp_all",
        )

    if model_name == PENINSULA_MODEL:
        return (
            "alpha_ap",
            "beta_temp_ap",
        )

    raise ValueError(
        model_name
    )


def posterior_probability_draws(
    idata,
    model_name,
    temperature_c,
    *,
    temperature_mean,
    temperature_sd,
    maximum_draws,
    seed,
):
    """Calculate posterior expected ponding probabilities."""

    alpha_name, beta_name = (
        response_parameter_names(
            model_name
        )
    )

    alpha = flatten_posterior(
        idata,
        alpha_name,
    )

    beta_temp = flatten_posterior(
        idata,
        beta_name,
    )

    indices = thin_draw_indices(
        len(alpha),
        maximum_draws,
        seed,
    )

    temperature_z = (
        np.asarray(
            temperature_c,
            dtype=float,
        )
        - temperature_mean
    ) / temperature_sd

    eta = (
        alpha[
            indices,
            None,
        ]
        + beta_temp[
            indices,
            None,
        ]
        * temperature_z[
            None,
            :,
        ]
    )

    return (
        inv_cloglog_np(
            eta
        ),
        indices,
    )


def posterior_curve(
    idata,
    model_name,
    temperature_grid,
    *,
    temperature_mean,
    temperature_sd,
    maximum_draws,
    seed,
):
    """Summarize one posterior response curve."""

    probability, _ = (
        posterior_probability_draws(
            idata,
            model_name,
            temperature_grid,
            temperature_mean=(
                temperature_mean
            ),
            temperature_sd=(
                temperature_sd
            ),
            maximum_draws=(
                maximum_draws
            ),
            seed=seed,
        )
    )

    output = pd.DataFrame(
        {
            "model_name": model_name,
            "model_label": (
                MODEL_LABELS[
                    model_name
                ]
            ),
            "temperature_c": (
                temperature_grid
            ),
            "expected_mean": (
                np.mean(
                    probability,
                    axis=0,
                )
            ),
            "expected_q025": (
                np.quantile(
                    probability,
                    0.025,
                    axis=0,
                )
            ),
            "expected_median": (
                np.median(
                    probability,
                    axis=0,
                )
            ),
            "expected_q975": (
                np.quantile(
                    probability,
                    0.975,
                    axis=0,
                )
            ),
        }
    )

    for column in [
        "expected_mean",
        "expected_q025",
        "expected_median",
        "expected_q975",
    ]:
        output[
            f"{column}_percent"
        ] = (
            100.0
            * output[column]
        )

    return output


def posterior_predictions_at_rows(
    idata,
    model_name,
    data,
    *,
    temperature_mean,
    temperature_sd,
    maximum_draws,
    chunk_size,
    seed,
):
    """Calculate row-level expected-probability summaries in chunks."""

    alpha_name, beta_name = (
        response_parameter_names(
            model_name
        )
    )

    alpha = flatten_posterior(
        idata,
        alpha_name,
    )

    beta_temp = flatten_posterior(
        idata,
        beta_name,
    )

    indices = thin_draw_indices(
        len(alpha),
        maximum_draws,
        seed,
    )

    alpha = alpha[
        indices
    ]

    beta_temp = beta_temp[
        indices
    ]

    temperature = data[
        TEMPERATURE_COLUMN
    ].to_numpy(
        dtype=float
    )

    means = np.empty(
        len(data),
        dtype=float,
    )

    medians = np.empty(
        len(data),
        dtype=float,
    )

    lower = np.empty(
        len(data),
        dtype=float,
    )

    upper = np.empty(
        len(data),
        dtype=float,
    )

    for start in range(
        0,
        len(data),
        chunk_size,
    ):
        stop = min(
            start
            + chunk_size,
            len(data),
        )

        temperature_z = (
            temperature[
                start:stop
            ]
            - temperature_mean
        ) / temperature_sd

        eta = (
            alpha[
                :,
                None,
            ]
            + beta_temp[
                :,
                None,
            ]
            * temperature_z[
                None,
                :,
            ]
        )

        probability = inv_cloglog_np(
            eta
        )

        means[
            start:stop
        ] = np.mean(
            probability,
            axis=0,
        )

        medians[
            start:stop
        ] = np.median(
            probability,
            axis=0,
        )

        lower[
            start:stop
        ] = np.quantile(
            probability,
            0.025,
            axis=0,
        )

        upper[
            start:stop
        ] = np.quantile(
            probability,
            0.975,
            axis=0,
        )

    output = data.copy()

    output["predicted_mean"] = means
    output["predicted_median"] = medians
    output["predicted_q025"] = lower
    output["predicted_q975"] = upper

    # SI Figures S15 and S16 define residual as observed minus predicted.
    output[
        "residual_observed_minus_predicted"
    ] = (
        output[
            OBSERVED_FRACTION_COLUMN
        ]
        - output[
            "predicted_mean"
        ]
    )

    return output


def summarize_threshold_crossing(
    curve,
    *,
    central_threshold,
    lower_threshold,
    upper_threshold,
):
    """Estimate model-implied threshold temperatures from saved curves."""

    rows = []

    thresholds = [
        (
            "lower_68",
            lower_threshold,
        ),
        (
            "central",
            central_threshold,
        ),
        (
            "upper_68",
            upper_threshold,
        ),
    ]

    for (
        threshold_name,
        threshold_value,
    ) in thresholds:
        for curve_column in [
            "expected_q025",
            "expected_median",
            "expected_q975",
        ]:
            values = curve[
                curve_column
            ].to_numpy(
                dtype=float
            )

            temperature = curve[
                "temperature_c"
            ].to_numpy(
                dtype=float
            )

            crossing = np.where(
                values
                >= threshold_value
            )[0]

            if crossing.size == 0:
                crossing_temperature = np.nan
            else:
                crossing_temperature = float(
                    temperature[
                        crossing[0]
                    ]
                )

            rows.append(
                {
                    "model_name": curve[
                        "model_name"
                    ].iloc[0],
                    "threshold_name": (
                        threshold_name
                    ),
                    "threshold_fraction": (
                        threshold_value
                    ),
                    "curve_statistic": (
                        curve_column
                    ),
                    "first_crossing_temperature_c": (
                        crossing_temperature
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 8. Calibration and aggregation
# =============================================================================

def make_prediction_calibration(
    predictions,
    *,
    number_of_bins,
):
    """Group records into bins of similar fitted probability."""

    source = predictions.copy()

    source["_risk_bin"] = pd.qcut(
        source[
            "predicted_mean"
        ],
        q=min(
            number_of_bins,
            source[
                "predicted_mean"
            ].nunique(),
            len(source),
        ),
        duplicates="drop",
    )

    source["_risk_code"] = (
        source[
            "_risk_bin"
        ].cat.codes
    )

    source = source[
        source["_risk_code"]
        >= 0
    ].copy()

    rows = []

    for risk_code, group in source.groupby(
        "_risk_code",
        observed=True,
        sort=True,
    ):
        valid_pixels = float(
            group[
                N_COLUMN
            ].sum()
        )

        observed_fraction = float(
            group[
                Y_COLUMN
            ].sum()
            / valid_pixels
        )

        predicted_fraction = float(
            np.average(
                group[
                    "predicted_mean"
                ],
                weights=group[
                    N_COLUMN
                ],
            )
        )

        rows.append(
            {
                "risk_bin": int(
                    risk_code
                ),
                "n_rows": int(
                    len(group)
                ),
                "valid_pixels": int(
                    round(
                        valid_pixels
                    )
                ),
                "observed_fraction": (
                    observed_fraction
                ),
                "predicted_fraction": (
                    predicted_fraction
                ),
                "residual_observed_minus_predicted": (
                    observed_fraction
                    - predicted_fraction
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def aggregate_shelf_year_residuals(
    predictions,
):
    """Pixel-pool observations and predictions by shelf and year."""

    source = predictions.copy()

    source[
        "expected_ponded_pixels"
    ] = (
        source[
            "predicted_mean"
        ]
        * source[
            N_COLUMN
        ]
    )

    output = (
        source.groupby(
            [
                SHELF_COLUMN,
                YEAR_COLUMN,
            ],
            observed=True,
        )
        .agg(
            valid_pixels=(
                N_COLUMN,
                "sum",
            ),
            observed_ponded_pixels=(
                Y_COLUMN,
                "sum",
            ),
            expected_ponded_pixels=(
                "expected_ponded_pixels",
                "sum",
            ),
            temperature_mean_c=(
                TEMPERATURE_COLUMN,
                "mean",
            ),
        )
        .reset_index()
    )

    output[
        "observed_fraction"
    ] = (
        output[
            "observed_ponded_pixels"
        ]
        / output[
            "valid_pixels"
        ]
    )

    output[
        "predicted_fraction"
    ] = (
        output[
            "expected_ponded_pixels"
        ]
        / output[
            "valid_pixels"
        ]
    )

    output[
        "residual_observed_minus_predicted"
    ] = (
        output[
            "observed_fraction"
        ]
        - output[
            "predicted_fraction"
        ]
    )

    return output


def aggregate_shelf_residuals(
    predictions,
):
    """Pixel-pool observations and predictions over years by shelf."""

    source = predictions.copy()

    source[
        "expected_ponded_pixels"
    ] = (
        source[
            "predicted_mean"
        ]
        * source[
            N_COLUMN
        ]
    )

    output = (
        source.groupby(
            SHELF_COLUMN,
            observed=True,
        )
        .agg(
            n_rows=(
                Y_COLUMN,
                "size",
            ),
            n_years=(
                YEAR_COLUMN,
                "nunique",
            ),
            valid_pixels=(
                N_COLUMN,
                "sum",
            ),
            observed_ponded_pixels=(
                Y_COLUMN,
                "sum",
            ),
            expected_ponded_pixels=(
                "expected_ponded_pixels",
                "sum",
            ),
        )
        .reset_index()
    )

    output[
        "observed_fraction"
    ] = (
        output[
            "observed_ponded_pixels"
        ]
        / output[
            "valid_pixels"
        ]
    )

    output[
        "predicted_fraction"
    ] = (
        output[
            "expected_ponded_pixels"
        ]
        / output[
            "valid_pixels"
        ]
    )

    output[
        "residual_observed_minus_predicted"
    ] = (
        output[
            "observed_fraction"
        ]
        - output[
            "predicted_fraction"
        ]
    )

    return output.sort_values(
        "residual_observed_minus_predicted",
        ascending=False,
    ).reset_index(
        drop=True
    )


def row_fit_metrics(
    predictions,
):
    """Calculate fitted-period row-level and pixel-weighted diagnostics."""

    observed = predictions[
        OBSERVED_FRACTION_COLUMN
    ].to_numpy(
        dtype=float
    )

    predicted = predictions[
        "predicted_mean"
    ].to_numpy(
        dtype=float
    )

    weights = predictions[
        N_COLUMN
    ].to_numpy(
        dtype=float
    )

    weights = (
        weights
        / weights.sum()
    )

    residual = (
        predicted
        - observed
    )

    return {
        "residual_definition": (
            "predicted_minus_observed"
        ),
        "row_rmse": float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        ),
        "row_mae": float(
            np.mean(
                np.abs(
                    residual
                )
            )
        ),
        "row_bias_predicted_minus_observed": float(
            np.mean(
                residual
            )
        ),
        "pixel_weighted_rmse": float(
            np.sqrt(
                np.sum(
                    weights
                    * residual ** 2
                )
            )
        ),
        "pixel_weighted_mae": float(
            np.sum(
                weights
                * np.abs(
                    residual
                )
            )
        ),
        "pixel_weighted_bias_predicted_minus_observed": float(
            np.sum(
                weights
                * residual
            )
        ),
        "observed_pixel_pooled_fraction": float(
            np.sum(
                weights
                * observed
            )
        ),
        "predicted_pixel_pooled_fraction": float(
            np.sum(
                weights
                * predicted
            )
        ),
    }


# =============================================================================
# 9. Plotting
# =============================================================================

def plot_trace_and_rank(
    idata,
    model_name,
    output_directory,
):
    """Create trace and rank diagnostics."""

    variables = model_parameter_names(
        model_name
    )

    trace_axes = az.plot_trace(
        idata,
        var_names=variables,
        compact=True,
    )

    trace_figure = np.asarray(
        trace_axes
    ).ravel()[0].figure

    trace_figure.suptitle(
        MODEL_LABELS[
            model_name
        ],
        fontsize=13,
    )

    trace_figure.tight_layout()

    save_figure(
        trace_figure,
        output_directory
        / f"{model_name}_trace",
    )

    rank_axes = az.plot_rank(
        idata,
        var_names=variables,
        kind="bars",
    )

    rank_figure = np.asarray(
        rank_axes
    ).ravel()[0].figure

    rank_figure.suptitle(
        MODEL_LABELS[
            model_name
        ],
        fontsize=13,
    )

    rank_figure.tight_layout()

    save_figure(
        rank_figure,
        output_directory
        / f"{model_name}_rank",
    )


def plot_temperature_distribution(
    all_shelves_data,
    bins,
    output_directory,
):
    """Plot the temperature distribution and empirical support."""

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(
            10,
            8,
        ),
        constrained_layout=True,
    )

    histogram_edges = np.arange(
        np.floor(
            all_shelves_data[
                TEMPERATURE_COLUMN
            ].min()
            * 2
        )
        / 2,
        np.ceil(
            all_shelves_data[
                TEMPERATURE_COLUMN
            ].max()
            * 2
        )
        / 2
        + 0.5,
        0.5,
    )

    zero = all_shelves_data[
        all_shelves_data[
            Y_COLUMN
        ]
        == 0
    ]

    positive = all_shelves_data[
        all_shelves_data[
            Y_COLUMN
        ]
        > 0
    ]

    axes[0].hist(
        [
            zero[
                TEMPERATURE_COLUMN
            ],
            positive[
                TEMPERATURE_COLUMN
            ],
        ],
        bins=histogram_edges,
        stacked=True,
        color=[
            "0.80",
            "#E69F00",
        ],
        edgecolor="0.45",
        linewidth=0.4,
        label=[
            "Exactly zero visible ponding",
            "Positive visible ponding",
        ],
    )

    axes[0].set_ylabel(
        "Shelf-cell-year observations"
    )

    axes[0].set_title(
        "Temperature distribution of visible-ponding observations"
    )

    axes[0].legend(
        frameon=False,
    )

    axis_support = axes[1].twinx()

    axes[1].plot(
        bins[
            "temperature_mean_pixel_weighted"
        ],
        100.0
        * bins[
            "observed_pixel_pooled_fraction"
        ],
        color="#0072B2",
        marker="o",
        label=(
            "Pixel-pooled ponded fraction"
        ),
    )

    axes[1].bar(
        bins[
            "temperature_mean_pixel_weighted"
        ],
        100.0
        * bins[
            "positive_observation_fraction"
        ],
        width=0.35,
        color="#D55E00",
        alpha=0.75,
        label=(
            "Observations with visible ponding"
        ),
    )

    axis_support.plot(
        bins[
            "temperature_mean_pixel_weighted"
        ],
        bins[
            "n_shelves"
        ],
        color="#009E73",
        marker="o",
        label=(
            "Observed shelves"
        ),
    )

    axis_support.plot(
        bins[
            "temperature_mean_pixel_weighted"
        ],
        bins[
            "effective_shelves_pixel_weighted"
        ],
        color="#6A3D9A",
        marker="s",
        linestyle="--",
        label=(
            "Effective shelves"
        ),
    )

    axes[1].set_xlabel(
        "ERA5 DJF 2 m temperature (°C)"
    )

    axes[1].set_ylabel(
        "Visible-ponding statistic (%)"
    )

    axis_support.set_ylabel(
        "Shelf support"
    )

    handles_left, labels_left = (
        axes[1].get_legend_handles_labels()
    )

    handles_right, labels_right = (
        axis_support.get_legend_handles_labels()
    )

    axes[1].legend(
        handles_left
        + handles_right,
        labels_left
        + labels_right,
        frameon=False,
        ncol=2,
        loc="upper left",
    )

    style_axis(
        axes[0]
    )

    style_axis(
        axes[1]
    )

    save_figure(
        figure,
        output_directory
        / "temperature_distribution_and_support",
    )


def plot_response_comparison(
    all_shelves_bins,
    curves,
    output_directory,
    *,
    central_threshold,
    lower_threshold,
    upper_threshold,
):
    """Plot descriptive observations and both posterior response curves."""

    figure, axis = plt.subplots(
        figsize=(
            10,
            6.5,
        ),
        constrained_layout=True,
    )

    point_size = all_shelves_bins[
        "valid_pixels"
    ].to_numpy(
        dtype=float
    )

    point_size = (
        35.0
        + 180.0
        * np.sqrt(
            point_size
            / point_size.max()
        )
    )

    axis.scatter(
        all_shelves_bins[
            "temperature_mean_pixel_weighted"
        ],
        100.0
        * all_shelves_bins[
            "observed_pixel_pooled_fraction"
        ],
        color="black",
        edgecolor="white",
        linewidth=0.7,
        s=point_size,
        zorder=10,
        label=(
            "Binned pixel-pooled observations"
        ),
    )

    axis.scatter(
        all_shelves_bins[
            "temperature_mean_equal_row"
        ],
        100.0
        * all_shelves_bins[
            "observed_equal_row_mean_fraction"
        ],
        color="0.45",
        marker="s",
        s=36,
        zorder=9,
        label=(
            "Equal-row empirical bin mean"
        ),
    )

    for model_name in MODEL_NAMES:
        subset = curves[
            curves[
                "model_name"
            ]
            == model_name
        ].copy()

        color = MODEL_COLORS[
            model_name
        ]

        axis.fill_between(
            subset[
                "temperature_c"
            ],
            subset[
                "expected_q025_percent"
            ],
            subset[
                "expected_q975_percent"
            ],
            color=color,
            alpha=0.16,
            linewidth=0,
        )

        axis.plot(
            subset[
                "temperature_c"
            ],
            subset[
                "expected_median_percent"
            ],
            color=color,
            linewidth=2.5,
            label=MODEL_LABELS[
                model_name
            ],
        )

    axis.axhspan(
        100.0
        * lower_threshold,
        100.0
        * upper_threshold,
        color="#E78AC3",
        alpha=0.12,
        zorder=0,
    )

    axis.axhline(
        100.0
        * central_threshold,
        color="#C51B7D",
        linestyle="-.",
        linewidth=1.4,
        label=(
            "Connected-ponding threshold"
        ),
    )

    axis.set_xlabel(
        "ERA5 DJF 2 m temperature (°C)"
    )

    axis.set_ylabel(
        "Visible ponded fraction (%)"
    )

    axis.set_title(
        "Observed and modeled temperature response"
    )

    axis.set_ylim(
        bottom=0.0
    )

    style_axis(
        axis
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    save_figure(
        figure,
        output_directory
        / "temperature_response_model_comparison",
    )


def plot_calibration(
    calibration_lookup,
    output_directory,
):
    """Plot fitted-period calibration for both models."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            11,
            5,
        ),
        constrained_layout=True,
    )

    for axis, model_name in zip(
        axes,
        MODEL_NAMES,
    ):
        calibration = (
            calibration_lookup[
                model_name
            ]
        )

        maximum = max(
            calibration[
                "observed_fraction"
            ].max(),
            calibration[
                "predicted_fraction"
            ].max(),
            1.0e-6,
        )

        sizes = calibration[
            "valid_pixels"
        ].to_numpy(
            dtype=float
        )

        sizes = (
            35.0
            + 120.0
            * np.sqrt(
                sizes
                / sizes.max()
            )
        )

        axis.scatter(
            calibration[
                "predicted_fraction"
            ],
            calibration[
                "observed_fraction"
            ],
            color=MODEL_COLORS[
                model_name
            ],
            s=sizes,
            alpha=0.85,
        )

        axis.plot(
            [
                0.0,
                maximum,
            ],
            [
                0.0,
                maximum,
            ],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Perfect calibration",
        )

        axis.set_xlim(
            0.0,
            maximum
            * 1.05,
        )

        axis.set_ylim(
            0.0,
            maximum
            * 1.05,
        )

        axis.set_xlabel(
            "Predicted ponded fraction"
        )

        axis.set_ylabel(
            "Observed ponded fraction"
        )

        axis.set_title(
            MODEL_LABELS[
                model_name
            ]
        )

        axis.xaxis.set_major_formatter(
            mticker.PercentFormatter(
                xmax=1.0,
                decimals=1,
            )
        )

        axis.yaxis.set_major_formatter(
            mticker.PercentFormatter(
                xmax=1.0,
                decimals=1,
            )
        )

        style_axis(
            axis
        )

        axis.legend(
            frameon=False,
        )

    save_figure(
        figure,
        output_directory
        / "temperature_model_calibration",
    )


def plot_row_residuals(
    prediction_lookup,
    output_directory,
):
    """Plot residuals against temperature and fitted probability."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            12,
            9,
        ),
        constrained_layout=True,
    )

    for row_index, model_name in enumerate(
        MODEL_NAMES
    ):
        data = prediction_lookup[
            model_name
        ]

        color = MODEL_COLORS[
            model_name
        ]

        residual = data[
            "residual_observed_minus_predicted"
        ]

        axes[
            row_index,
            0
        ].scatter(
            data[
                TEMPERATURE_COLUMN
            ],
            residual,
            color=color,
            alpha=0.07,
            s=6,
            rasterized=True,
        )

        axes[
            row_index,
            1
        ].scatter(
            data[
                "predicted_mean"
            ],
            residual,
            color=color,
            alpha=0.07,
            s=6,
            rasterized=True,
        )

        axes[
            row_index,
            0
        ].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
        )

        axes[
            row_index,
            1
        ].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
        )

        axes[
            row_index,
            0
        ].set_title(
            f"{MODEL_LABELS[model_name]}: residual by temperature"
        )

        axes[
            row_index,
            1
        ].set_title(
            f"{MODEL_LABELS[model_name]}: residual by fitted value"
        )

        axes[
            row_index,
            0
        ].set_xlabel(
            "ERA5 DJF 2 m temperature (°C)"
        )

        axes[
            row_index,
            1
        ].set_xlabel(
            "Predicted ponded fraction"
        )

        axes[
            row_index,
            0
        ].set_ylabel(
            "Observed − predicted fraction"
        )

        axes[
            row_index,
            1
        ].set_ylabel(
            "Observed − predicted fraction"
        )

        axes[
            row_index,
            1
        ].xaxis.set_major_formatter(
            mticker.PercentFormatter(
                xmax=1.0,
                decimals=1,
            )
        )

        for column_index in range(
            2
        ):
            axes[
                row_index,
                column_index
            ].yaxis.set_major_formatter(
                mticker.PercentFormatter(
                    xmax=1.0,
                    decimals=0,
                )
            )

            style_axis(
                axes[
                    row_index,
                    column_index
                ]
            )

    save_figure(
        figure,
        output_directory
        / "temperature_model_row_residuals",
    )


def plot_shelf_year_heatmaps(
    shelf_year_lookup,
    output_directory,
):
    """Plot observed-minus-predicted shelf-year residuals."""

    shelves = sorted(
        set().union(
            *[
                set(
                    table[
                        SHELF_COLUMN
                    ].astype(str)
                )
                for table
                in shelf_year_lookup.values()
            ]
        )
    )

    years = sorted(
        set().union(
            *[
                set(
                    table[
                        YEAR_COLUMN
                    ].astype(int)
                )
                for table
                in shelf_year_lookup.values()
            ]
        )
    )

    matrices = {}

    maximum_absolute = 0.0

    for model_name in MODEL_NAMES:
        pivot = (
            shelf_year_lookup[
                model_name
            ]
            .pivot(
                index=SHELF_COLUMN,
                columns=YEAR_COLUMN,
                values=(
                    "residual_observed_minus_predicted"
                ),
            )
            .reindex(
                index=shelves,
                columns=years,
            )
        )

        matrices[
            model_name
        ] = pivot

        finite_values = np.abs(
            pivot.to_numpy(
                dtype=float
            )
        )

        finite_values = finite_values[
            np.isfinite(
                finite_values
            )
        ]

        if finite_values.size:
            maximum_absolute = max(
                maximum_absolute,
                float(
                    np.quantile(
                        finite_values,
                        0.98,
                    )
                ),
            )

    maximum_absolute = max(
        maximum_absolute,
        1.0e-6,
    )

    norm = mcolors.TwoSlopeNorm(
        vmin=-maximum_absolute,
        vcenter=0.0,
        vmax=maximum_absolute,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            13,
            11,
        ),
        constrained_layout=True,
        sharey=True,
    )

    image = None

    for axis, model_name in zip(
        axes,
        MODEL_NAMES,
    ):
        image = axis.imshow(
            matrices[
                model_name
            ].to_numpy(
                dtype=float
            ),
            cmap="RdBu_r",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )

        axis.set_title(
            MODEL_LABELS[
                model_name
            ]
        )

        axis.set_xticks(
            np.arange(
                len(years)
            )
        )

        axis.set_xticklabels(
            years,
            rotation=90,
        )

        axis.set_yticks(
            np.arange(
                len(shelves)
            )
        )

        axis.set_yticklabels(
            shelves,
        )

        axis.set_xlabel(
            "Year"
        )

    axes[0].set_ylabel(
        "Ice shelf"
    )

    colorbar = figure.colorbar(
        image,
        ax=axes.tolist(),
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )

    colorbar.set_label(
        "Observed − predicted ponded fraction"
    )

    colorbar.ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(
            xmax=1.0,
            decimals=1,
        )
    )

    save_figure(
        figure,
        output_directory
        / "temperature_model_shelf_year_residuals",
    )


# =============================================================================
# 10. Main
# =============================================================================

def main():
    """Run the manuscript temperature-model analysis."""

    if "MODEL_DATAFRAME" not in os.environ:
        raise ValueError(
            "MODEL_DATAFRAME must point to the processed "
            "common-support observation table."
        )

    input_path = Path(
        os.environ[
            "MODEL_DATAFRAME"
        ]
    )

    require_file(
        input_path,
        "Model input table",
    )

    output_directory = Path(
        os.environ.get(
            "MODEL_OUT_DIR",
            "results/manuscript_temperature_models",
        )
    )

    table_directory = (
        output_directory
        / "tables"
    )

    trace_directory = (
        output_directory
        / "traces"
    )

    figure_directory = (
        output_directory
        / "figures"
    )

    for directory in [
        output_directory,
        table_directory,
        trace_directory,
        figure_directory,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    start_year = int(
        os.environ.get(
            "START_YEAR",
            str(
                DEFAULT_START_YEAR
            ),
        )
    )

    end_year = int(
        os.environ.get(
            "END_YEAR",
            str(
                DEFAULT_END_YEAR
            ),
        )
    )

    minimum_coverage_fraction = float(
        os.environ.get(
            "MIN_COVERAGE_FRACTION",
            "0",
        )
    )

    draws = int(
        os.environ.get(
            "DRAWS",
            str(
                DEFAULT_DRAWS
            ),
        )
    )

    tune = int(
        os.environ.get(
            "TUNE",
            str(
                DEFAULT_TUNE
            ),
        )
    )

    chains = int(
        os.environ.get(
            "CHAINS",
            str(
                DEFAULT_CHAINS
            ),
        )
    )

    cores = int(
        os.environ.get(
            "CORES",
            str(
                chains
            ),
        )
    )

    target_accept = float(
        os.environ.get(
            "TARGET_ACCEPT",
            str(
                DEFAULT_TARGET_ACCEPT
            ),
        )
    )

    maximum_tree_depth = int(
        os.environ.get(
            "MAX_TREEDPTH",
            "12",
        )
    )

    random_seed = int(
        os.environ.get(
            "RANDOM_SEED",
            "42",
        )
    )

    compute_log_likelihood = env_bool(
        "LOG_LIKELIHOOD",
        "1",
    )

    overwrite_traces = env_bool(
        "OVERWRITE_TRACES",
        "0",
    )

    kappa_alpha = float(
        os.environ.get(
            "KAPPA_ALPHA",
            str(
                DEFAULT_KAPPA_ALPHA
            ),
        )
    )

    kappa_beta = float(
        os.environ.get(
            "KAPPA_BETA",
            str(
                DEFAULT_KAPPA_BETA
            ),
        )
    )

    number_of_temperature_bins = int(
        os.environ.get(
            "N_TEMP_BINS",
            "18",
        )
    )

    calibration_bin_count = int(
        os.environ.get(
            "CALIBRATION_BIN_COUNT",
            "10",
        )
    )

    maximum_prediction_draws = int(
        os.environ.get(
            "MAX_PREDICTION_DRAWS",
            "1000",
        )
    )

    prediction_chunk_size = int(
        os.environ.get(
            "PREDICTION_CHUNK_SIZE",
            "5000",
        )
    )

    curve_temperature_minimum = float(
        os.environ.get(
            "CURVE_TEMP_MIN",
            "-18",
        )
    )

    curve_temperature_maximum = float(
        os.environ.get(
            "CURVE_TEMP_MAX",
            "3",
        )
    )

    curve_point_count = int(
        os.environ.get(
            "CURVE_POINTS",
            "600",
        )
    )

    peninsula_scaling = (
        os.environ.get(
            "PENINSULA_SCALING",
            "peninsula",
        )
        .strip()
        .lower()
    )

    if peninsula_scaling not in {
        "peninsula",
        "all_shelves",
    }:
        raise ValueError(
            "PENINSULA_SCALING must be "
            "'peninsula' or 'all_shelves'."
        )

    central_threshold = float(
        os.environ.get(
            "THRESHOLD_CENTRAL",
            str(
                DEFAULT_THRESHOLD_CENTRAL
            ),
        )
    )

    lower_threshold = float(
        os.environ.get(
            "THRESHOLD_LOWER_68",
            str(
                DEFAULT_THRESHOLD_LOWER_68
            ),
        )
    )

    upper_threshold = float(
        os.environ.get(
            "THRESHOLD_UPPER_68",
            str(
                DEFAULT_THRESHOLD_UPPER_68
            ),
        )
    )

    print("\n" + "=" * 100)
    print(
        "MANUSCRIPT TEMPERATURE-RESPONSE MODELS"
    )
    print("=" * 100)

    print(
        "Input:",
        input_path.resolve(),
        flush=True,
    )

    print(
        "Output:",
        output_directory.resolve(),
        flush=True,
    )

    print(
        "Observation period:",
        f"{start_year}–{end_year}",
        flush=True,
    )

    print(
        "Peninsula scaling:",
        peninsula_scaling,
        flush=True,
    )

    print(
        "Sampling:",
        {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": cores,
            "target_accept": target_accept,
            "maximum_tree_depth": (
                maximum_tree_depth
            ),
        },
        flush=True,
    )

    print("=" * 100 + "\n")

    # -------------------------------------------------------------------------
    # Read and prepare all-shelf data
    # -------------------------------------------------------------------------

    raw = read_table(
        input_path
    )

    raw_row_count = len(
        raw
    )

    all_shelves = (
        validate_and_prepare_data(
            raw,
            start_year=start_year,
            end_year=end_year,
            minimum_coverage_fraction=(
                minimum_coverage_fraction
            ),
        )
    )

    validated_row_count = len(
        all_shelves
    )

    (
        all_shelves,
        all_temperature_mean,
        all_temperature_sd,
    ) = standardize_temperature(
        all_shelves,
        TEMPERATURE_COLUMN,
    )

    peninsula_unscaled = (
        select_peninsula_training_data(
            all_shelves.drop(
                columns=[
                    "temperature_z",
                ]
            )
        )
    )

    if (
        peninsula_scaling
        == "peninsula"
    ):
        (
            peninsula,
            peninsula_temperature_mean,
            peninsula_temperature_sd,
        ) = standardize_temperature(
            peninsula_unscaled,
            TEMPERATURE_COLUMN,
        )

    else:
        peninsula = (
            peninsula_unscaled.copy()
        )

        peninsula_temperature_mean = (
            all_temperature_mean
        )

        peninsula_temperature_sd = (
            all_temperature_sd
        )

        peninsula[
            "temperature_z"
        ] = (
            peninsula[
                TEMPERATURE_COLUMN
            ]
            - peninsula_temperature_mean
        ) / peninsula_temperature_sd

    all_shelves.to_parquet(
        table_directory
        / "all_shelves_model_input_records.parquet",
        index=False,
    )

    peninsula.to_parquet(
        table_directory
        / "peninsula_model_input_records.parquet",
        index=False,
    )

    all_shelves.to_csv(
        table_directory
        / "all_shelves_model_input_records.csv",
        index=False,
    )

    peninsula.to_csv(
        table_directory
        / "peninsula_model_input_records.csv",
        index=False,
    )

    all_data_summary = data_summary(
        all_shelves,
        model_name=ALL_SHELVES_MODEL,
        temperature_mean=(
            all_temperature_mean
        ),
        temperature_sd=(
            all_temperature_sd
        ),
    )

    peninsula_data_summary = data_summary(
        peninsula,
        model_name=PENINSULA_MODEL,
        temperature_mean=(
            peninsula_temperature_mean
        ),
        temperature_sd=(
            peninsula_temperature_sd
        ),
    )

    atomic_json_write(
        {
            ALL_SHELVES_MODEL: (
                all_data_summary
            ),
            PENINSULA_MODEL: (
                peninsula_data_summary
            ),
        },
        table_directory
        / "model_training_sample_summary.json",
    )

    # -------------------------------------------------------------------------
    # Empirical temperature bins
    # -------------------------------------------------------------------------

    all_shelves_bins = (
        make_temperature_bins(
            all_shelves,
            TEMPERATURE_COLUMN,
            number_of_bins=(
                number_of_temperature_bins
            ),
        )
    )

    peninsula_bins = (
        make_temperature_bins(
            peninsula,
            TEMPERATURE_COLUMN,
            number_of_bins=(
                number_of_temperature_bins
            ),
        )
    )

    atomic_csv_write(
        all_shelves_bins,
        table_directory
        / "all_shelves_temperature_bins.csv",
    )

    atomic_csv_write(
        peninsula_bins,
        table_directory
        / "peninsula_temperature_bins.csv",
    )

    plot_temperature_distribution(
        all_shelves,
        all_shelves_bins,
        figure_directory,
    )

    # -------------------------------------------------------------------------
    # Fit All-Shelves Beta-Binomial
    # -------------------------------------------------------------------------

    print(
        "\n[FIT] All-Shelves Beta-Binomial",
        flush=True,
    )

    all_model = (
        build_all_shelves_beta_binomial(
            all_shelves,
            kappa_alpha=kappa_alpha,
            kappa_beta=kappa_beta,
        )
    )

    all_trace = sample_model(
        all_model,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        maximum_tree_depth=(
            maximum_tree_depth
        ),
        random_seed=random_seed,
        compute_log_likelihood=(
            compute_log_likelihood
        ),
    )

    all_trace_path = save_idata_safely(
        all_trace,
        trace_directory
        / (
            "all_shelves_beta_binomial.nc"
        ),
        overwrite=overwrite_traces,
    )

    # -------------------------------------------------------------------------
    # Fit Peninsula Binomial
    # -------------------------------------------------------------------------

    print(
        "\n[FIT] Peninsula Binomial sensitivity",
        flush=True,
    )

    peninsula_model = (
        build_peninsula_binomial(
            peninsula
        )
    )

    peninsula_trace = sample_model(
        peninsula_model,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        maximum_tree_depth=(
            maximum_tree_depth
        ),
        random_seed=random_seed + 1,
        compute_log_likelihood=(
            compute_log_likelihood
        ),
    )

    peninsula_trace_path = (
        save_idata_safely(
            peninsula_trace,
            trace_directory
            / "peninsula_binomial.nc",
            overwrite=overwrite_traces,
        )
    )

    trace_lookup = {
        ALL_SHELVES_MODEL: (
            all_trace
        ),
        PENINSULA_MODEL: (
            peninsula_trace
        ),
    }

    training_lookup = {
        ALL_SHELVES_MODEL: (
            all_shelves
        ),
        PENINSULA_MODEL: (
            peninsula
        ),
    }

    scaling_lookup = {
        ALL_SHELVES_MODEL: (
            all_temperature_mean,
            all_temperature_sd,
        ),
        PENINSULA_MODEL: (
            peninsula_temperature_mean,
            peninsula_temperature_sd,
        ),
    }

    # -------------------------------------------------------------------------
    # Parameter summaries and MCMC diagnostics
    # -------------------------------------------------------------------------

    summary_tables = []
    sampler_rows = []

    for model_name in MODEL_NAMES:
        idata = trace_lookup[
            model_name
        ]

        summary = az.summary(
            idata,
            var_names=model_parameter_names(
                model_name
            ),
            hdi_prob=0.95,
            round_to=None,
        ).reset_index()

        summary = summary.rename(
            columns={
                "index": "parameter",
            }
        )

        summary.insert(
            0,
            "model_name",
            model_name,
        )

        summary_tables.append(
            summary
        )

        atomic_csv_write(
            summary,
            table_directory
            / (
                f"{model_name}_"
                "parameter_summary.csv"
            ),
        )

        sample_stats = (
            idata.sample_stats
        )

        divergences = (
            int(
                sample_stats[
                    "diverging"
                ]
                .values
                .sum()
            )
            if "diverging"
            in sample_stats
            else np.nan
        )

        maximum_observed_tree_depth = (
            int(
                sample_stats[
                    "tree_depth"
                ]
                .values
                .max()
            )
            if "tree_depth"
            in sample_stats
            else np.nan
        )

        bfmi = az.bfmi(
            idata
        )

        sampler_rows.append(
            {
                "model_name": (
                    model_name
                ),
                "n_divergences": (
                    divergences
                ),
                "maximum_tree_depth": (
                    maximum_observed_tree_depth
                ),
                "minimum_bfmi": float(
                    np.min(
                        bfmi
                    )
                ),
                "maximum_rhat": float(
                    summary[
                        "r_hat"
                    ].max()
                ),
                "minimum_bulk_ess": float(
                    summary[
                        "ess_bulk"
                    ].min()
                ),
                "minimum_tail_ess": float(
                    summary[
                        "ess_tail"
                    ].min()
                ),
            }
        )

        plot_trace_and_rank(
            idata,
            model_name,
            figure_directory,
        )

    parameter_summary = pd.concat(
        summary_tables,
        ignore_index=True,
    )

    sampler_diagnostics = pd.DataFrame(
        sampler_rows
    )

    atomic_csv_write(
        parameter_summary,
        table_directory
        / "all_model_parameter_summaries.csv",
    )

    atomic_csv_write(
        sampler_diagnostics,
        table_directory
        / "all_model_sampler_diagnostics.csv",
    )

    # -------------------------------------------------------------------------
    # Posterior response curves
    # -------------------------------------------------------------------------

    temperature_grid = np.linspace(
        curve_temperature_minimum,
        curve_temperature_maximum,
        curve_point_count,
    )

    curve_tables = []
    threshold_tables = []

    for model_number, model_name in enumerate(
        MODEL_NAMES
    ):
        temperature_mean, temperature_sd = (
            scaling_lookup[
                model_name
            ]
        )

        curve = posterior_curve(
            trace_lookup[
                model_name
            ],
            model_name,
            temperature_grid,
            temperature_mean=(
                temperature_mean
            ),
            temperature_sd=(
                temperature_sd
            ),
            maximum_draws=(
                maximum_prediction_draws
            ),
            seed=(
                random_seed
                + 100
                + model_number
            ),
        )

        curve[
            "training_temperature_minimum_c"
        ] = training_lookup[
            model_name
        ][
            TEMPERATURE_COLUMN
        ].min()

        curve[
            "training_temperature_maximum_c"
        ] = training_lookup[
            model_name
        ][
            TEMPERATURE_COLUMN
        ].max()

        curve[
            "outside_training_range"
        ] = (
            (
                curve[
                    "temperature_c"
                ]
                < curve[
                    "training_temperature_minimum_c"
                ]
            )
            | (
                curve[
                    "temperature_c"
                ]
                > curve[
                    "training_temperature_maximum_c"
                ]
            )
        )

        curve_tables.append(
            curve
        )

        threshold_tables.append(
            summarize_threshold_crossing(
                curve,
                central_threshold=(
                    central_threshold
                ),
                lower_threshold=(
                    lower_threshold
                ),
                upper_threshold=(
                    upper_threshold
                ),
            )
        )

    curves = pd.concat(
        curve_tables,
        ignore_index=True,
    )

    threshold_crossings = pd.concat(
        threshold_tables,
        ignore_index=True,
    )

    atomic_csv_write(
        curves,
        table_directory
        / "temperature_response_curves.csv",
    )

    atomic_csv_write(
        threshold_crossings,
        table_directory
        / "model_implied_threshold_temperatures.csv",
    )

    plot_response_comparison(
        all_shelves_bins,
        curves,
        figure_directory,
        central_threshold=(
            central_threshold
        ),
        lower_threshold=(
            lower_threshold
        ),
        upper_threshold=(
            upper_threshold
        ),
    )

    # -------------------------------------------------------------------------
    # Row predictions, calibration, and residuals
    # -------------------------------------------------------------------------

    prediction_lookup = {}
    calibration_lookup = {}
    shelf_year_lookup = {}
    fit_metric_rows = []

    for model_number, model_name in enumerate(
        MODEL_NAMES
    ):
        temperature_mean, temperature_sd = (
            scaling_lookup[
                model_name
            ]
        )

        predictions = (
            posterior_predictions_at_rows(
                trace_lookup[
                    model_name
                ],
                model_name,
                training_lookup[
                    model_name
                ],
                temperature_mean=(
                    temperature_mean
                ),
                temperature_sd=(
                    temperature_sd
                ),
                maximum_draws=(
                    maximum_prediction_draws
                ),
                chunk_size=(
                    prediction_chunk_size
                ),
                seed=(
                    random_seed
                    + 200
                    + model_number
                ),
            )
        )

        prediction_lookup[
            model_name
        ] = predictions

        predictions.to_parquet(
            table_directory
            / (
                f"{model_name}_"
                "row_predictions.parquet"
            ),
            index=False,
        )

        calibration = (
            make_prediction_calibration(
                predictions,
                number_of_bins=(
                    calibration_bin_count
                ),
            )
        )

        calibration[
            "model_name"
        ] = model_name

        calibration_lookup[
            model_name
        ] = calibration

        atomic_csv_write(
            calibration,
            table_directory
            / (
                f"{model_name}_"
                "calibration.csv"
            ),
        )

        shelf_year = (
            aggregate_shelf_year_residuals(
                predictions
            )
        )

        shelf_year[
            "model_name"
        ] = model_name

        shelf_year_lookup[
            model_name
        ] = shelf_year

        atomic_csv_write(
            shelf_year,
            table_directory
            / (
                f"{model_name}_"
                "shelf_year_residuals.csv"
            ),
        )

        shelf_residuals = (
            aggregate_shelf_residuals(
                predictions
            )
        )

        shelf_residuals[
            "model_name"
        ] = model_name

        atomic_csv_write(
            shelf_residuals,
            table_directory
            / (
                f"{model_name}_"
                "shelf_residuals.csv"
            ),
        )

        fit_metric_rows.append(
            {
                "model_name": (
                    model_name
                ),
                "evaluation_scope": (
                    "in_sample_training_records"
                ),
                **row_fit_metrics(
                    predictions
                ),
            }
        )

    calibration_table = pd.concat(
        calibration_lookup.values(),
        ignore_index=True,
    )

    fit_metrics = pd.DataFrame(
        fit_metric_rows
    )

    atomic_csv_write(
        calibration_table,
        table_directory
        / "all_model_calibration.csv",
    )

    atomic_csv_write(
        fit_metrics,
        table_directory
        / "all_model_in_sample_fit_metrics.csv",
    )

    plot_calibration(
        calibration_lookup,
        figure_directory,
    )

    plot_row_residuals(
        prediction_lookup,
        figure_directory,
    )

    plot_shelf_year_heatmaps(
        shelf_year_lookup,
        figure_directory,
    )

    # -------------------------------------------------------------------------
    # Information criteria within native training samples
    # -------------------------------------------------------------------------

    loo_rows = []

    if compute_log_likelihood:
        for model_name in MODEL_NAMES:
            try:
                loo = az.loo(
                    trace_lookup[
                        model_name
                    ],
                    pointwise=True,
                )

                loo_rows.append(
                    {
                        "model_name": (
                            model_name
                        ),
                        "training_scope": (
                            "all_shelves"
                            if model_name
                            == ALL_SHELVES_MODEL
                            else (
                                "five_peninsula_shelves"
                            )
                        ),
                        "elpd_loo": float(
                            loo.elpd_loo
                        ),
                        "se": float(
                            loo.se
                        ),
                        "p_loo": float(
                            loo.p_loo
                        ),
                        "warning": bool(
                            loo.warning
                        ),
                        "comparison_note": (
                            "Do not directly compare LOO "
                            "between these models because "
                            "their training samples differ."
                        ),
                    }
                )

            except Exception as error:
                loo_rows.append(
                    {
                        "model_name": (
                            model_name
                        ),
                        "training_scope": (
                            "unknown"
                        ),
                        "elpd_loo": np.nan,
                        "se": np.nan,
                        "p_loo": np.nan,
                        "warning": True,
                        "comparison_note": repr(
                            error
                        ),
                    }
                )

    loo_table = pd.DataFrame(
        loo_rows
    )

    if not loo_table.empty:
        atomic_csv_write(
            loo_table,
            table_directory
            / "native_training_sample_loo.csv",
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "analysis": (
            "Manuscript temperature-response models"
        ),
        "input_table": str(
            input_path.resolve()
        ),
        "input_sha256": file_sha256(
            input_path
        ),
        "raw_row_count": int(
            raw_row_count
        ),
        "validated_all_shelf_row_count": int(
            validated_row_count
        ),
        "observation_period": [
            start_year,
            end_year,
        ],
        "minimum_coverage_fraction": (
            minimum_coverage_fraction
        ),
        "models": {
            ALL_SHELVES_MODEL: {
                "training_scope": (
                    "all valid ice-shelf records"
                ),
                "likelihood": (
                    "BetaBinomial"
                ),
                "formula": (
                    "y_i ~ BetaBinomial("
                    "n_i, mu_i*kappa, "
                    "(1-mu_i)*kappa); "
                    "cloglog(mu_i) = "
                    "alpha_all + beta_temp_all*T_all_z_i"
                ),
                "temperature_mean_c": (
                    all_temperature_mean
                ),
                "temperature_sd_c": (
                    all_temperature_sd
                ),
                "trace_path": str(
                    all_trace_path.resolve()
                ),
            },
            PENINSULA_MODEL: {
                "training_scope": (
                    "George VI, Larsen B remnant, "
                    "Larsen C, Larsen D, and Stange"
                ),
                "likelihood": (
                    "Binomial"
                ),
                "formula": (
                    "y_i ~ Binomial(n_i, mu_i); "
                    "cloglog(mu_i) = "
                    "alpha_ap + beta_temp_ap*T_ap_z_i"
                ),
                "temperature_scaling": (
                    peninsula_scaling
                ),
                "temperature_mean_c": (
                    peninsula_temperature_mean
                ),
                "temperature_sd_c": (
                    peninsula_temperature_sd
                ),
                "trace_path": str(
                    peninsula_trace_path.resolve()
                ),
                "bootstrap_note": (
                    "The manuscript 95% Peninsula uncertainty "
                    "interval requires a separate 1,000-replicate "
                    "whole-shelf bootstrap."
                ),
            },
        },
        "priors": {
            "alpha": (
                "Normal(mean=-6, sd=3)"
            ),
            "beta_temperature": (
                "Normal(mean=0, sd=1.5)"
            ),
            "kappa": (
                f"Gamma(shape={kappa_alpha}, "
                f"rate={kappa_beta})"
            ),
        },
        "sampling": {
            "draws_per_chain": (
                draws
            ),
            "tuning_iterations_per_chain": (
                tune
            ),
            "chains": (
                chains
            ),
            "cores": (
                cores
            ),
            "target_accept": (
                target_accept
            ),
            "maximum_tree_depth": (
                maximum_tree_depth
            ),
            "all_shelves_random_seed": (
                random_seed
            ),
            "peninsula_random_seed": (
                random_seed + 1
            ),
            "compute_log_likelihood": (
                compute_log_likelihood
            ),
        },
        "thresholds_fraction": {
            "lower_68": (
                lower_threshold
            ),
            "central": (
                central_threshold
            ),
            "upper_68": (
                upper_threshold
            ),
        },
        "residual_conventions": {
            "si_row_and_shelf_year_plots": (
                "observed_minus_predicted; "
                "positive means underprediction"
            ),
            "fit_metric_bias": (
                "predicted_minus_observed; "
                "positive means overprediction"
            ),
        },
        "fit_metrics_note": (
            "The fit metrics are in-sample descriptions "
            "and are not independent predictive skill estimates."
        ),
        "loo_note": (
            "Native-sample LOO values must not be directly "
            "compared because the two models use different "
            "training samples."
        ),
    }

    atomic_json_write(
        metadata,
        output_directory
        / "analysis_metadata.json",
    )

    print("\n" + "=" * 100)
    print(
        "SAMPLER DIAGNOSTICS"
    )
    print("=" * 100)

    print(
        sampler_diagnostics.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print(
        "IN-SAMPLE FIT METRICS"
    )
    print("=" * 100)

    print(
        fit_metrics.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print(
        "DONE"
    )
    print("=" * 100)

    print(
        "Outputs:",
        output_directory.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()