#!/usr/bin/env python
# coding: utf-8

"""
Generate historical hindcasts for all Antarctic ice shelves and create the
vertically stacked Larsen B/Larsen C Figure 3.
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

import json
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_DIR = Path(
    os.environ.get(
        "PROJECT_DIR",
        "/raid01/mafields/project_two",
    )
)

CODE_DIR = Path(
    os.environ.get(
        "CODE_DIR",
        "/raid01/mafields/tas/MODELS_filtered/ssp585/jupyter",
    )
)

DEFAULT_MODEL_RESULTS_DIR = (
    "/raid01/mafields/tas/MODELS_filtered/ssp585/jupyter/"
    "old_python_scripts/results/manuscript_temperature_models"
)

MODEL_RESULTS_DIR = Path(
    os.environ.get(
        "MODEL_RESULTS_DIR",
        DEFAULT_MODEL_RESULTS_DIR,
    )
)

TRACE_DIR = MODEL_RESULTS_DIR / "traces"
MODEL_TABLE_DIR = MODEL_RESULTS_DIR / "tables"

ALL_SHELVES_TRACE_PATH = Path(
    os.environ.get(
        "ALL_SHELVES_TRACE_PATH",
        str(
            TRACE_DIR
            / "all_shelves_beta_binomial.nc"
        ),
    )
)

PENINSULA_TRACE_PATH = Path(
    os.environ.get(
        "PENINSULA_TRACE_PATH",
        str(
            TRACE_DIR
            / "peninsula_binomial.nc"
        ),
    )
)

ALL_SHELVES_TRAINING_PATH = Path(
    os.environ.get(
        "ALL_SHELVES_TRAINING_PATH",
        str(
            MODEL_TABLE_DIR
            / "all_shelves_model_input_records.parquet"
        ),
    )
)

PENINSULA_TRAINING_PATH = Path(
    os.environ.get(
        "PENINSULA_TRAINING_PATH",
        str(
            MODEL_TABLE_DIR
            / "peninsula_model_input_records.parquet"
        ),
    )
)

ALL_SHELVES_METADATA_PATH = Path(
    os.environ.get(
        "ALL_SHELVES_METADATA_PATH",
        str(
            MODEL_RESULTS_DIR
            / "all_shelves_model_metadata.json"
        ),
    )
)

PENINSULA_METADATA_PATH = Path(
    os.environ.get(
        "PENINSULA_METADATA_PATH",
        str(
            MODEL_RESULTS_DIR
            / "peninsula_model_metadata.json"
        ),
    )
)

ERA5_TEMPERATURE_CELLS_PATH = Path(
    os.environ.get(
        "ERA5_TEMPERATURE_CELLS_PATH",
        str(
            PROJECT_DIR
            / (
                "era5_all_shelves_native_djf_"
                "temperature_cells_1979_2025.parquet"
            )
        ),
    )
)

COUNT_PATH = Path(
    os.environ.get(
        "COUNT_PATH",
        str(
            CODE_DIR
            / "native_climate_grid_30m_counts_for_beta_binomial"
            / "era5_native_grid_30m_ponding_counts.parquet"
        ),
    )
)

OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        "all_shelves_historical_hindcast",
    )
)

FIGURE_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
METADATA_DIR = OUT_DIR / "metadata"

for directory in (
    OUT_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    METADATA_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

PENINSULA_BOOTSTRAP_PARAMETER_PATH = Path(
    os.environ.get(
        "PENINSULA_BOOTSTRAP_PARAMETER_PATH",
        str(
            TABLE_DIR
            / "peninsula_binomial_shelf_bootstrap_parameters.csv"
        ),
    )
)

OUTPUT_FIGURE_NAME = (
    "Figure_3_larsen_b_and_larsen_c_hindcasts.png"
)

ALL_SHELF_PREDICTION_OUTPUT_NAME = (
    "all_shelves_hindcast_predictions.csv"
)

ALL_SHELF_SUMMARY_OUTPUT_NAME = (
    "all_shelves_hindcast_summary.csv"
)

THRESHOLD_CROSSING_OUTPUT_NAME = (
    "all_shelves_historical_threshold_crossings.csv"
)

ANNUAL_CROSS_SHELF_OUTPUT_NAME = (
    "annual_cross_shelf_hindcast_summary.csv"
)

MODEL_DIFFERENCE_OUTPUT_NAME = (
    "all_shelves_model_difference_summary.csv"
)

LARSEN_C_OBSERVATION_OUTPUT_NAME = (
    "larsen_c_annual_observed_ponded_fraction_2006_2020.csv"
)

METADATA_OUTPUT_NAME = (
    "all_shelves_hindcast_metadata.json"
)

# Initial settings 

TEMP_COL_REQUESTED = os.environ.get(
    "TEMP_COL",
    "era5_t2m_djf",
)

SHELF_COL = "shelf"
YEAR_COL = "year"
N_COL = "n_pixels_30m"
Y_COL = "y_ponded_pixels"

OBSERVATION_START_YEAR = 2006
OBSERVATION_END_YEAR = 2020

HINDCAST_START_YEAR = int(
    os.environ.get(
        "HINDCAST_START_YEAR",
        "1979",
    )
)

HINDCAST_END_YEAR = int(
    os.environ.get(
        "HINDCAST_END_YEAR",
        "2025",
    )
)

HINDCAST_PLOT_START_YEAR = int(
    os.environ.get(
        "HINDCAST_PLOT_START_YEAR",
        "1980",
    )
)

HINDCAST_PLOT_END_YEAR = int(
    os.environ.get(
        "HINDCAST_PLOT_END_YEAR",
        "2025",
    )
)

PRIMARY_WINDOW = int(
    os.environ.get(
        "PRIMARY_WINDOW",
        "10",
    )
)

LARSEN_B_COLLAPSE_YEAR = 2002
LARSEN_B_DATA_END_YEAR = 2002
  
# Connected-cluster threshold
# Not calculated in this script
  
CONNECTED_CLUSTER_THRESHOLD_PERCENT = float(
    os.environ.get(
        "CONNECTED_CLUSTER_THRESHOLD_PERCENT",
        "0.8718",
    )
)

CONNECTED_CLUSTER_THRESHOLD_LOW_PERCENT = float(
    os.environ.get(
        "CONNECTED_CLUSTER_THRESHOLD_LOW_PERCENT",
        "0.7177",
    )
)

CONNECTED_CLUSTER_THRESHOLD_HIGH_PERCENT = float(
    os.environ.get(
        "CONNECTED_CLUSTER_THRESHOLD_HIGH_PERCENT",
        "1.2638",
    )
)

CONNECTED_CLUSTER_THRESHOLD_COLOR = "#C51B7D"
  
# Posterior and bootstrap settings

MAX_POSTERIOR_DRAWS = int(
    os.environ.get(
        "MAX_POSTERIOR_DRAWS",
        "0",
    )
)

RANDOM_SEED = int(
    os.environ.get(
        "RANDOM_SEED",
        "42",
    )
)

N_PENINSULA_BOOTSTRAP = int(
    os.environ.get(
        "N_PENINSULA_BOOTSTRAP",
        "1000",
    )
)

MIN_SUCCESSFUL_BOOTSTRAP = int(
    os.environ.get(
        "MIN_SUCCESSFUL_BOOTSTRAP",
        str(
            max(
                50,
                int(
                    0.70
                    * N_PENINSULA_BOOTSTRAP
                ),
            )
        ),
    )
)

REUSE_PENINSULA_BOOTSTRAP = (
    os.environ.get(
        "REUSE_PENINSULA_BOOTSTRAP",
        "1",
    )
    == "1"
)

PENINSULA_BOOTSTRAP_ALPHA = float(
    os.environ.get(
        "PENINSULA_BOOTSTRAP_ALPHA",
        "0.22",
    )
)

# Figure settings

FIGURE_WIDTH_CM = float(
    os.environ.get(
        "FIGURE_WIDTH_CM",
        "11.0",
    )
)

FIGURE_HEIGHT_CM = float(
    os.environ.get(
        "FIGURE_HEIGHT_CM",
        "11.0",
    )
)

FIGURE_SIZE_INCHES = (
    FIGURE_WIDTH_CM / 2.54,
    FIGURE_HEIGHT_CM / 2.54,
)

SAVE_PDF = (
    os.environ.get(
        "SAVE_PDF",
        "1",
    )
    == "1"
)

AXIS_LABEL_FONT_SIZE = 8.5
AXIS_TICK_FONT_SIZE = 8.0
PANEL_LABEL_FONT_SIZE = 10.0
LEGEND_FONT_SIZE = 7.0

LARSEN_Y_MINIMUM = 0.0

LARSEN_Y_MAXIMUM = float(
    os.environ.get(
        "LARSEN_Y_MAXIMUM",
        "4.0",
    )
)
  
# Colors and general plotting style

COLOR_BETA_BINOMIAL = "#D95F02"
COLOR_PENINSULA = "#6A3D9A"
COLOR_OBSERVATIONS = "#1B1B1B"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.25",
        "grid.color": "0.86",
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

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
    "larsenbremnant": "LarsenB",
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


def normalize_shelf_name(value):
    """Normalize shelf names for matching."""

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

    return KEY_MAP.get(
        normalize_shelf_name(value),
        str(value).strip(),
    )

def check_file(path):
    """Require a file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )


def resolve_table_path(path):

    path = Path(path)

    candidates = [path]

    if path.suffix.lower() in {
        ".parquet",
        ".pq",
    }:
        candidates.extend(
            [
                path.with_suffix(".csv"),
                path.with_suffix(".csv.gz"),
            ]
        )

    elif path.suffix.lower() == ".csv":
        candidates.extend(
            [
                path.with_suffix(".parquet"),
                Path(str(path) + ".gz"),
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate table. Tried:\n"
        + "\n".join(
            str(candidate)
            for candidate in candidates
        )
    )


def read_table(path):

    path = resolve_table_path(path)

    suffixes = [
        suffix.lower()
        for suffix in path.suffixes
    ]

    if (
        ".parquet" in suffixes
        or ".pq" in suffixes
    ):
        return pd.read_parquet(path)

    if ".csv" in suffixes:
        return pd.read_csv(
            path,
            low_memory=False,
        )

    raise ValueError(
        f"Unsupported table type: {path}"
    )


def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def read_optional_json(path):

    path = Path(path)

    if not path.is_file():
        return {}

    return read_json(path)


def find_metadata_value(
    metadata,
    candidate_keys,
):

    if not isinstance(
        metadata,
        dict,
    ):
        return None

    for candidate in candidate_keys:
        if candidate in metadata:
            value = metadata[candidate]

            if (
                np.isscalar(value)
                and value is not None
            ):
                try:
                    return float(value)

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

    for value in metadata.values():
        if isinstance(value, dict):
            result = find_metadata_value(
                value,
                candidate_keys,
            )

            if result is not None:
                return result

    return None


def save_figure(figure, filename):

    output_path = FIGURE_DIR / filename

    figure.savefig(
        output_path,
        dpi=600,
        facecolor="white",
        edgecolor="none",
        pad_inches=0,
    )

    print(
        "[SAVED]",
        output_path,
    )

    if SAVE_PDF:
        pdf_path = output_path.with_suffix(
            ".pdf"
        )

        figure.savefig(
            pdf_path,
            facecolor="white",
            edgecolor="none",
            pad_inches=0,
        )

        print(
            "[SAVED]",
            pdf_path,
        )

    plt.close(figure)


def atomic_json_write(payload, path):

    path = Path(path)

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
    )


def first_present(
    columns,
    candidates,
    required=True,
):

    for candidate in candidates:
        if (
            candidate
            and candidate in columns
        ):
            return candidate

    if required:
        raise ValueError(
            "Could not find a required column.\n"
            f"Candidates: {candidates}\n"
            f"Available: {list(columns)}"
        )

    return None


def first_posterior_variable(
    trace,
    candidates,
):

    available = list(
        trace.posterior.data_vars
    )

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        "Could not identify a posterior variable.\n"
        f"Candidates: {candidates}\n"
        f"Available: {available}"
    )


def posterior_vector(
    trace,
    variable,
):

    values = np.asarray(
        trace.posterior[variable].values
    )

    if values.ndim != 2:
        raise ValueError(
            f"Expected scalar posterior variable {variable}; "
            f"received shape {values.shape}."
        )

    return values.reshape(-1).astype(float)


def thin_indices(
    number_of_draws,
    maximum_draws,
    seed,
):

    if (
        maximum_draws <= 0
        or number_of_draws
        <= maximum_draws
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


def inv_cloglog(eta):

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
        1.0e-12,
        1.0 - 1.0e-12,
    )


def weighted_mean(
    values,
    weights,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.sum(
            values[valid]
            * weights[valid]
        )
        / np.sum(
            weights[valid]
        )
    )


def identify_temperature_column(
    dataframe,
    preferred=None,
):

    return first_present(
        dataframe.columns,
        [
            preferred,
            "era5_t2m_djf_c",
            "era5_t2m_djf",
            "temperature",
            "temperature_c",
            "temp_c",
        ],
    )


def identify_count_columns(
    dataframe,
):

    n_column = first_present(
        dataframe.columns,
        [
            N_COL,
            "n_valid_pixels",
            "valid_pixels",
            "trials",
            "n",
        ],
    )

    y_column = first_present(
        dataframe.columns,
        [
            Y_COL,
            "n_ponded_pixels",
            "ponded_pixels",
            "successes",
            "y",
        ],
    )

    return n_column, y_column
  
#Load all shelves model and AP model
  
def load_model(
    *,
    key,
    label,
    family,
    trace_path,
    metadata_path,
    training_path,
    seed,
):

    check_file(trace_path)

    resolved_training_path = (
        resolve_table_path(
            training_path
        )
    )

    trace = az.from_netcdf(
        trace_path
    )

    training = read_table(
        resolved_training_path
    )

    metadata = read_optional_json(
        metadata_path
    )

    temperature_column = (
        identify_temperature_column(
            training,
            metadata.get(
                "temperature_column"
            ),
        )
    )

    temperatures = pd.to_numeric(
        training[
            temperature_column
        ],
        errors="coerce",
    )

    temperatures = temperatures[
        np.isfinite(
            temperatures
        )
    ].to_numpy(
        dtype=float
    )

    if temperatures.size == 0:
        raise ValueError(
            f"No finite training temperatures for {label}."
        )

    if key == "all_shelves_bb":
        alpha_candidates = [
            "alpha_all",
            "alpha",
            "intercept",
        ]

        beta_candidates = [
            "beta_temp_all",
            "beta_temp",
            "beta_T",
            "beta",
        ]

    elif key == "peninsula_binomial":
        alpha_candidates = [
            "alpha_ap",
            "alpha",
            "intercept",
        ]

        beta_candidates = [
            "beta_temp_ap",
            "beta_temp",
            "beta_T",
            "beta",
        ]

    else:
        raise ValueError(
            f"Unknown model key: {key}"
        )

    alpha_name = (
        first_posterior_variable(
            trace,
            alpha_candidates,
        )
    )

    beta_name = (
        first_posterior_variable(
            trace,
            beta_candidates,
        )
    )

    alpha_all = posterior_vector(
        trace,
        alpha_name,
    )

    beta_all = posterior_vector(
        trace,
        beta_name,
    )

    if len(alpha_all) != len(beta_all):
        raise ValueError(
            f"Posterior alpha and beta lengths differ for {label}."
        )

    indices = thin_indices(
        len(alpha_all),
        MAX_POSTERIOR_DRAWS,
        seed,
    )

    temperature_mean = (
        find_metadata_value(
            metadata,
            [
                "temperature_mean_for_standardization",
                "temperature_mean",
                "temperature_mean_c",
                "temperature_center",
                "temp_mean",
                "t_mean",
            ],
        )
    )

    temperature_sd = (
        find_metadata_value(
            metadata,
            [
                "temperature_sd_for_standardization",
                "temperature_sd",
                "temperature_sd_c",
                "temperature_scale",
                "temp_sd",
                "t_sd",
            ],
        )
    )

    if temperature_mean is None:
        temperature_mean = float(
            temperatures.mean()
        )

    if temperature_sd is None:
        temperature_sd = float(
            temperatures.std(
                ddof=0
            )
        )

    temperature_mean = float(
        temperature_mean
    )

    temperature_sd = float(
        temperature_sd
    )

    if (
        not np.isfinite(
            temperature_sd
        )
        or temperature_sd <= 0
    ):
        raise ValueError(
            f"Invalid temperature standard deviation for {label}."
        )

    model = {
        "key": key,
        "label": label,
        "family": family,
        "alpha": (
            alpha_all[
                indices
            ]
        ),
        "beta": (
            beta_all[
                indices
            ]
        ),
        "temperature_mean": (
            temperature_mean
        ),
        "temperature_sd": (
            temperature_sd
        ),
        "temperature_minimum": float(
            temperatures.min()
        ),
        "temperature_maximum": float(
            temperatures.max()
        ),
        "temperature_column": (
            temperature_column
        ),
        "trace_path": str(
            Path(
                trace_path
            ).resolve()
        ),
        "training_path": str(
            resolved_training_path.resolve()
        ),
        "metadata_path": (
            str(
                Path(
                    metadata_path
                ).resolve()
            )
            if Path(
                metadata_path
            ).exists()
            else str(
                metadata_path
            )
        ),
        "alpha_variable": alpha_name,
        "beta_variable": beta_name,
        "posterior_draws_available": int(
            len(
                alpha_all
            )
        ),
        "posterior_draws_used": int(
            len(
                indices
            )
        ),
    }

    print(
        f"[MODEL] {label}: "
        f"alpha_mean={np.mean(model['alpha']):.8f}; "
        f"beta_mean={np.mean(model['beta']):.8f}; "
        f"temperature_mean={temperature_mean:.8f}; "
        f"temperature_sd={temperature_sd:.8f}; "
        f"draws={len(indices):,}; "
        f"variables=({alpha_name}, {beta_name})"
    )

    return model
  
# Posterior predictions from both models

def posterior_mu_samples(
    temperature,
    model,
):

    temperature = np.asarray(
        temperature,
        dtype=float,
    )

    standardized = (
        temperature
        - model[
            "temperature_mean"
        ]
    ) / model[
        "temperature_sd"
    ]

    eta = (
        model[
            "alpha"
        ][
            :,
            None,
        ]
        + model[
            "beta"
        ][
            :,
            None,
        ]
        * standardized[
            None,
            :,
        ]
    )

    return inv_cloglog(
        eta
    )
  
# Peninsula whole-shelf bootstrap
  
def fit_peninsula_bootstrap_replicate(
    training,
    *,
    temperature_column,
    n_column,
    y_column,
    multiplicities,
    temperature_mean,
    temperature_sd,
):

    work = training.copy()

    work[
        "_bootstrap_weight"
    ] = (
        work[
            SHELF_COL
        ]
        .map(
            multiplicities
        )
        .fillna(
            0
        )
        .to_numpy(
            dtype=float
        )
    )

    work = work.loc[
        work[
            "_bootstrap_weight"
        ]
        > 0
    ].copy()

    temperature = pd.to_numeric(
        work[
            temperature_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid_pixels = pd.to_numeric(
        work[
            n_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    ponded_pixels = pd.to_numeric(
        work[
            y_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    bootstrap_weight = work[
        "_bootstrap_weight"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            temperature
        )
        & np.isfinite(
            valid_pixels
        )
        & np.isfinite(
            ponded_pixels
        )
        & np.isfinite(
            bootstrap_weight
        )
        & (
            valid_pixels
            > 0
        )
        & (
            ponded_pixels
            >= 0
        )
        & (
            ponded_pixels
            <= valid_pixels
        )
        & (
            bootstrap_weight
            > 0
        )
    )

    temperature = temperature[
        valid
    ]

    valid_pixels = valid_pixels[
        valid
    ]

    ponded_pixels = ponded_pixels[
        valid
    ]

    bootstrap_weight = (
        bootstrap_weight[
            valid
        ]
    )

    if len(
        temperature
    ) == 0:
        raise RuntimeError(
            "Bootstrap replicate has no valid rows."
        )

    standardized = (
        temperature
        - temperature_mean
    ) / temperature_sd

    design = np.column_stack(
        [
            np.ones(
                len(
                    standardized
                )
            ),
            standardized,
        ]
    )

    response = np.column_stack(
        [
            ponded_pixels,
            (
                valid_pixels
                - ponded_pixels
            ),
        ]
    )

    result = sm.GLM(
        response,
        design,
        family=sm.families.Binomial(
            link=(
                sm.families.links.CLogLog()
            )
        ),
        freq_weights=(
            bootstrap_weight
        ),
    ).fit(
        maxiter=300,
        disp=0,
    )

    if not result.converged:
        raise RuntimeError(
            "Bootstrap model did not converge."
        )

    return {
        "alpha": float(
            result.params[
                0
            ]
        ),
        "beta": float(
            result.params[
                1
            ]
        ),
        "temperature_mean": float(
            temperature_mean
        ),
        "temperature_sd": float(
            temperature_sd
        ),
    }


def generate_peninsula_shelf_bootstrap(
    training_path,
    peninsula_model,
    seed,
):

    training = read_table(
        training_path
    )

    temperature_column = (
        identify_temperature_column(
            training,
            peninsula_model[
                "temperature_column"
            ],
        )
    )

    (
        n_column,
        y_column,
    ) = identify_count_columns(
        training
    )

    shelf_source = first_present(
        training.columns,
        [
            SHELF_COL,
            "shelf_name",
            "ice_shelf",
            "iceshelf",
        ],
    )

    training[
        SHELF_COL
    ] = (
        training[
            shelf_source
        ]
        .map(
            canonical_shelf_name
        )
    )

    training[
        temperature_column
    ] = pd.to_numeric(
        training[
            temperature_column
        ],
        errors="coerce",
    )

    training[
        n_column
    ] = pd.to_numeric(
        training[
            n_column
        ],
        errors="coerce",
    )

    training[
        y_column
    ] = pd.to_numeric(
        training[
            y_column
        ],
        errors="coerce",
    )

    valid = (
        training[
            SHELF_COL
        ].notna()
        & np.isfinite(
            training[
                temperature_column
            ]
        )
        & np.isfinite(
            training[
                n_column
            ]
        )
        & np.isfinite(
            training[
                y_column
            ]
        )
        & (
            training[
                n_column
            ]
            > 0
        )
        & (
            training[
                y_column
            ]
            >= 0
        )
        & (
            training[
                y_column
            ]
            <= training[
                n_column
            ]
        )
    )

    training = training.loc[
        valid
    ].copy()

    shelves = sorted(
        training[
            SHELF_COL
        ].unique()
    )

    if len(
        shelves
    ) < 2:
        raise RuntimeError(
            "At least two Peninsula shelves are required."
        )

    generator = np.random.default_rng(
        seed
    )

    rows = []

    for draw in range(
        N_PENINSULA_BOOTSTRAP
    ):
        sampled = generator.choice(
            shelves,
            size=len(
                shelves
            ),
            replace=True,
        )

        counts = (
            pd.Series(
                sampled
            )
            .value_counts()
        )

        multiplicities = {
            shelf: int(
                counts.get(
                    shelf,
                    0,
                )
            )
            for shelf in shelves
        }

        try:
            result = (
                fit_peninsula_bootstrap_replicate(
                    training,
                    temperature_column=(
                        temperature_column
                    ),
                    n_column=(
                        n_column
                    ),
                    y_column=(
                        y_column
                    ),
                    multiplicities=(
                        multiplicities
                    ),
                    temperature_mean=(
                        peninsula_model[
                            "temperature_mean"
                        ]
                    ),
                    temperature_sd=(
                        peninsula_model[
                            "temperature_sd"
                        ]
                    ),
                )
            )

            rows.append(
                {
                    "bootstrap_draw": (
                        draw + 1
                    ),
                    "success": True,
                    **result,
                    "sampled_shelves": (
                        ";".join(
                            sampled
                        )
                    ),
                    "error": "",
                }
            )

        except Exception as error:
            rows.append(
                {
                    "bootstrap_draw": (
                        draw + 1
                    ),
                    "success": False,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "temperature_mean": (
                        peninsula_model[
                            "temperature_mean"
                        ]
                    ),
                    "temperature_sd": (
                        peninsula_model[
                            "temperature_sd"
                        ]
                    ),
                    "sampled_shelves": (
                        ";".join(
                            sampled
                        )
                    ),
                    "error": repr(
                        error
                    ),
                }
            )

        if (
            draw == 0
            or (
                draw + 1
            )
            % 100
            == 0
            or (
                draw + 1
            )
            == N_PENINSULA_BOOTSTRAP
        ):
            print(
                "[BOOTSTRAP]",
                draw + 1,
                "of",
                N_PENINSULA_BOOTSTRAP,
            )

    parameters = pd.DataFrame(
        rows
    )

    parameters.to_csv(
        PENINSULA_BOOTSTRAP_PARAMETER_PATH,
        index=False,
    )

    print(
        "[SAVED]",
        PENINSULA_BOOTSTRAP_PARAMETER_PATH,
    )

    return parameters


def bootstrap_file_matches_model(
    parameters,
    peninsula_model,
):
    required = {
        "alpha",
        "beta",
        "temperature_mean",
        "temperature_sd",
    }

    if not required.issubset(
        parameters.columns
    ):
        return False

    means = pd.to_numeric(
        parameters[
            "temperature_mean"
        ],
        errors="coerce",
    )

    standard_deviations = (
        pd.to_numeric(
            parameters[
                "temperature_sd"
            ],
            errors="coerce",
        )
    )

    finite = (
        np.isfinite(
            means
        )
        & np.isfinite(
            standard_deviations
        )
    )

    if not finite.any():
        return False

    return bool(
        np.allclose(
            means[
                finite
            ],
            peninsula_model[
                "temperature_mean"
            ],
            rtol=0,
            atol=1.0e-10,
        )
        and np.allclose(
            standard_deviations[
                finite
            ],
            peninsula_model[
                "temperature_sd"
            ],
            rtol=0,
            atol=1.0e-10,
        )
    )


def load_peninsula_bootstrap(
    training_path,
    peninsula_model,
    seed,
):

    parameters = None
    regenerate = True

    if (
        REUSE_PENINSULA_BOOTSTRAP
        and PENINSULA_BOOTSTRAP_PARAMETER_PATH.is_file()
    ):
        parameters = pd.read_csv(
            PENINSULA_BOOTSTRAP_PARAMETER_PATH
        )

        if bootstrap_file_matches_model(
            parameters,
            peninsula_model,
        ):
            regenerate = False

            print(
                "[INFO] Reusing Peninsula bootstrap:",
                PENINSULA_BOOTSTRAP_PARAMETER_PATH,
            )

    if regenerate:
        parameters = (
            generate_peninsula_shelf_bootstrap(
                training_path,
                peninsula_model,
                seed,
            )
        )

    for column in [
        "alpha",
        "beta",
        "temperature_mean",
        "temperature_sd",
    ]:
        parameters[
            column
        ] = pd.to_numeric(
            parameters[
                column
            ],
            errors="coerce",
        )

    valid = (
        np.isfinite(
            parameters[
                "alpha"
            ]
        )
        & np.isfinite(
            parameters[
                "beta"
            ]
        )
        & np.isfinite(
            parameters[
                "temperature_mean"
            ]
        )
        & np.isfinite(
            parameters[
                "temperature_sd"
            ]
        )
        & (
            parameters[
                "temperature_sd"
            ]
            > 0
        )
    )

    if "success" in parameters.columns:
        valid &= (
            parameters[
                "success"
            ]
            .astype(
                str
            )
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

    successful = parameters.loc[
        valid
    ].copy()

    if len(
        successful
    ) < MIN_SUCCESSFUL_BOOTSTRAP:
        raise RuntimeError(
            "Too few successful bootstrap fits. "
            f"Successful={len(successful):,}; "
            f"required={MIN_SUCCESSFUL_BOOTSTRAP:,}."
        )

    print(
        "[INFO] Successful Peninsula bootstrap fits:",
        f"{len(successful):,}",
    )

    return {
        "alpha": successful[
            "alpha"
        ].to_numpy(
            dtype=float
        ),
        "beta": successful[
            "beta"
        ].to_numpy(
            dtype=float
        ),
        "temperature_mean": successful[
            "temperature_mean"
        ].to_numpy(
            dtype=float
        ),
        "temperature_sd": successful[
            "temperature_sd"
        ].to_numpy(
            dtype=float
        ),
        "number_of_draws": int(
            len(
                successful
            )
        ),
    }


def peninsula_bootstrap_mu_samples(
    temperature,
    bootstrap,
):
    """Return bootstrap expected-fraction samples."""

    temperature = np.asarray(
        temperature,
        dtype=float,
    )

    standardized = (
        temperature[
            None,
            :,
        ]
        - bootstrap[
            "temperature_mean"
        ][
            :,
            None,
        ]
    ) / bootstrap[
        "temperature_sd"
    ][
        :,
        None,
    ]

    eta = (
        bootstrap[
            "alpha"
        ][
            :,
            None,
        ]
        + bootstrap[
            "beta"
        ][
            :,
            None,
        ]
        * standardized
    )

    return inv_cloglog(
        eta
    )
  
# Historical ERA5 temperatures
  
def load_temperature_cells():

    check_file(
        ERA5_TEMPERATURE_CELLS_PATH
    )

    data = read_table(
        ERA5_TEMPERATURE_CELLS_PATH
    )

    shelf_source = first_present(
        data.columns,
        [
            SHELF_COL,
            "shelf_name",
            "ice_shelf",
            "iceshelf",
        ],
    )

    year_source = first_present(
        data.columns,
        [
            YEAR_COL,
            "season_year",
            "observation_year",
        ],
    )

    temperature_column = (
        identify_temperature_column(
            data,
            TEMP_COL_REQUESTED,
        )
    )

    data[
        SHELF_COL
    ] = (
        data[
            shelf_source
        ]
        .map(
            canonical_shelf_name
        )
    )

    data[
        YEAR_COL
    ] = pd.to_numeric(
        data[
            year_source
        ],
        errors="coerce",
    )

    data[
        temperature_column
    ] = pd.to_numeric(
        data[
            temperature_column
        ],
        errors="coerce",
    )

    if "weight" in data.columns:
        weights = pd.to_numeric(
            data[
                "weight"
            ],
            errors="coerce",
        )

        weight_source = (
            "weight"
        )

    elif N_COL in data.columns:
        weights = pd.to_numeric(
            data[
                N_COL
            ],
            errors="coerce",
        )

        weight_source = (
            N_COL
        )

    elif (
        "overlap_area_m2"
        in data.columns
    ):
        weights = pd.to_numeric(
            data[
                "overlap_area_m2"
            ],
            errors="coerce",
        )

        weight_source = (
            "overlap_area_m2"
        )

    elif (
        "effective_overlap_area_m2"
        in data.columns
    ):
        weights = pd.to_numeric(
            data[
                "effective_overlap_area_m2"
            ],
            errors="coerce",
        )

        weight_source = (
            "effective_overlap_area_m2"
        )

    else:
        weights = pd.Series(
            1.0,
            index=data.index,
        )

        weight_source = (
            "equal_cell_weight"
        )

    data[
        "weight"
    ] = np.where(
        np.isfinite(
            weights
        )
        & (
            weights > 0
        ),
        weights,
        np.nan,
    )

    valid = (
        data[
            SHELF_COL
        ].notna()
        & np.isfinite(
            data[
                YEAR_COL
            ]
        )
        & np.isfinite(
            data[
                temperature_column
            ]
        )
        & np.isfinite(
            data[
                "weight"
            ]
        )
        & (
            data[
                "weight"
            ]
            > 0
        )
        & (
            data[
                YEAR_COL
            ]
            >= HINDCAST_START_YEAR
        )
        & (
            data[
                YEAR_COL
            ]
            <= HINDCAST_END_YEAR
        )
    )

    data = data.loc[
        valid
    ].copy()

    data[
        YEAR_COL
    ] = (
        data[
            YEAR_COL
        ]
        .round()
        .astype(
            int
        )
    )

    data[
        "temperature"
    ] = data[
        temperature_column
    ]

    data[
        "weight_source"
    ] = weight_source

    if data.empty:
        raise RuntimeError(
            "No valid historical shelf-temperature rows were found."
        )

    available_shelves = set(
        data[
            SHELF_COL
        ]
        .dropna()
        .unique()
    )

    required_figure_shelves = {
        "LarsenB",
        "LarsenC",
    }

    missing_figure_shelves = (
        required_figure_shelves
        - available_shelves
    )

    if missing_figure_shelves:
        raise RuntimeError(
            "Historical temperature table is missing Figure 3 shelves: "
            + ", ".join(
                sorted(
                    missing_figure_shelves
                )
            )
        )

    geometry_source_column = (
        first_present(
            data.columns,
            [
                "geometry_source",
                "boundary_source",
                "outline_source",
                "shelf_geometry_source",
                "shapefile_path",
            ],
            required=False,
        )
    )

    if (
        geometry_source_column
        is not None
    ):
        larsen_b_sources = (
            data.loc[
                data[
                    SHELF_COL
                ]
                == "LarsenB",
                geometry_source_column,
            ]
            .dropna()
            .astype(
                str
            )
        )

        if not larsen_b_sources.empty:
            pre_collapse_match = (
                larsen_b_sources
                .str.contains(
                    (
                        "pre.?collapse|pre.?2002|"
                        "gone.?outline|reconstruct"
                    ),
                    case=False,
                    regex=True,
                    na=False,
                )
            )

            if not pre_collapse_match.all():
                examples = sorted(
                    larsen_b_sources.loc[
                        ~pre_collapse_match
                    ].unique()
                )[
                    :10
                ]

                raise RuntimeError(
                    "Some Larsen B rows do not identify a reconstructed "
                    "pre-collapse boundary. Examples:\n"
                    + "\n".join(
                        examples
                    )
                )

    print(
        "[INFO] Historical temperature column:",
        temperature_column,
    )

    print(
        "[INFO] Historical weighting source:",
        weight_source,
    )

    print(
        "[INFO] Historical shelves:",
        data[
            SHELF_COL
        ].nunique(),
    )

    print(
        "[INFO] Historical shelf-cell rows:",
        f"{len(data):,}",
    )

    return data
  
# Shelf-year hindcast
  
def add_causal_moving_averages(
    dataframe,
):

    pieces = []

    for (
        shelf,
        group,
    ) in dataframe.groupby(
        SHELF_COL,
        observed=True,
    ):
        group = (
            group.sort_values(
                YEAR_COL
            )
            .drop_duplicates(
                YEAR_COL,
                keep="first",
            )
            .set_index(
                YEAR_COL
            )
        )

        complete_years = pd.Index(
            range(
                int(
                    group.index.min()
                ),
                int(
                    group.index.max()
                )
                + 1,
            ),
            name=YEAR_COL,
        )

        group = group.reindex(
            complete_years
        )

        group[
            SHELF_COL
        ] = shelf

        for model_key in [
            "all_shelves_bb",
            "peninsula_binomial",
        ]:
            median_column = (
                f"{model_key}_median"
            )

            moving_column = (
                f"{model_key}_prior_"
                f"{PRIMARY_WINDOW}yr"
            )

            group[
                moving_column
            ] = (
                group[
                    median_column
                ]
                .shift(
                    1
                )
                .rolling(
                    window=(
                        PRIMARY_WINDOW
                    ),
                    min_periods=(
                        PRIMARY_WINDOW
                    ),
                )
                .mean()
            )

        pieces.append(
            group.reset_index()
        )

    return (
        pd.concat(
            pieces,
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            [
                SHELF_COL,
                YEAR_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )


def project_all_shelf_years(
    cells,
    models,
    bootstrap,
):

    rows = []

    grouped = cells.groupby(
        [
            SHELF_COL,
            YEAR_COL,
        ],
        observed=True,
    )

    for (
        shelf,
        year,
    ), group in grouped:

        temperature = group[
            "temperature"
        ].to_numpy(
            dtype=float
        )

        weights = group[
            "weight"
        ].to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(
                temperature
            )
            & np.isfinite(
                weights
            )
            & (
                weights > 0
            )
        )

        if not valid.any():
            continue

        temperature = temperature[
            valid
        ]

        weights = weights[
            valid
        ]

        normalized_weights = (
            weights
            / weights.sum()
        )

        row = {
            SHELF_COL: shelf,
            YEAR_COL: int(
                year
            ),
            "n_cells": int(
                valid.sum()
            ),
            "temperature_weighted_mean": (
                weighted_mean(
                    temperature,
                    weights,
                )
            ),
            "total_cell_weight": float(
                weights.sum()
            ),
        }

        for (
            model_key,
            model,
        ) in models.items():
            samples = (
                posterior_mu_samples(
                    temperature,
                    model,
                )
            )

            shelf_samples = (
                samples
                @ normalized_weights
            )

            row[
                f"{model_key}_mean"
            ] = float(
                np.mean(
                    shelf_samples
                )
            )

            row[
                f"{model_key}_q025"
            ] = float(
                np.quantile(
                    shelf_samples,
                    0.025,
                )
            )

            row[
                f"{model_key}_median"
            ] = float(
                np.quantile(
                    shelf_samples,
                    0.500,
                )
            )

            row[
                f"{model_key}_q975"
            ] = float(
                np.quantile(
                    shelf_samples,
                    0.975,
                )
            )

        bootstrap_samples = (
            peninsula_bootstrap_mu_samples(
                temperature,
                bootstrap,
            )
            @ normalized_weights
        )

        row[
            "peninsula_binomial_bootstrap_q025"
        ] = float(
            np.quantile(
                bootstrap_samples,
                0.025,
            )
        )

        row[
            "peninsula_binomial_bootstrap_median"
        ] = float(
            np.quantile(
                bootstrap_samples,
                0.500,
            )
        )

        row[
            "peninsula_binomial_bootstrap_q975"
        ] = float(
            np.quantile(
                bootstrap_samples,
                0.975,
            )
        )

        rows.append(
            row
        )

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        raise RuntimeError(
            "All-shelf shelf-year projection produced no rows."
        )

    output = (
        output.sort_values(
            [
                SHELF_COL,
                YEAR_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return add_causal_moving_averages(
        output
    )

# Summary tables for hindcast using both models
# and every ice shelf (47 used in the study)
  
def make_all_shelf_hindcast_summary(
    predictions,
):

    rows = []

    for (
        shelf,
        shelf_data,
    ) in predictions.groupby(
        SHELF_COL,
        observed=True,
    ):
        for model_key in [
            "all_shelves_bb",
            "peninsula_binomial",
        ]:
            annual_column = (
                f"{model_key}_median"
            )

            moving_column = (
                f"{model_key}_prior_"
                f"{PRIMARY_WINDOW}yr"
            )

            annual_values = pd.to_numeric(
                shelf_data[
                    annual_column
                ],
                errors="coerce",
            )

            valid_annual = np.isfinite(
                annual_values
            )

            data = shelf_data.loc[
                valid_annual
            ].copy()

            if data.empty:
                continue

            annual_fraction = (
                annual_values.loc[
                    valid_annual
                ]
                .to_numpy(
                    dtype=float
                )
            )

            annual_percent = (
                100.0
                * annual_fraction
            )

            years = data[
                YEAR_COL
            ].to_numpy(
                dtype=int
            )

            maximum_index = int(
                np.argmax(
                    annual_percent
                )
            )

            annual_crossing = (
                annual_percent
                >= CONNECTED_CLUSTER_THRESHOLD_PERCENT
            )

            moving_percent = (
                100.0
                * pd.to_numeric(
                    data[
                        moving_column
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
            )

            valid_moving = np.isfinite(
                moving_percent
            )

            moving_crossing = (
                valid_moving
                & (
                    moving_percent
                    >= CONNECTED_CLUSTER_THRESHOLD_PERCENT
                )
            )

            rows.append(
                {
                    "shelf": shelf,
                    "model": model_key,
                    "first_year": int(
                        years.min()
                    ),
                    "last_year": int(
                        years.max()
                    ),
                    "n_years": int(
                        len(
                            years
                        )
                    ),
                    "mean_predicted_percent": float(
                        np.mean(
                            annual_percent
                        )
                    ),
                    "median_predicted_percent": float(
                        np.median(
                            annual_percent
                        )
                    ),
                    "minimum_predicted_percent": float(
                        np.min(
                            annual_percent
                        )
                    ),
                    "maximum_predicted_percent": float(
                        annual_percent[
                            maximum_index
                        ]
                    ),
                    "maximum_year": int(
                        years[
                            maximum_index
                        ]
                    ),
                    "n_annual_central_exceedances": int(
                        annual_crossing.sum()
                    ),
                    "fraction_annual_central_exceedances": float(
                        annual_crossing.mean()
                    ),
                    "first_annual_central_exceedance_year": (
                        int(
                            years[
                                np.flatnonzero(
                                    annual_crossing
                                )[
                                    0
                                ]
                            ]
                        )
                        if annual_crossing.any()
                        else np.nan
                    ),
                    "n_prior_10yr_central_exceedances": int(
                        moving_crossing.sum()
                    ),
                    "first_prior_10yr_central_exceedance_year": (
                        int(
                            years[
                                np.flatnonzero(
                                    moving_crossing
                                )[
                                    0
                                ]
                            ]
                        )
                        if moving_crossing.any()
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def make_threshold_crossing_table(
    predictions,
):

    rows = []

    for (
        shelf,
        shelf_data,
    ) in predictions.groupby(
        SHELF_COL,
        observed=True,
    ):
        shelf_data = shelf_data.sort_values(
            YEAR_COL
        )

        for model_key in [
            "all_shelves_bb",
            "peninsula_binomial",
        ]:
            annual_percent = (
                100.0
                * pd.to_numeric(
                    shelf_data[
                        f"{model_key}_median"
                    ],
                    errors="coerce",
                )
            )

            moving_percent = (
                100.0
                * pd.to_numeric(
                    shelf_data[
                        (
                            f"{model_key}_prior_"
                            f"{PRIMARY_WINDOW}yr"
                        )
                    ],
                    errors="coerce",
                )
            )

            valid_annual = np.isfinite(
                annual_percent
            )

            valid_moving = np.isfinite(
                moving_percent
            )

            annual_exceedance = (
                valid_annual
                & (
                    annual_percent
                    >= CONNECTED_CLUSTER_THRESHOLD_PERCENT
                )
            )

            moving_exceedance = (
                valid_moving
                & (
                    moving_percent
                    >= CONNECTED_CLUSTER_THRESHOLD_PERCENT
                )
            )

            annual_years = shelf_data.loc[
                annual_exceedance,
                YEAR_COL,
            ]

            moving_years = shelf_data.loc[
                moving_exceedance,
                YEAR_COL,
            ]

            rows.append(
                {
                    "shelf": shelf,
                    "model": model_key,
                    "central_threshold_percent": (
                        CONNECTED_CLUSTER_THRESHOLD_PERCENT
                    ),
                    "threshold_68_low_percent": (
                        CONNECTED_CLUSTER_THRESHOLD_LOW_PERCENT
                    ),
                    "threshold_68_high_percent": (
                        CONNECTED_CLUSTER_THRESHOLD_HIGH_PERCENT
                    ),
                    "moving_average_window_years": (
                        PRIMARY_WINDOW
                    ),
                    "moving_average_alignment": (
                        f"year t uses t-{PRIMARY_WINDOW} through t-1"
                    ),
                    "n_annual_exceedances": int(
                        annual_exceedance.sum()
                    ),
                    "first_annual_exceedance_year": (
                        int(
                            annual_years.iloc[
                                0
                            ]
                        )
                        if not annual_years.empty
                        else np.nan
                    ),
                    "last_annual_exceedance_year": (
                        int(
                            annual_years.iloc[
                                -1
                            ]
                        )
                        if not annual_years.empty
                        else np.nan
                    ),
                    "n_moving_average_exceedances": int(
                        moving_exceedance.sum()
                    ),
                    "first_moving_average_exceedance_year": (
                        int(
                            moving_years.iloc[
                                0
                            ]
                        )
                        if not moving_years.empty
                        else np.nan
                    ),
                    "last_moving_average_exceedance_year": (
                        int(
                            moving_years.iloc[
                                -1
                            ]
                        )
                        if not moving_years.empty
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def make_annual_cross_shelf_summary(
    predictions,
):

    rows = []

    for (
        year,
        group,
    ) in predictions.groupby(
        YEAR_COL,
        observed=True,
    ):
        row = {
            "year": int(
                year
            ),
            "n_shelves": int(
                group[
                    SHELF_COL
                ].nunique()
            ),
        }

        shelf_weights = pd.to_numeric(
            group[
                "total_cell_weight"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        for model_key in [
            "all_shelves_bb",
            "peninsula_binomial",
        ]:
            values = (
                100.0
                * pd.to_numeric(
                    group[
                        f"{model_key}_median"
                    ],
                    errors="coerce",
                )
            )

            valid = (
                np.isfinite(
                    values
                )
                & np.isfinite(
                    shelf_weights
                )
                & (
                    shelf_weights
                    > 0
                )
            )

            finite_values = values.loc[
                np.isfinite(
                    values
                )
            ]

            if finite_values.empty:
                continue

            row[
                f"{model_key}_equal_shelf_mean_percent"
            ] = float(
                finite_values.mean()
            )

            row[
                f"{model_key}_median_percent"
            ] = float(
                finite_values.median()
            )

            row[
                f"{model_key}_q10_percent"
            ] = float(
                finite_values.quantile(
                    0.10
                )
            )

            row[
                f"{model_key}_q90_percent"
            ] = float(
                finite_values.quantile(
                    0.90
                )
            )

            row[
                f"{model_key}_minimum_percent"
            ] = float(
                finite_values.min()
            )

            row[
                f"{model_key}_maximum_percent"
            ] = float(
                finite_values.max()
            )

            row[
                f"{model_key}_weighted_mean_percent"
            ] = (
                weighted_mean(
                    values.to_numpy(
                        dtype=float
                    )[
                        valid
                    ],
                    shelf_weights[
                        valid
                    ],
                )
                if valid.any()
                else np.nan
            )

            row[
                f"{model_key}_n_shelves_above_threshold"
            ] = int(
                (
                    finite_values
                    >= CONNECTED_CLUSTER_THRESHOLD_PERCENT
                ).sum()
            )

            row[
                f"{model_key}_percent_shelves_above_threshold"
            ] = float(
                100.0
                * (
                    finite_values
                    >= CONNECTED_CLUSTER_THRESHOLD_PERCENT
                ).mean()
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "year"
        )
        .reset_index(
            drop=True
        )
    )


def make_model_difference_summary(
    predictions,
):

    data = predictions.copy()

    data[
        "peninsula_minus_all_shelves_pp"
    ] = (
        100.0
        * (
            data[
                "peninsula_binomial_median"
            ]
            - data[
                "all_shelves_bb_median"
            ]
        )
    )

    rows = []

    for (
        shelf,
        group,
    ) in data.groupby(
        SHELF_COL,
        observed=True,
    ):
        difference = pd.to_numeric(
            group[
                "peninsula_minus_all_shelves_pp"
            ],
            errors="coerce",
        )

        difference = difference.loc[
            np.isfinite(
                difference
            )
        ]

        if difference.empty:
            continue

        rows.append(
            {
                "shelf": shelf,
                "n_years": int(
                    len(
                        difference
                    )
                ),
                "mean_peninsula_minus_all_shelves_pp": float(
                    difference.mean()
                ),
                "median_peninsula_minus_all_shelves_pp": float(
                    difference.median()
                ),
                "mean_absolute_model_difference_pp": float(
                    np.abs(
                        difference
                    ).mean()
                ),
                "maximum_absolute_model_difference_pp": float(
                    np.abs(
                        difference
                    ).max()
                ),
                "peninsula_above_all_shelves_fraction": float(
                    (
                        difference
                        > 0
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )

# Larsen C observations for comparison with Larsen B

def load_larsen_c_observations():

    check_file(
        COUNT_PATH
    )

    data = read_table(
        COUNT_PATH
    )

    shelf_source = first_present(
        data.columns,
        [
            SHELF_COL,
            "shelf_name",
            "ice_shelf",
            "iceshelf",
        ],
    )

    year_source = first_present(
        data.columns,
        [
            YEAR_COL,
            "season_year",
            "observation_year",
        ],
    )

    (
        n_source,
        y_source,
    ) = identify_count_columns(
        data
    )

    data[
        SHELF_COL
    ] = (
        data[
            shelf_source
        ]
        .map(
            canonical_shelf_name
        )
    )

    data[
        YEAR_COL
    ] = pd.to_numeric(
        data[
            year_source
        ],
        errors="coerce",
    )

    data[
        n_source
    ] = pd.to_numeric(
        data[
            n_source
        ],
        errors="coerce",
    )

    data[
        y_source
    ] = pd.to_numeric(
        data[
            y_source
        ],
        errors="coerce",
    )

    valid = (
        data[
            SHELF_COL
        ].notna()
        & np.isfinite(
            data[
                YEAR_COL
            ]
        )
        & np.isfinite(
            data[
                n_source
            ]
        )
        & np.isfinite(
            data[
                y_source
            ]
        )
        & (
            data[
                n_source
            ]
            > 0
        )
        & (
            data[
                y_source
            ]
            >= 0
        )
        & (
            data[
                y_source
            ]
            <= data[
                n_source
            ]
        )
    )

    data = data.loc[
        valid
    ].copy()

    data[
        YEAR_COL
    ] = (
        data[
            YEAR_COL
        ]
        .round()
        .astype(
            int
        )
    )

    data = data.loc[
        (
            data[
                SHELF_COL
            ]
            == "LarsenC"
        )
        & (
            data[
                YEAR_COL
            ]
            >= OBSERVATION_START_YEAR
        )
        & (
            data[
                YEAR_COL
            ]
            <= OBSERVATION_END_YEAR
        )
    ].copy()

    if data.empty:
        raise RuntimeError(
            "No Larsen C observation rows were found."
        )

    observations = (
        data.groupby(
            [
                SHELF_COL,
                YEAR_COL,
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            n_pixels_30m=(
                n_source,
                "sum",
            ),
            y_ponded_pixels=(
                y_source,
                "sum",
            ),
        )
    )

    observations[
        "observed_fraction"
    ] = (
        observations[
            "y_ponded_pixels"
        ]
        / observations[
            "n_pixels_30m"
        ]
    )

    observations[
        "observed_percent"
    ] = (
        100.0
        * observations[
            "observed_fraction"
        ]
    )

    return (
        observations.sort_values(
            YEAR_COL
        )
        .reset_index(
            drop=True
        )
    )

# Figure 3 plotting style and settings
  
def style_hindcast_axis(
    axis,
    *,
    show_xlabel,
):

    axis.set_xlim(
        HINDCAST_PLOT_START_YEAR,
        HINDCAST_PLOT_END_YEAR,
    )

    axis.set_ylim(
        LARSEN_Y_MINIMUM,
        LARSEN_Y_MAXIMUM,
    )

    axis.xaxis.set_major_locator(
        mticker.MultipleLocator(
            5
        )
    )

    axis.xaxis.set_minor_locator(
        mticker.MultipleLocator(
            1
        )
    )

    if LARSEN_Y_MAXIMUM <= 4:
        major_y_interval = 1.0
        minor_y_interval = 0.5

    elif LARSEN_Y_MAXIMUM <= 10:
        major_y_interval = 2.0
        minor_y_interval = 1.0

    else:
        major_y_interval = 5.0
        minor_y_interval = 2.5

    axis.yaxis.set_major_locator(
        mticker.MultipleLocator(
            major_y_interval
        )
    )

    axis.yaxis.set_minor_locator(
        mticker.MultipleLocator(
            minor_y_interval
        )
    )

    axis.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, position: (
                f"{value:g}%"
            )
        )
    )

    if show_xlabel:
        axis.set_xlabel(
            "Year",
            fontsize=(
                AXIS_LABEL_FONT_SIZE
            ),
            labelpad=3,
        )

    axis.set_ylabel(
        "Ponded fraction (%)",
        fontsize=(
            AXIS_LABEL_FONT_SIZE
        ),
        labelpad=4,
    )

    axis.grid(
        True,
        which="major",
        color="0.84",
        linewidth=0.7,
        alpha=0.75,
    )

    axis.grid(
        True,
        which="minor",
        color="0.92",
        linewidth=0.5,
        alpha=0.65,
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=(
            AXIS_TICK_FONT_SIZE
        ),
        direction="out",
        length=4,
        width=0.8,
    )

    axis.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2.5,
        width=0.6,
    )


def plot_shelf_hindcast(
    axis,
    predictions,
    *,
    shelf,
    panel_label,
    collapse_year=None,
    data_end_year=None,
    observations=None,
    show_observations=False,
    show_xlabel=False,
):
    """Plot one Figure 3 panel."""

    subset = predictions.loc[
        (
            predictions[
                SHELF_COL
            ]
            == shelf
        )
        & (
            predictions[
                YEAR_COL
            ]
            >= HINDCAST_PLOT_START_YEAR
        )
        & (
            predictions[
                YEAR_COL
            ]
            <= HINDCAST_PLOT_END_YEAR
        )
    ].sort_values(
        YEAR_COL
    ).copy()

    if data_end_year is not None:
        subset = subset.loc[
            subset[
                YEAR_COL
            ]
            <= data_end_year
        ].copy()

    if subset.empty:
        raise RuntimeError(
            f"No hindcast rows are available for {shelf}."
        )

    year = subset[
        YEAR_COL
    ].to_numpy(
        dtype=float
    )

    axis.axhspan(
        CONNECTED_CLUSTER_THRESHOLD_LOW_PERCENT,
        CONNECTED_CLUSTER_THRESHOLD_HIGH_PERCENT,
        color=(
            CONNECTED_CLUSTER_THRESHOLD_COLOR
        ),
        alpha=0.12,
        linewidth=0,
        zorder=0,
    )

    axis.axhline(
        CONNECTED_CLUSTER_THRESHOLD_PERCENT,
        color=(
            CONNECTED_CLUSTER_THRESHOLD_COLOR
        ),
        linestyle="-.",
        linewidth=1.4,
        zorder=5,
    )

    axis.fill_between(
        year,
        100.0
        * subset[
            "all_shelves_bb_q025"
        ].to_numpy(
            dtype=float
        ),
        100.0
        * subset[
            "all_shelves_bb_q975"
        ].to_numpy(
            dtype=float
        ),
        color=(
            COLOR_BETA_BINOMIAL
        ),
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )

    axis.plot(
        year,
        100.0
        * subset[
            "all_shelves_bb_median"
        ].to_numpy(
            dtype=float
        ),
        color=(
            COLOR_BETA_BINOMIAL
        ),
        linestyle="-",
        linewidth=1.9,
        zorder=6,
    )

    axis.fill_between(
        year,
        100.0
        * subset[
            "peninsula_binomial_bootstrap_q025"
        ].to_numpy(
            dtype=float
        ),
        100.0
        * subset[
            "peninsula_binomial_bootstrap_q975"
        ].to_numpy(
            dtype=float
        ),
        color=(
            COLOR_PENINSULA
        ),
        alpha=(
            PENINSULA_BOOTSTRAP_ALPHA
        ),
        linewidth=0,
        zorder=2,
    )

    axis.plot(
        year,
        100.0
        * subset[
            "peninsula_binomial_median"
        ].to_numpy(
            dtype=float
        ),
        color=(
            COLOR_PENINSULA
        ),
        linestyle="-",
        linewidth=1.9,
        zorder=7,
    )

    for (
        model_key,
        color,
    ) in [
        (
            "all_shelves_bb",
            COLOR_BETA_BINOMIAL,
        ),
        (
            "peninsula_binomial",
            COLOR_PENINSULA,
        ),
    ]:
        moving_column = (
            f"{model_key}_prior_"
            f"{PRIMARY_WINDOW}yr"
        )

        moving_values = (
            100.0
            * pd.to_numeric(
                subset[
                    moving_column
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
        )

        valid_moving = (
            np.isfinite(
                year
            )
            & np.isfinite(
                moving_values
            )
        )

        if valid_moving.any():
            axis.plot(
                year[
                    valid_moving
                ],
                moving_values[
                    valid_moving
                ],
                color=color,
                linestyle="--",
                linewidth=1.5,
                zorder=8,
            )

    if (
        show_observations
        and observations is not None
        and not observations.empty
    ):
        observed_subset = observations.loc[
            (
                observations[
                    SHELF_COL
                ]
                == shelf
            )
            & (
                observations[
                    YEAR_COL
                ]
                >= OBSERVATION_START_YEAR
            )
            & (
                observations[
                    YEAR_COL
                ]
                <= OBSERVATION_END_YEAR
            )
        ].sort_values(
            YEAR_COL
        ).copy()

        if not observed_subset.empty:
            observation_years = np.arange(
                OBSERVATION_START_YEAR,
                OBSERVATION_END_YEAR
                + 1,
            )

            observed_subset = (
                observed_subset
                .set_index(
                    YEAR_COL
                )
                .reindex(
                    observation_years
                )
            )

            observed_percent = (
                pd.to_numeric(
                    observed_subset[
                        "observed_percent"
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            axis.plot(
                observation_years,
                observed_percent,
                color=(
                    COLOR_OBSERVATIONS
                ),
                linestyle="-",
                linewidth=1.05,
                marker="o",
                markersize=4.0,
                markerfacecolor=(
                    "white"
                ),
                markeredgecolor=(
                    COLOR_OBSERVATIONS
                ),
                markeredgewidth=0.9,
                zorder=20,
            )

    if collapse_year is not None:
        axis.axvline(
            collapse_year,
            color="black",
            linestyle="--",
            linewidth=1.4,
            zorder=21,
        )

    style_hindcast_axis(
        axis,
        show_xlabel=show_xlabel,
    )

    axis.text(
        -0.070,
        1.025,
        panel_label,
        transform=(
            axis.transAxes
        ),
        ha="left",
        va="bottom",
        fontsize=(
            PANEL_LABEL_FONT_SIZE
        ),
        fontweight="bold",
        clip_on=False,
        zorder=100,
    )


  
# Legend
  

def create_compact_legend_handles():
    """Create legend handles for Figure 3."""

    all_shelves_handle = (
        Patch(
            facecolor=(
                COLOR_BETA_BINOMIAL
            ),
            edgecolor="none",
            alpha=0.18,
        ),
        Line2D(
            [0],
            [0],
            color=(
                COLOR_BETA_BINOMIAL
            ),
            linestyle="-",
            linewidth=2.0,
        ),
    )

    peninsula_handle = (
        Patch(
            facecolor=(
                COLOR_PENINSULA
            ),
            edgecolor="none",
            alpha=(
                PENINSULA_BOOTSTRAP_ALPHA
            ),
        ),
        Line2D(
            [0],
            [0],
            color=(
                COLOR_PENINSULA
            ),
            linestyle="-",
            linewidth=2.0,
        ),
    )

    causal_mean_handle = (
        Line2D(
            [0],
            [0],
            color=(
                COLOR_BETA_BINOMIAL
            ),
            linestyle="--",
            linewidth=1.6,
        ),
        Line2D(
            [0],
            [0],
            color=(
                COLOR_PENINSULA
            ),
            linestyle="--",
            linewidth=1.6,
        ),
    )

    observation_handle = Line2D(
        [0],
        [0],
        color=(
            COLOR_OBSERVATIONS
        ),
        linestyle="-",
        linewidth=1.05,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=(
            COLOR_OBSERVATIONS
        ),
        markeredgewidth=0.9,
    )

    threshold_handle = (
        Patch(
            facecolor=(
                CONNECTED_CLUSTER_THRESHOLD_COLOR
            ),
            edgecolor="none",
            alpha=0.12,
        ),
        Line2D(
            [0],
            [0],
            color=(
                CONNECTED_CLUSTER_THRESHOLD_COLOR
            ),
            linestyle="-.",
            linewidth=1.6,
        ),
    )

    collapse_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    handles = [
        all_shelves_handle,
        peninsula_handle,
        causal_mean_handle,
        observation_handle,
        threshold_handle,
        collapse_handle,
    ]

    labels = [
        (
            "All-shelves median\n"
            "(95% interval)"
        ),
        (
            "Peninsula median\n"
            "(95% bootstrap)"
        ),
        (
            f"Prior {PRIMARY_WINDOW}-year means"
        ),
        (
            "Larsen C observations\n"
            f"({OBSERVATION_START_YEAR}–"
            f"{OBSERVATION_END_YEAR})"
        ),
        (
            "Cluster threshold "
            f"({CONNECTED_CLUSTER_THRESHOLD_PERCENT:.2f}%;\n"
            f"68%: "
            f"{CONNECTED_CLUSTER_THRESHOLD_LOW_PERCENT:.2f}–"
            f"{CONNECTED_CLUSTER_THRESHOLD_HIGH_PERCENT:.2f}%)"
        ),
        (
            f"Larsen B "
            f"{LARSEN_B_COLLAPSE_YEAR} collapse"
        ),
    ]

    return (
        handles,
        labels,
    )


def style_figure_legend(
    legend,
):
    """Style the Figure 3 legend."""

    legend.get_frame().set_facecolor(
        "white"
    )

    legend.get_frame().set_edgecolor(
        "0.65"
    )

    legend.get_frame().set_alpha(
        0.93
    )

    legend.get_frame().set_linewidth(
        0.7
    )
  
# Figure 3 from Main Manuscript

def create_hindcast_figure(
    predictions,
    observations,
):

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(
            FIGURE_SIZE_INCHES
        ),
        sharex=True,
        sharey=True,
        facecolor="white",
        gridspec_kw={
            "height_ratios": [
                1,
                1,
            ],
        },
    )

    figure.subplots_adjust(
        left=0.145,
        right=0.970,
        bottom=0.090,
        top=0.950,
        hspace=0.145,
    )

    (
        larsen_b_axis,
        larsen_c_axis,
    ) = axes

    plot_shelf_hindcast(
        larsen_b_axis,
        predictions,
        shelf="LarsenB",
        panel_label="A",
        collapse_year=(
            LARSEN_B_COLLAPSE_YEAR
        ),
        data_end_year=(
            LARSEN_B_DATA_END_YEAR
        ),
        observations=None,
        show_observations=False,
        show_xlabel=False,
    )

    plot_shelf_hindcast(
        larsen_c_axis,
        predictions,
        shelf="LarsenC",
        panel_label="B",
        collapse_year=None,
        data_end_year=None,
        observations=(
            observations
        ),
        show_observations=True,
        show_xlabel=True,
    )

    larsen_b_axis.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
    )

    (
        legend_handles,
        legend_labels,
    ) = create_compact_legend_handles()

    legend = larsen_b_axis.legend(
        handles=(
            legend_handles
        ),
        labels=(
            legend_labels
        ),
        loc="upper right",
        bbox_to_anchor=(
            0.98,
            0.985,
        ),
        bbox_transform=(
            larsen_b_axis.transAxes
        ),
        ncol=1,
        frameon=True,
        fontsize=(
            LEGEND_FONT_SIZE
        ),
        handlelength=3.1,
        handletextpad=0.60,
        borderpad=0.46,
        labelspacing=0.36,
        borderaxespad=0.0,
        handler_map={
            tuple: HandlerTuple(
                ndivide=None,
                pad=0.22,
            ),
        },
    )

    style_figure_legend(
        legend
    )

    legend.set_zorder(
        100
    )

    save_figure(
        figure,
        OUTPUT_FIGURE_NAME,
    )
  
# Main Function - generates hindcast for all shelves (1979 to 2025)
# Also creates Figure 3 from manuscript
  
def main():

    print(
        "\n"
        + "=" * 92
    )

    print(
        "ALL-SHELF HISTORICAL HINDCAST AND FIGURE 3"
    )

    print(
        "=" * 92
    )

    print(
        "[INFO] Model results:",
        MODEL_RESULTS_DIR,
    )

    print(
        "[INFO] Historical temperature input:",
        ERA5_TEMPERATURE_CELLS_PATH,
    )

    print(
        "[INFO] Calculation period:",
        (
            f"{HINDCAST_START_YEAR}–"
            f"{HINDCAST_END_YEAR}"
        ),
    )

    print(
        "[INFO] Figure 3 period:",
        (
            f"{HINDCAST_PLOT_START_YEAR}–"
            f"{HINDCAST_PLOT_END_YEAR}"
        ),
    )

    print(
        "[INFO] Figure orientation:",
        "two vertically stacked panels",
    )

    print(
        "[INFO] Figure dimensions:",
        (
            f"{FIGURE_WIDTH_CM:g} × "
            f"{FIGURE_HEIGHT_CM:g} cm"
        ),
    )

    all_shelves_model = load_model(
        key="all_shelves_bb",
        label=(
            "All-shelves beta-binomial"
        ),
        family="beta_binomial",
        trace_path=(
            ALL_SHELVES_TRACE_PATH
        ),
        metadata_path=(
            ALL_SHELVES_METADATA_PATH
        ),
        training_path=(
            ALL_SHELVES_TRAINING_PATH
        ),
        seed=(
            RANDOM_SEED
        ),
    )

    peninsula_model = load_model(
        key="peninsula_binomial",
        label=(
            "Antarctic Peninsula-trained binomial"
        ),
        family="binomial",
        trace_path=(
            PENINSULA_TRACE_PATH
        ),
        metadata_path=(
            PENINSULA_METADATA_PATH
        ),
        training_path=(
            PENINSULA_TRAINING_PATH
        ),
        seed=(
            RANDOM_SEED + 1
        ),
    )

    models = {
        "all_shelves_bb": (
            all_shelves_model
        ),
        "peninsula_binomial": (
            peninsula_model
        ),
    }

    bootstrap = (
        load_peninsula_bootstrap(
            PENINSULA_TRAINING_PATH,
            peninsula_model,
            RANDOM_SEED + 1000,
        )
    )

    temperature_cells = (
        load_temperature_cells()
    )

    predictions = (
        project_all_shelf_years(
            temperature_cells,
            models,
            bootstrap,
        )
    )

    prediction_path = (
        TABLE_DIR
        / ALL_SHELF_PREDICTION_OUTPUT_NAME
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    print(
        "[SAVED]",
        prediction_path,
    )

    print(
        "[INFO] Hindcast shelves:",
        predictions[
            SHELF_COL
        ].nunique(),
    )

    print(
        "[INFO] Hindcast shelf-years:",
        len(
            predictions
        ),
    )

    shelf_summary = (
        make_all_shelf_hindcast_summary(
            predictions
        )
    )

    shelf_summary_path = (
        TABLE_DIR
        / ALL_SHELF_SUMMARY_OUTPUT_NAME
    )

    shelf_summary.to_csv(
        shelf_summary_path,
        index=False,
    )

    print(
        "[SAVED]",
        shelf_summary_path,
    )

    threshold_crossings = (
        make_threshold_crossing_table(
            predictions
        )
    )

    threshold_crossing_path = (
        TABLE_DIR
        / THRESHOLD_CROSSING_OUTPUT_NAME
    )

    threshold_crossings.to_csv(
        threshold_crossing_path,
        index=False,
    )

    print(
        "[SAVED]",
        threshold_crossing_path,
    )

    annual_cross_shelf = (
        make_annual_cross_shelf_summary(
            predictions
        )
    )

    annual_cross_shelf_path = (
        TABLE_DIR
        / ANNUAL_CROSS_SHELF_OUTPUT_NAME
    )

    annual_cross_shelf.to_csv(
        annual_cross_shelf_path,
        index=False,
    )

    print(
        "[SAVED]",
        annual_cross_shelf_path,
    )

    model_difference = (
        make_model_difference_summary(
            predictions
        )
    )

    model_difference_path = (
        TABLE_DIR
        / MODEL_DIFFERENCE_OUTPUT_NAME
    )

    model_difference.to_csv(
        model_difference_path,
        index=False,
    )

    print(
        "[SAVED]",
        model_difference_path,
    )

    observations = (
        load_larsen_c_observations()
    )

    observation_path = (
        TABLE_DIR
        / LARSEN_C_OBSERVATION_OUTPUT_NAME
    )

    observations.to_csv(
        observation_path,
        index=False,
    )

    print(
        "[SAVED]",
        observation_path,
    )

    figure_coverage = {}

    for shelf in [
        "LarsenB",
        "LarsenC",
    ]:
        shelf_years = (
            predictions.loc[
                predictions[
                    SHELF_COL
                ]
                == shelf,
                YEAR_COL,
            ]
        )

        if shelf_years.empty:
            raise RuntimeError(
                f"No predictions were generated for {shelf}."
            )

        figure_coverage[
            shelf
        ] = {
            "minimum_year": int(
                shelf_years.min()
            ),
            "maximum_year": int(
                shelf_years.max()
            ),
        }

    create_hindcast_figure(
        predictions,
        observations,
    )

    metadata = {
        "analysis": (
            "All-shelf historical visible-ponding hindcast "
            "and Larsen B/Larsen C Figure 3"
        ),

        "figure": {
            "orientation": (
                "two vertically stacked panels"
            ),
            "width_cm": (
                FIGURE_WIDTH_CM
            ),
            "height_cm": (
                FIGURE_HEIGHT_CM
            ),
            "y_axis_label": (
                "Ponded fraction (%)"
            ),
            "y_minimum_percent": (
                LARSEN_Y_MINIMUM
            ),
            "y_maximum_percent": (
                LARSEN_Y_MAXIMUM
            ),
            "png_path": str(
                FIGURE_DIR
                / OUTPUT_FIGURE_NAME
            ),
            "pdf_path": (
                str(
                    (
                        FIGURE_DIR
                        / OUTPUT_FIGURE_NAME
                    ).with_suffix(
                        ".pdf"
                    )
                )
                if SAVE_PDF
                else None
            ),
        },

        "calculation_period": {
            "start_year": (
                HINDCAST_START_YEAR
            ),
            "end_year": (
                HINDCAST_END_YEAR
            ),
        },

        "output_scope": {
            "n_shelves": int(
                predictions[
                    SHELF_COL
                ].nunique()
            ),
            "n_shelf_years": int(
                len(
                    predictions
                )
            ),
        },

        "causal_moving_average": {
            "window_years": (
                PRIMARY_WINDOW
            ),
            "current_year_excluded": (
                True
            ),
            "definition": (
                f"year t uses years "
                f"t-{PRIMARY_WINDOW} through t-1"
            ),
        },

        "connected_cluster_threshold": {
            "central_percent": (
                CONNECTED_CLUSTER_THRESHOLD_PERCENT
            ),
            "central_68_interval_percent": [
                CONNECTED_CLUSTER_THRESHOLD_LOW_PERCENT,
                CONNECTED_CLUSTER_THRESHOLD_HIGH_PERCENT,
            ],
        },

        "models": {
            "all_shelves": {
                "trace_path": (
                    all_shelves_model[
                        "trace_path"
                    ]
                ),
                "training_path": (
                    all_shelves_model[
                        "training_path"
                    ]
                ),
                "temperature_mean": (
                    all_shelves_model[
                        "temperature_mean"
                    ]
                ),
                "temperature_sd": (
                    all_shelves_model[
                        "temperature_sd"
                    ]
                ),
                "alpha_variable": (
                    all_shelves_model[
                        "alpha_variable"
                    ]
                ),
                "beta_variable": (
                    all_shelves_model[
                        "beta_variable"
                    ]
                ),
            },

            "peninsula": {
                "trace_path": (
                    peninsula_model[
                        "trace_path"
                    ]
                ),
                "training_path": (
                    peninsula_model[
                        "training_path"
                    ]
                ),
                "temperature_mean": (
                    peninsula_model[
                        "temperature_mean"
                    ]
                ),
                "temperature_sd": (
                    peninsula_model[
                        "temperature_sd"
                    ]
                ),
                "alpha_variable": (
                    peninsula_model[
                        "alpha_variable"
                    ]
                ),
                "beta_variable": (
                    peninsula_model[
                        "beta_variable"
                    ]
                ),
            },
        },

        "peninsula_bootstrap": {
            "requested_replicates": (
                N_PENINSULA_BOOTSTRAP
            ),
            "successful_replicates": (
                bootstrap[
                    "number_of_draws"
                ]
            ),
            "parameter_path": str(
                PENINSULA_BOOTSTRAP_PARAMETER_PATH
            ),
        },

        "figure_shelves": {
            "LarsenB": {
                "prediction_start_year": (
                    figure_coverage[
                        "LarsenB"
                    ][
                        "minimum_year"
                    ]
                ),
                "prediction_end_year_plotted": (
                    LARSEN_B_DATA_END_YEAR
                ),
                "collapse_year": (
                    LARSEN_B_COLLAPSE_YEAR
                ),
            },

            "LarsenC": {
                "prediction_start_year": (
                    figure_coverage[
                        "LarsenC"
                    ][
                        "minimum_year"
                    ]
                ),
                "prediction_end_year": (
                    figure_coverage[
                        "LarsenC"
                    ][
                        "maximum_year"
                    ]
                ),
                "observations_path": str(
                    observation_path
                ),
            },
        },

        "tables": {
            "all_shelf_predictions": str(
                prediction_path
            ),
            "all_shelf_summary": str(
                shelf_summary_path
            ),
            "threshold_crossings": str(
                threshold_crossing_path
            ),
            "annual_cross_shelf_summary": str(
                annual_cross_shelf_path
            ),
            "model_difference_summary": str(
                model_difference_path
            ),
            "larsen_c_observations": str(
                observation_path
            ),
        },
    }

    metadata_path = (
        METADATA_DIR
        / METADATA_OUTPUT_NAME
    )

    atomic_json_write(
        metadata,
        metadata_path,
    )

    print(
        "\n"
        + "=" * 92
    )

    print(
        "DONE"
    )

    print(
        "=" * 92
    )

    print(
        "Figures:",
        FIGURE_DIR.resolve(),
    )

    print(
        "Tables:",
        TABLE_DIR.resolve(),
    )

    print(
        "Metadata:",
        METADATA_DIR.resolve(),
    )
  
# Run Main Function

if __name__ == "__main__":
    main()