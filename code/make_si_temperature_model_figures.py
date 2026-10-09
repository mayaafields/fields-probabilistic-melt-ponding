#!/usr/bin/env python
# coding: utf-8

"""
Generate model-related Supporting Information Figures S10–S16 using the All-Shelves and Antarctic Peninsula model traces.

Models:
1. All-Shelves temperature-only Beta-Binomial model
2. Antarctic Peninsula-trained temperature-only Binomial model

Figures: 
Fig. S10: All-Shelves trace plot
Fig. S11: All-Shelves rank plot
Fig. S12: Antarctic Peninsula model trace plot
Fig. S13: Antarctic Peninsula model rank plot
Fig. S14: Fitted-period calibration
Fig. S15: Row-level residual diagnostics
Fig. S16: Shelf-year residual heatmaps

Environment variables:

MODEL_RESULTS_DIR
ALL_SHELVES_TRACE
PENINSULA_TRACE
ALL_SHELVES_INPUT
PENINSULA_INPUT
ALL_SHELVES_METADATA
PENINSULA_METADATA
SI_MODEL_OUTPUT_DIR
MAX_POSTERIOR_DRAWS
CALIBRATION_BIN_COUNT
PREDICTION_ROW_CHUNK
SHARED_TEMPERATURE_STANDARDIZATION
OUTPUT_DPI
RANDOM_SEED
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
import re
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

warnings.filterwarnings("ignore")


DEFAULT_RESULTS_DIRECTORY = (
    "/raid01/mafields/tas/MODELS_filtered/ssp585/jupyter/"
    "old_python_scripts/results/manuscript_temperature_models"
)

MODEL_RESULTS_DIR = Path(
    os.environ.get(
        "MODEL_RESULTS_DIR",
        DEFAULT_RESULTS_DIRECTORY,
    )
)

TRACE_DIR = MODEL_RESULTS_DIR / "traces"
TABLE_DIR = MODEL_RESULTS_DIR / "tables"

ALL_SHELVES_TRACE = Path(
    os.environ.get(
        "ALL_SHELVES_TRACE",
        str(TRACE_DIR / "all_shelves_beta_binomial.nc"),
    )
)

PENINSULA_TRACE = Path(
    os.environ.get(
        "PENINSULA_TRACE",
        str(TRACE_DIR / "peninsula_binomial.nc"),
    )
)

ALL_SHELVES_INPUT_REQUESTED = Path(
    os.environ.get(
        "ALL_SHELVES_INPUT",
        str(TABLE_DIR / "all_shelves_model_input_records.parquet"),
    )
)

PENINSULA_INPUT_REQUESTED = Path(
    os.environ.get(
        "PENINSULA_INPUT",
        str(TABLE_DIR / "peninsula_model_input_records.parquet"),
    )
)

ALL_SHELVES_METADATA = Path(
    os.environ.get(
        "ALL_SHELVES_METADATA",
        str(MODEL_RESULTS_DIR / "all_shelves_model_metadata.json"),
    )
)

PENINSULA_METADATA = Path(
    os.environ.get(
        "PENINSULA_METADATA",
        str(MODEL_RESULTS_DIR / "peninsula_model_metadata.json"),
    )
)

SI_OUTPUT_DIR = Path(
    os.environ.get(
        "SI_MODEL_OUTPUT_DIR",
        "si_outputs/model_diagnostics",
    )
)

FIGURE_DIR = SI_OUTPUT_DIR / "figures"
OUTPUT_TABLE_DIR = SI_OUTPUT_DIR / "tables"

for directory in (
    SI_OUTPUT_DIR,
    FIGURE_DIR,
    OUTPUT_TABLE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. Model names, labels, and expected columns
# =============================================================================

ALL_SHELVES_MODEL = "all_shelves_beta_binomial"
PENINSULA_MODEL = "peninsula_binomial"

MODEL_NAMES = [
    ALL_SHELVES_MODEL,
    PENINSULA_MODEL,
]

MODEL_LABELS = {
    ALL_SHELVES_MODEL: "All-Shelves beta-binomial",
    PENINSULA_MODEL: "Antarctic Peninsula-trained binomial",
}

MODEL_COLORS = {
    ALL_SHELVES_MODEL: "#D55E00",
    PENINSULA_MODEL: "#6A3D9A",
}

PARAMETER_CANDIDATES = {
    ALL_SHELVES_MODEL: {
        "alpha": [
            "alpha_all",
            "alpha",
            "intercept",
        ],
        "beta": [
            "beta_temp_all",
            "beta_temp",
            "beta_T",
            "beta",
        ],
        "kappa": [
            "kappa",
            "phi",
        ],
    },
    PENINSULA_MODEL: {
        "alpha": [
            "alpha_ap",
            "alpha",
            "intercept",
        ],
        "beta": [
            "beta_temp_ap",
            "beta_temp",
            "beta_T",
            "beta",
        ],
    },
}

SHELF_COLUMN = "shelf"
YEAR_COLUMN = "year"
TEMPERATURE_COLUMN = "era5_t2m_djf_c"
N_COLUMN = "n_pixels_30m"
Y_COLUMN = "y_ponded_pixels"
OBSERVED_COLUMN = "observed_fraction"

RESIDUAL_COLUMN = "residual_predicted_minus_observed"

MAX_POSTERIOR_DRAWS = int(
    os.environ.get(
        "MAX_POSTERIOR_DRAWS",
        "0",
    )
)

CALIBRATION_BIN_COUNT = int(
    os.environ.get(
        "CALIBRATION_BIN_COUNT",
        "10",
    )
)

PREDICTION_ROW_CHUNK = int(
    os.environ.get(
        "PREDICTION_ROW_CHUNK",
        "4000",
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

SHARED_TEMPERATURE_STANDARDIZATION = (
    os.environ.get(
        "SHARED_TEMPERATURE_STANDARDIZATION",
        "0",
    )
    == "1"
)

# Universal Plot style

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
        "savefig.dpi": OUTPUT_DPI,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# General utility functions to support plotting and reading data

def info(message):
    print(f"[INFO] {message}", flush=True)


def warn(message):
    print(f"[WARN] {message}", flush=True)


def require_file(path, description):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def resolve_table_path(requested_path):
    """Resolve Parquet, CSV, or compressed CSV alternatives."""

    requested_path = Path(requested_path)

    candidates = [requested_path]

    if requested_path.suffix.lower() in {".parquet", ".pq"}:
        candidates.extend(
            [
                requested_path.with_suffix(".csv"),
                requested_path.with_suffix(".csv.gz"),
            ]
        )
    elif requested_path.suffix.lower() == ".csv":
        candidates.extend(
            [
                requested_path.with_suffix(".parquet"),
                Path(str(requested_path) + ".gz"),
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate model-input table. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def read_table(path):
    path = Path(path)
    require_file(path, "Input table")

    suffixes = [suffix.lower() for suffix in path.suffixes]

    if ".parquet" in suffixes or ".pq" in suffixes:
        return pd.read_parquet(path)

    if ".csv" in suffixes:
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported table type: {path}")


def read_optional_json(path):
    path = Path(path)

    if not path.is_file():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_metadata_value(metadata, candidate_keys):
    """Recursively locate a scalar value in a metadata dictionary."""

    if not isinstance(metadata, dict):
        return None

    for candidate in candidate_keys:
        if candidate in metadata:
            value = metadata[candidate]

            if np.isscalar(value) and value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
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


def atomic_csv_write(dataframe, path, *, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    dataframe.to_csv(
        temporary,
        index=index,
    )

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


def save_large_table(dataframe, basename):
    """Prefer Parquet but fall back to compressed CSV."""

    basename = Path(basename)
    parquet_path = basename.with_suffix(".parquet")

    try:
        dataframe.to_parquet(
            parquet_path,
            index=False,
        )

        print("[SAVED]", parquet_path, flush=True)
        return parquet_path

    except Exception as error:
        warn(
            f"Could not save Parquet ({type(error).__name__}: {error}). "
            "Saving compressed CSV instead."
        )

        csv_path = basename.with_suffix(".csv.gz")

        dataframe.to_csv(
            csv_path,
            index=False,
            compression="gzip",
        )

        print("[SAVED]", csv_path, flush=True)
        return csv_path


def save_figure(figure, basename):
    basename = Path(basename)

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
        color="0.87",
        linewidth=0.7,
        alpha=0.8,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def normalize_shelf_name(value):
    """Normalize common shelf-name variants."""

    if pd.isna(value):
        return ""

    value = re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )

    replacements = {
        "LarsenB": "Larsen B",
        "LarsenC": "Larsen C",
        "LarsenD": "Larsen D",
        "pine island": "Pine Island",
        "Pine island": "Pine Island",
        "Brunt Stancomb": "Brunt–Stancomb",
        "Brunt-Stancomb": "Brunt–Stancomb",
        "Conger Glenzer": "Conger–Glenzer",
    }

    return replacements.get(value, value)


def identify_temperature_column(dataframe):
    candidates = [
        TEMPERATURE_COLUMN,
        "era5_t2m_djf",
        "temperature_c",
        "temperature",
        "tas_djf_c",
    ]

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise KeyError(
        "Could not identify temperature column. "
        f"Available columns: {list(dataframe.columns)}"
    )


def identify_column(dataframe, candidates, description):
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise KeyError(
        f"Could not identify {description}. "
        f"Tried {candidates}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def standardize_input_table(dataframe, model_name):
    """Standardize required model-input columns."""

    data = dataframe.copy()

    shelf_source = identify_column(
        data,
        [
            "shelf",
            "shelf_name",
            "ice_shelf",
            "iceshelf",
        ],
        "shelf column",
    )

    year_source = identify_column(
        data,
        [
            "year",
            "season_year",
            "observation_year",
        ],
        "year column",
    )

    temperature_source = identify_temperature_column(data)

    n_source = identify_column(
        data,
        [
            "n_pixels_30m",
            "n_valid_pixels",
            "valid_pixels",
            "trials",
            "n",
        ],
        "valid-pixel count column",
    )

    y_source = identify_column(
        data,
        [
            "y_ponded_pixels",
            "n_ponded_pixels",
            "ponded_pixels",
            "successes",
            "y",
        ],
        "ponded-pixel count column",
    )

    data[SHELF_COLUMN] = (
        data[shelf_source]
        .map(normalize_shelf_name)
    )

    data[YEAR_COLUMN] = pd.to_numeric(
        data[year_source],
        errors="coerce",
    )

    data[TEMPERATURE_COLUMN] = pd.to_numeric(
        data[temperature_source],
        errors="coerce",
    )

    data[N_COLUMN] = pd.to_numeric(
        data[n_source],
        errors="coerce",
    )

    data[Y_COLUMN] = pd.to_numeric(
        data[y_source],
        errors="coerce",
    )

    valid = (
        data[SHELF_COLUMN].ne("")
        & np.isfinite(data[YEAR_COLUMN])
        & np.isfinite(data[TEMPERATURE_COLUMN])
        & np.isfinite(data[N_COLUMN])
        & (data[N_COLUMN] > 0)
        & np.isfinite(data[Y_COLUMN])
        & (data[Y_COLUMN] >= 0)
        & (data[Y_COLUMN] <= data[N_COLUMN])
    )

    data = (
        data.loc[valid]
        .copy()
        .reset_index(drop=True)
    )

    if data.empty:
        raise RuntimeError(
            f"No valid input rows remain for {model_name}."
        )

    data[YEAR_COLUMN] = (
        data[YEAR_COLUMN]
        .round()
        .astype(int)
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

    data[OBSERVED_COLUMN] = (
        data[Y_COLUMN] / data[N_COLUMN]
    )

    return data


def resolve_posterior_variable(
    idata,
    candidates,
    description,
):
    available = list(idata.posterior.data_vars)

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise KeyError(
        f"Could not identify {description}. "
        f"Tried {candidates}. Available variables: {available}"
    )


def flatten_posterior(idata, variable):
    values = np.asarray(
        idata.posterior[variable].values
    )

    if values.ndim == 2:
        return values.reshape(-1)

    return values.reshape(
        (-1,) + values.shape[2:]
    )


def choose_draw_indices(number, maximum, seed):
    if maximum <= 0 or number <= maximum:
        return np.arange(number)

    generator = np.random.default_rng(seed)

    return np.sort(
        generator.choice(
            number,
            size=maximum,
            replace=False,
        )
    )


def inv_cloglog(eta):
    eta = np.clip(
        np.asarray(eta, dtype=float),
        -30.0,
        20.0,
    )

    return np.clip(
        -np.expm1(-np.exp(eta)),
        1.0e-12,
        1.0 - 1.0e-12,
    )

def load_one_model(
    *,
    model_name,
    trace_path,
    input_path,
    metadata_path,
    seed,
):
    require_file(trace_path, f"{model_name} trace")

    resolved_input_path = resolve_table_path(input_path)

    raw_data = read_table(resolved_input_path)
    data = standardize_input_table(
        raw_data,
        model_name,
    )

    idata = az.from_netcdf(trace_path)

    resolved_variables = {}

    for role, candidates in PARAMETER_CANDIDATES[model_name].items():
        try:
            resolved_variables[role] = resolve_posterior_variable(
                idata,
                candidates,
                f"{model_name} {role}",
            )
        except KeyError:
            if role == "kappa":
                continue
            raise

    alpha_all = flatten_posterior(
        idata,
        resolved_variables["alpha"],
    )

    beta_all = flatten_posterior(
        idata,
        resolved_variables["beta"],
    )

    if alpha_all.ndim != 1 or beta_all.ndim != 1:
        raise ValueError(
            f"{model_name}: expected scalar alpha and beta. "
            f"Shapes were {alpha_all.shape} and {beta_all.shape}."
        )

    if len(alpha_all) != len(beta_all):
        raise ValueError(
            f"{model_name}: alpha and beta draw counts differ."
        )

    draw_indices = choose_draw_indices(
        len(alpha_all),
        MAX_POSTERIOR_DRAWS,
        seed,
    )

    metadata = read_optional_json(metadata_path)

    temperature_mean = find_metadata_value(
        metadata,
        [
            "temperature_mean",
            "temperature_mean_c",
            "temperature_center",
            "temp_mean",
            "t_mean",
        ],
    )

    temperature_sd = find_metadata_value(
        metadata,
        [
            "temperature_sd",
            "temperature_sd_c",
            "temperature_scale",
            "temp_sd",
            "t_sd",
        ],
    )

    if temperature_mean is None:
        temperature_mean = float(
            data[TEMPERATURE_COLUMN].mean()
        )

        warn(
            f"{MODEL_LABELS[model_name]}: temperature mean was not found "
            "in metadata; recomputed from the model-input table."
        )

    if temperature_sd is None:
        temperature_sd = float(
            data[TEMPERATURE_COLUMN].std(ddof=0)
        )

        warn(
            f"{MODEL_LABELS[model_name]}: temperature SD was not found "
            "in metadata; recomputed from the model-input table."
        )

    if not np.isfinite(temperature_sd) or temperature_sd <= 0:
        raise ValueError(
            f"{model_name}: invalid temperature SD {temperature_sd}."
        )

    return {
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
        "trace_path": Path(trace_path),
        "input_path": resolved_input_path,
        "metadata_path": Path(metadata_path),
        "metadata": metadata,
        "idata": idata,
        "training_data": data,
        "evaluation_data": data.copy(),
        "resolved_variables": resolved_variables,
        "alpha": alpha_all[draw_indices],
        "beta": beta_all[draw_indices],
        "draw_indices": draw_indices,
        "posterior_draws_available": int(len(alpha_all)),
        "posterior_draws_used": int(len(draw_indices)),
        "temperature_mean": float(temperature_mean),
        "temperature_sd": float(temperature_sd),
        "temperature_minimum": float(
            data[TEMPERATURE_COLUMN].min()
        ),
        "temperature_maximum": float(
            data[TEMPERATURE_COLUMN].max()
        ),
    }


def load_models():
    models = {
        ALL_SHELVES_MODEL: load_one_model(
            model_name=ALL_SHELVES_MODEL,
            trace_path=ALL_SHELVES_TRACE,
            input_path=ALL_SHELVES_INPUT_REQUESTED,
            metadata_path=ALL_SHELVES_METADATA,
            seed=RANDOM_SEED,
        ),
        PENINSULA_MODEL: load_one_model(
            model_name=PENINSULA_MODEL,
            trace_path=PENINSULA_TRACE,
            input_path=PENINSULA_INPUT_REQUESTED,
            metadata_path=PENINSULA_METADATA,
            seed=RANDOM_SEED + 1,
        ),
    }

    if SHARED_TEMPERATURE_STANDARDIZATION:
        models[PENINSULA_MODEL]["temperature_mean"] = (
            models[ALL_SHELVES_MODEL]["temperature_mean"]
        )

        models[PENINSULA_MODEL]["temperature_sd"] = (
            models[ALL_SHELVES_MODEL]["temperature_sd"]
        )

        info(
            "Applied the All-Shelves temperature standardization constants "
            "to the Peninsula model."
        )

    # apply the Peninsula-trained model to all shelves.
    models[PENINSULA_MODEL]["evaluation_data"] = (
        models[ALL_SHELVES_MODEL]["training_data"].copy()
    )

    info(
        "The Peninsula model will be evaluated on the complete "
        "All-Shelves common-support dataset for Figs. S15 and S16."
    )

    return models

# Posterior predictions

def posterior_probability_draws(model, temperature):
    temperature = np.asarray(
        temperature,
        dtype=float,
    )

    temperature_z = (
        temperature
        - model["temperature_mean"]
    ) / model["temperature_sd"]

    eta = (
        model["alpha"][:, None]
        + model["beta"][:, None]
        * temperature_z[None, :]
    )

    return inv_cloglog(eta)


def calculate_row_predictions(model, data):
    """Calculate posterior summaries in row chunks."""

    output = data.copy().reset_index(drop=True)

    temperature = output[
        TEMPERATURE_COLUMN
    ].to_numpy(dtype=float)

    number_of_rows = len(output)

    predicted_mean = np.full(
        number_of_rows,
        np.nan,
        dtype=float,
    )

    predicted_median = np.full(
        number_of_rows,
        np.nan,
        dtype=float,
    )

    predicted_q025 = np.full(
        number_of_rows,
        np.nan,
        dtype=float,
    )

    predicted_q975 = np.full(
        number_of_rows,
        np.nan,
        dtype=float,
    )

    for start in range(
        0,
        number_of_rows,
        PREDICTION_ROW_CHUNK,
    ):
        stop = min(
            start + PREDICTION_ROW_CHUNK,
            number_of_rows,
        )

        draws = posterior_probability_draws(
            model,
            temperature[start:stop],
        )

        predicted_mean[start:stop] = np.mean(
            draws,
            axis=0,
        )

        predicted_median[start:stop] = np.quantile(
            draws,
            0.50,
            axis=0,
        )

        predicted_q025[start:stop] = np.quantile(
            draws,
            0.025,
            axis=0,
        )

        predicted_q975[start:stop] = np.quantile(
            draws,
            0.975,
            axis=0,
        )

    output["model_name"] = model["model_name"]
    output["model_label"] = model["model_label"]

    output["predicted_mean"] = predicted_mean
    output["predicted_median"] = predicted_median
    output["predicted_q025"] = predicted_q025
    output["predicted_q975"] = predicted_q975

    # Manuscript convention:
    # residual = predicted - observed.
    output[RESIDUAL_COLUMN] = (
        output["predicted_mean"]
        - output[OBSERVED_COLUMN]
    )

    output["expected_ponded_pixels"] = (
        output["predicted_mean"]
        * output[N_COLUMN]
    )

    return output

# Calibration and aggregation

def calculate_calibration(predictions):
    """Create valid-pixel-weighted calibration bins."""

    data = predictions.copy()

    number_of_bins = min(
        CALIBRATION_BIN_COUNT,
        data["predicted_mean"].nunique(),
        len(data),
    )

    if number_of_bins < 1:
        return pd.DataFrame()

    data["_calibration_bin"] = pd.qcut(
        data["predicted_mean"].rank(method="first"),
        q=number_of_bins,
        duplicates="drop",
    )

    rows = []

    for bin_number, (_, group) in enumerate(
        data.groupby(
            "_calibration_bin",
            observed=True,
            sort=True,
        )
    ):
        total_n = float(group[N_COLUMN].sum())
        total_y = float(group[Y_COLUMN].sum())

        predicted_fraction = float(
            group["expected_ponded_pixels"].sum()
            / total_n
        )

        observed_fraction = (
            total_y / total_n
        )

        rows.append(
            {
                "model_name": group["model_name"].iloc[0],
                "model_label": group["model_label"].iloc[0],
                "calibration_bin": int(bin_number),
                "n_rows": int(len(group)),
                "n_shelves": int(
                    group[SHELF_COLUMN].nunique()
                ),
                "valid_pixels": total_n,
                "predicted_fraction": predicted_fraction,
                "observed_fraction": observed_fraction,
                "residual_predicted_minus_observed": (
                    predicted_fraction
                    - observed_fraction
                ),
            }
        )

    return pd.DataFrame(rows)


def aggregate_shelf_year_residuals(predictions):
    """Aggregate predictions to shelf-year support."""

    output = (
        predictions.groupby(
            [
                "model_name",
                "model_label",
                SHELF_COLUMN,
                YEAR_COLUMN,
            ],
            observed=True,
        )
        .agg(
            n_rows=(Y_COLUMN, "size"),
            valid_pixels=(N_COLUMN, "sum"),
            ponded_pixels=(Y_COLUMN, "sum"),
            expected_ponded_pixels=(
                "expected_ponded_pixels",
                "sum",
            ),
        )
        .reset_index()
    )

    output["observed_fraction"] = (
        output["ponded_pixels"]
        / output["valid_pixels"]
    )

    output["predicted_fraction"] = (
        output["expected_ponded_pixels"]
        / output["valid_pixels"]
    )

    # residual = predicted - observed.
    output[RESIDUAL_COLUMN] = (
        output["predicted_fraction"]
        - output["observed_fraction"]
    )

    return output


def calculate_fit_metrics(predictions):
    observed = predictions[
        OBSERVED_COLUMN
    ].to_numpy(dtype=float)

    predicted = predictions[
        "predicted_mean"
    ].to_numpy(dtype=float)

    weights = predictions[
        N_COLUMN
    ].to_numpy(dtype=float)

    residual = predicted - observed
    normalized_weights = weights / weights.sum()

    return {
        "model_name": predictions["model_name"].iloc[0],
        "model_label": predictions["model_label"].iloc[0],
        "evaluation_scope": (
            "all_shelves"
            if predictions[SHELF_COLUMN].nunique() > 5
            else "model_training_shelves"
        ),
        "residual_definition": "predicted_minus_observed",
        "n_rows": int(len(predictions)),
        "n_shelves": int(
            predictions[SHELF_COLUMN].nunique()
        ),
        "unweighted_mae": float(
            np.mean(np.abs(residual))
        ),
        "unweighted_rmse": float(
            np.sqrt(np.mean(residual**2))
        ),
        "unweighted_bias_predicted_minus_observed": float(
            np.mean(residual)
        ),
        "pixel_weighted_mae": float(
            np.sum(
                normalized_weights
                * np.abs(residual)
            )
        ),
        "pixel_weighted_rmse": float(
            np.sqrt(
                np.sum(
                    normalized_weights
                    * residual**2
                )
            )
        ),
        "pixel_weighted_bias_predicted_minus_observed": float(
            np.sum(
                normalized_weights
                * residual
            )
        ),
    }

def calculate_binned_residual_summary(
    predictions,
    x_column,
    number_of_bins=30,
):
    data = predictions.loc[
        np.isfinite(predictions[x_column])
        & np.isfinite(predictions[RESIDUAL_COLUMN])
    ].copy()

    if data.empty:
        return pd.DataFrame()

    number_of_bins = min(
        number_of_bins,
        len(data),
        data[x_column].nunique(),
    )

    if number_of_bins < 2:
        return pd.DataFrame()

    data["_rank"] = data[
        x_column
    ].rank(method="first")

    data["_bin"] = pd.qcut(
        data["_rank"],
        q=number_of_bins,
        duplicates="drop",
    )

    return (
        data.groupby(
            "_bin",
            observed=True,
        )
        .agg(
            x=(x_column, "median"),
            residual=(
                RESIDUAL_COLUMN,
                "median",
            ),
            residual_q25=(
                RESIDUAL_COLUMN,
                lambda values: np.nanquantile(
                    values,
                    0.25,
                ),
            ),
            residual_q75=(
                RESIDUAL_COLUMN,
                lambda values: np.nanquantile(
                    values,
                    0.75,
                ),
            ),
        )
        .reset_index(drop=True)
    )

def plot_trace_figure(
    model,
    figure_number,
    basename,
):
    variables = [
        model["resolved_variables"]["alpha"],
        model["resolved_variables"]["beta"],
    ]

    if "kappa" in model["resolved_variables"]:
        variables.append(
            model["resolved_variables"]["kappa"]
        )

    axes = az.plot_trace(
        model["idata"],
        var_names=variables,
        compact=True,
        figsize=(
            12.0,
            max(5.0, 3.2 * len(variables)),
        ),
    )

    axes_array = np.asarray(axes)
    figure = axes_array.ravel()[0].figure

    figure.subplots_adjust(
        hspace=0.70,
        wspace=0.28,
        top=0.95,
        bottom=0.06,
        left=0.08,
        right=0.98,
    )

    save_figure(
        figure,
        FIGURE_DIR / basename,
    )


def plot_rank_figure(
    model,
    figure_number,
    basename,
):
    variables = [
        model["resolved_variables"]["alpha"],
        model["resolved_variables"]["beta"],
    ]

    if "kappa" in model["resolved_variables"]:
        variables.append(
            model["resolved_variables"]["kappa"]
        )

    axes = az.plot_rank(
        model["idata"],
        var_names=variables,
        kind="bars",
        figsize=(
            12.0,
            max(4.5, 2.8 * len(variables)),
        ),
    )

    axes_array = np.asarray(axes)
    figure = axes_array.ravel()[0].figure

    figure.subplots_adjust(
        hspace=0.60,
        wspace=0.28,
        top=0.95,
        bottom=0.06,
        left=0.08,
        right=0.98,
    )

    save_figure(
        figure,
        FIGURE_DIR / basename,
    )

def plot_calibration(calibration_tables):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.0),
        constrained_layout=True,
    )

    for axis, model_name in zip(
        axes,
        MODEL_NAMES,
    ):
        table = calibration_tables[model_name]

        maximum = max(
            float(table["predicted_fraction"].max()),
            float(table["observed_fraction"].max()),
            1.0e-6,
        )

        sizes = table[
            "valid_pixels"
        ].to_numpy(dtype=float)

        sizes = (
            35.0
            + 120.0
            * np.sqrt(
                sizes / sizes.max()
            )
        )

        axis.plot(
            [0, maximum],
            [0, maximum],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Perfect calibration",
        )

        axis.scatter(
            table["predicted_fraction"],
            table["observed_fraction"],
            color=MODEL_COLORS[model_name],
            s=sizes,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )

        axis.set_xlim(0, maximum * 1.05)
        axis.set_ylim(0, maximum * 1.05)

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

        axis.set_xlabel(
            "Predicted ponded fraction"
        )

        axis.set_ylabel(
            "Observed ponded fraction"
        )

        axis.set_title(
            MODEL_LABELS[model_name]
        )

        axis.legend(frameon=False)
        style_axis(axis)

    save_figure(
        figure,
        FIGURE_DIR
        / "Fig_S14_temperature_model_calibration",
    )


def plot_residuals(prediction_tables):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
        constrained_layout=True,
    )

    for row, model_name in enumerate(MODEL_NAMES):
        data = prediction_tables[model_name]
        color = MODEL_COLORS[model_name]

        residual = data[RESIDUAL_COLUMN]

        axes[row, 0].scatter(
            data[TEMPERATURE_COLUMN],
            residual,
            color=color,
            alpha=0.06,
            s=6,
            rasterized=True,
        )

        axes[row, 1].scatter(
            data["predicted_mean"],
            residual,
            color=color,
            alpha=0.06,
            s=6,
            rasterized=True,
        )

        temperature_summary = (
            calculate_binned_residual_summary(
                data,
                TEMPERATURE_COLUMN,
            )
        )

        fitted_summary = (
            calculate_binned_residual_summary(
                data,
                "predicted_mean",
            )
        )

        if not temperature_summary.empty:
            axes[row, 0].plot(
                temperature_summary["x"],
                temperature_summary["residual"],
                color="black",
                linewidth=2.0,
            )

            axes[row, 0].fill_between(
                temperature_summary["x"],
                temperature_summary["residual_q25"],
                temperature_summary["residual_q75"],
                color="black",
                alpha=0.10,
            )

        if not fitted_summary.empty:
            axes[row, 1].plot(
                fitted_summary["x"],
                fitted_summary["residual"],
                color="black",
                linewidth=2.0,
            )

            axes[row, 1].fill_between(
                fitted_summary["x"],
                fitted_summary["residual_q25"],
                fitted_summary["residual_q75"],
                color="black",
                alpha=0.10,
            )

        for column in range(2):
            axis = axes[row, column]

            axis.axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=0.8,
            )

            axis.yaxis.set_major_formatter(
                mticker.PercentFormatter(
                    xmax=1.0,
                    decimals=0,
                )
            )

            axis.set_ylabel(
                "Predicted − observed fraction"
            )

            style_axis(axis)

        axes[row, 0].set_xlabel(
            "ERA5 DJF 2 m temperature (°C)"
        )

        axes[row, 1].set_xlabel(
            "Predicted ponded fraction"
        )

        axes[row, 1].xaxis.set_major_formatter(
            mticker.PercentFormatter(
                xmax=1.0,
                decimals=1,
            )
        )

        axes[row, 0].set_title(
            (
                f"{MODEL_LABELS[model_name]}: "
                "residual by temperature"
            )
        )

        axes[row, 1].set_title(
            (
                f"{MODEL_LABELS[model_name]}: "
                "residual by fitted value"
            )
        )

    save_figure(
        figure,
        FIGURE_DIR
        / "Fig_S15_temperature_model_residual_diagnostics",
    )

def plot_shelf_year_heatmaps(shelf_year_tables):
    """
    Plot shelf-year residuals for both models.

    The Peninsula-trained model table is generated by applying its fitted
    temperature response to the full All-Shelves common-support data.
    """

    reference = shelf_year_tables[
        ALL_SHELVES_MODEL
    ]

    # Preserve a meaningful order based on the average observed fraction.
    shelf_order = (
        reference.groupby(
            SHELF_COLUMN,
            observed=True,
        )[OBSERVED_COLUMN]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    years = sorted(
        set().union(
            *[
                set(
                    table[YEAR_COLUMN].astype(int)
                )
                for table
                in shelf_year_tables.values()
            ]
        )
    )

    matrices = {}
    residual_values = []

    for model_name in MODEL_NAMES:
        table = shelf_year_tables[model_name]

        pivot = (
            table.pivot(
                index=SHELF_COLUMN,
                columns=YEAR_COLUMN,
                values=RESIDUAL_COLUMN,
            )
            .reindex(
                index=shelf_order,
                columns=years,
            )
        )

        matrices[model_name] = pivot

        values = pivot.to_numpy(dtype=float)

        residual_values.extend(
            values[np.isfinite(values)]
        )

    residual_values = np.asarray(
        residual_values,
        dtype=float,
    )

    residual_limit = max(
        float(
            np.quantile(
                np.abs(residual_values),
                0.98,
            )
        ),
        1.0e-6,
    )

    norm = mcolors.TwoSlopeNorm(
        vmin=-residual_limit,
        vcenter=0.0,
        vmax=residual_limit,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 11.0),
        constrained_layout=True,
        sharey=True,
    )

    image = None

    for axis, model_name in zip(
        axes,
        MODEL_NAMES,
    ):
        image = axis.imshow(
            matrices[model_name].to_numpy(
                dtype=float
            ),
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
        )

        axis.set_xticks(
            np.arange(len(years))
        )

        axis.set_xticklabels(
            years,
            rotation=90,
        )

        axis.set_yticks(
            np.arange(len(shelf_order))
        )

        axis.set_yticklabels(
            shelf_order,
            fontsize=7,
        )

        axis.set_xlabel("Year")
        axis.set_title(
            MODEL_LABELS[model_name]
        )

    axes[0].set_ylabel("Ice shelf")

    colorbar = figure.colorbar(
        image,
        ax=axes.tolist(),
        fraction=0.025,
        pad=0.02,
    )

    colorbar.set_label(
        "Predicted − observed ponded fraction"
    )

    colorbar.ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(
            xmax=1.0,
            decimals=1,
        )
    )

    save_figure(
        figure,
        FIGURE_DIR
        / "Fig_S16_temperature_model_shelf_year_residuals",
    )


def make_parameter_summary(models):
    tables = []

    for model_name in MODEL_NAMES:
        model = models[model_name]

        variables = [
            model["resolved_variables"]["alpha"],
            model["resolved_variables"]["beta"],
        ]

        if "kappa" in model["resolved_variables"]:
            variables.append(
                model["resolved_variables"]["kappa"]
            )

        summary = az.summary(
            model["idata"],
            var_names=variables,
            hdi_prob=0.95,
            round_to=None,
        ).reset_index()

        summary = summary.rename(
            columns={"index": "parameter"}
        )

        summary.insert(
            0,
            "model_name",
            model_name,
        )

        summary.insert(
            1,
            "model_label",
            model["model_label"],
        )

        tables.append(summary)

    return pd.concat(
        tables,
        ignore_index=True,
    )


def make_model_metadata_table(models):
    rows = []

    for model_name in MODEL_NAMES:
        model = models[model_name]

        rows.append(
            {
                "model_name": model_name,
                "model_label": model["model_label"],
                "trace_path": str(
                    model["trace_path"]
                ),
                "training_data_path": str(
                    model["input_path"]
                ),
                "metadata_path": str(
                    model["metadata_path"]
                ),
                "alpha_variable": (
                    model["resolved_variables"]["alpha"]
                ),
                "beta_variable": (
                    model["resolved_variables"]["beta"]
                ),
                "kappa_variable": (
                    model["resolved_variables"].get(
                        "kappa",
                        "",
                    )
                ),
                "temperature_mean_for_standardization_c": (
                    model["temperature_mean"]
                ),
                "temperature_sd_for_standardization_c": (
                    model["temperature_sd"]
                ),
                "n_training_rows": int(
                    len(model["training_data"])
                ),
                "n_training_shelves": int(
                    model["training_data"][
                        SHELF_COLUMN
                    ].nunique()
                ),
                "n_evaluation_rows": int(
                    len(model["evaluation_data"])
                ),
                "n_evaluation_shelves": int(
                    model["evaluation_data"][
                        SHELF_COLUMN
                    ].nunique()
                ),
                "posterior_draws_available": (
                    model["posterior_draws_available"]
                ),
                "posterior_draws_used": (
                    model["posterior_draws_used"]
                ),
                "shared_temperature_standardization": (
                    SHARED_TEMPERATURE_STANDARDIZATION
                ),
                "residual_definition": (
                    "predicted_minus_observed"
                ),
            }
        )

    return pd.DataFrame(rows)


# Main Function Call

def main():
    print("\n" + "=" * 88)
    print("BUILD SI FIGURES S10–S16 FROM NEW MODEL TRACES")
    print("=" * 88)

    print(
        "Model results directory:",
        MODEL_RESULTS_DIR.resolve(),
        flush=True,
    )

    print(
        "All-Shelves trace:",
        ALL_SHELVES_TRACE,
        flush=True,
    )

    print(
        "Peninsula trace:",
        PENINSULA_TRACE,
        flush=True,
    )

    print(
        "All-Shelves input:",
        ALL_SHELVES_INPUT_REQUESTED,
        flush=True,
    )

    print(
        "Peninsula input:",
        PENINSULA_INPUT_REQUESTED,
        flush=True,
    )

    print(
        "Output directory:",
        SI_OUTPUT_DIR.resolve(),
        flush=True,
    )

    print(
        "Residual definition: predicted - observed",
        flush=True,
    )

    models = load_models()

    metadata_table = make_model_metadata_table(
        models
    )

    parameter_summary = make_parameter_summary(
        models
    )

    atomic_csv_write(
        metadata_table,
        OUTPUT_TABLE_DIR
        / "temperature_model_metadata.csv",
    )

    atomic_csv_write(
        parameter_summary,
        OUTPUT_TABLE_DIR
        / "temperature_model_parameter_summary.csv",
    )

    # Figures S10–S13

    plot_trace_figure(
        models[ALL_SHELVES_MODEL],
        figure_number=10,
        basename=(
            "Fig_S10_all_shelves_"
            "beta_binomial_trace"
        ),
    )

    plot_rank_figure(
        models[ALL_SHELVES_MODEL],
        figure_number=11,
        basename=(
            "Fig_S11_all_shelves_"
            "beta_binomial_rank"
        ),
    )

    plot_trace_figure(
        models[PENINSULA_MODEL],
        figure_number=12,
        basename=(
            "Fig_S12_peninsula_"
            "binomial_trace"
        ),
    )

    plot_rank_figure(
        models[PENINSULA_MODEL],
        figure_number=13,
        basename=(
            "Fig_S13_peninsula_"
            "binomial_rank"
        ),
    )

    training_prediction_tables = {}
    calibration_tables = {}

    for model_name in MODEL_NAMES:
        training_predictions = (
            calculate_row_predictions(
                models[model_name],
                models[model_name]["training_data"],
            )
        )

        training_prediction_tables[
            model_name
        ] = training_predictions

        calibration_tables[
            model_name
        ] = calculate_calibration(
            training_predictions
        )

        save_large_table(
            training_predictions,
            OUTPUT_TABLE_DIR
            / (
                f"{model_name}_"
                "training_row_predictions"
            ),
        )

        atomic_csv_write(
            calibration_tables[model_name],
            OUTPUT_TABLE_DIR
            / (
                f"{model_name}_"
                "training_calibration.csv"
            ),
        )

    plot_calibration(
        calibration_tables
    )

    # Figs. S15 and S16:

    all_shelf_prediction_tables = {}
    shelf_year_tables = {}
    fit_metric_rows = []

    all_shelf_data = models[
        ALL_SHELVES_MODEL
    ]["training_data"]

    for model_name in MODEL_NAMES:
        predictions = calculate_row_predictions(
            models[model_name],
            all_shelf_data,
        )

        all_shelf_prediction_tables[
            model_name
        ] = predictions

        shelf_year = (
            aggregate_shelf_year_residuals(
                predictions
            )
        )

        shelf_year_tables[
            model_name
        ] = shelf_year

        fit_metric_rows.append(
            calculate_fit_metrics(
                predictions
            )
        )

        save_large_table(
            predictions,
            OUTPUT_TABLE_DIR
            / (
                f"{model_name}_"
                "all_shelves_row_predictions"
            ),
        )

        atomic_csv_write(
            shelf_year,
            OUTPUT_TABLE_DIR
            / (
                f"{model_name}_"
                "all_shelves_shelf_year_residuals.csv"
            ),
        )

    fit_metrics = pd.DataFrame(
        fit_metric_rows
    )

    combined_shelf_year = pd.concat(
        shelf_year_tables.values(),
        ignore_index=True,
    )

    atomic_csv_write(
        fit_metrics,
        OUTPUT_TABLE_DIR
        / "all_shelves_model_fit_metrics.csv",
    )

    atomic_csv_write(
        combined_shelf_year,
        OUTPUT_TABLE_DIR
        / "all_models_shelf_year_residuals.csv",
    )

    plot_residuals(
        all_shelf_prediction_tables
    )

    plot_shelf_year_heatmaps(
        shelf_year_tables
    )

    metadata = {
        "model_results_directory": str(
            MODEL_RESULTS_DIR.resolve()
        ),
        "all_shelves_trace": str(
            ALL_SHELVES_TRACE.resolve()
        ),
        "peninsula_trace": str(
            PENINSULA_TRACE.resolve()
        ),
        "all_shelves_input": str(
            models[ALL_SHELVES_MODEL][
                "input_path"
            ].resolve()
        ),
        "peninsula_input": str(
            models[PENINSULA_MODEL][
                "input_path"
            ].resolve()
        ),
        "residual_definition": (
            "predicted_fraction - observed_fraction"
        ),
        "positive_residual_interpretation": (
            "model overprediction"
        ),
        "negative_residual_interpretation": (
            "model underprediction"
        ),
        "peninsula_model_s15_evaluation_scope": (
            "all_shelves_common_support_records"
        ),
        "peninsula_model_s16_evaluation_scope": (
            "all_shelves_common_support_records"
        ),
        "shared_temperature_standardization": (
            SHARED_TEMPERATURE_STANDARDIZATION
        ),
        "maximum_posterior_draws_setting": (
            MAX_POSTERIOR_DRAWS
        ),
        "random_seed": RANDOM_SEED,
        "figures": sorted(
            path.name
            for path in FIGURE_DIR.glob("*")
            if path.is_file()
        ),
        "tables": sorted(
            path.name
            for path in OUTPUT_TABLE_DIR.glob("*")
            if path.is_file()
        ),
    }

    atomic_json_write(
        metadata,
        SI_OUTPUT_DIR
        / "model_si_metadata.json",
    )

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)

    print(
        "Figures:",
        FIGURE_DIR.resolve(),
        flush=True,
    )

    print(
        "Tables:",
        OUTPUT_TABLE_DIR.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    main()