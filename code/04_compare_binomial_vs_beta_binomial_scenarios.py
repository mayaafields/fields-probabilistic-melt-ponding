#!/usr/bin/env python
# coding: utf-8

"""
Compare Bayesian Binomial and Beta-Binomial temperature-response models.

The two likelihoods are fit using exactly the same response data and
temperature predictor so their differences reflect the likelihood assumption.

Supported observational supports
---------------------------------
MODEL_SUPPORT=row

    Each observation is a shelf × climate-cell × year unit:

        y_i = ponded 30 m pixels
        n_i = valid 30 m pixels
        y_i ~ Binomial(n_i, mu_i)

    or:

        y_i ~ BetaBinomial(
            n_i,
            mu_i * kappa,
            (1 - mu_i) * kappa
        )

MODEL_SUPPORT=raw_aggregate

    Shelf-year-cell observations are grouped into temperature bins:

        Y_b = sum(y_i in bin b)
        N_b = sum(n_i in bin b)

        Y_b ~ Binomial(N_b, mu_b)

    or:

        Y_b ~ BetaBinomial(
            N_b,
            mu_b * kappa,
            (1 - mu_b) * kappa
        )

For both supports:

    cloglog(mu) = alpha + beta_temp * temp_z

Supported CASE_NAME values
--------------------------
all_shelves
all_no_amery
antarctic_peninsula
amery
weddell_sea
ross_sea
east_antarctica_no_amery
east_antarctica_no_amery_lon_0_90
east_antarctica_no_amery_lon_90_180

Examples
--------
Conservative row-level comparison:

    PRODUCT=ERA5 \
    CASE_NAME=all_no_amery \
    MODEL_SUPPORT=row \
    DRAWS=300 TUNE=300 CHAINS=2 CORES=2 \
    python 04_compare_binomial_vs_beta_binomial_scenarios.py

Antarctic Peninsula aggregate comparison:

    PRODUCT=ERA5 \
    CASE_NAME=antarctic_peninsula \
    MODEL_SUPPORT=raw_aggregate \
    DRAWS=300 TUNE=300 CHAINS=2 CORES=2 \
    python 04_compare_binomial_vs_beta_binomial_scenarios.py
"""

# =============================================================================
# Imports
# =============================================================================

import os
import json
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from bb_model_utils import read_table, diagnostics_count_table

warnings.filterwarnings("ignore")


# =============================================================================
# Plot style
# =============================================================================

COLOR_BINOMIAL = "#2b7bba"
COLOR_BETA_BINOMIAL = "#d95f02"
COLOR_OBSERVED = "black"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "0.2",
        "grid.color": "0.86",
        "grid.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# General helpers
# =============================================================================

def env_bool(name, default="0"):
    return os.environ.get(name, default).strip() == "1"


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    pdf_path = os.path.splitext(path)[0] + ".pdf"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("[SAVED]", path, flush=True)
    print("[SAVED]", pdf_path, flush=True)


def save_idata_safely(idata, path, overwrite=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    root, ext = os.path.splitext(path)

    if not ext:
        ext = ".nc"
        path = root + ext

    if os.path.exists(path):
        if overwrite:
            backup = (
                f"{root}.backup_"
                f"{time.strftime('%Y%m%d_%H%M%S')}{ext}"
            )
            os.replace(path, backup)
        else:
            path = (
                f"{root}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}{ext}"
            )

    temporary = (
        f"{root}.tmp_{os.getpid()}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    )

    az.to_netcdf(idata, temporary)
    os.replace(temporary, path)

    print("[SAVED]", path, flush=True)

    return path


def flatten_posterior(trace, variable):
    values = trace.posterior[variable].values

    if values.ndim == 2:
        return values.reshape(-1)

    return values.reshape((-1,) + values.shape[2:])


def thin_draw_indices(n_draws, max_draws, seed=42):
    if max_draws <= 0 or n_draws <= max_draws:
        return np.arange(n_draws)

    rng = np.random.default_rng(seed)

    return np.sort(
        rng.choice(
            n_draws,
            size=max_draws,
            replace=False,
        )
    )


def normalize_label(value):
    value = str(value).lower().strip()

    chars = []
    depth = 0

    for char in value:
        if char == "(":
            depth += 1
            continue

        if char == ")":
            depth = max(depth - 1, 0)
            continue

        if depth == 0:
            chars.append(char)

    value = "".join(chars)
    value = " " + value + " "

    for phrase in [
        " ice shelves ",
        " ice shelf ",
        " shelves ",
        " shelf ",
        " glacier ",
    ]:
        value = value.replace(phrase, " ")

    value = "".join(
        char if char.isalnum() else " "
        for char in value
    )

    return " ".join(value.split())


def resolve_one_label(requested, available, label_type="label"):
    requested_normalized = normalize_label(requested)
    available = [str(value) for value in available]

    exact = [
        value for value in available
        if value == str(requested)
    ]

    if exact:
        return exact[0]

    matches = [
        value for value in available
        if normalize_label(value) == requested_normalized
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ValueError(
            f"Could not resolve {label_type} {requested!r}. "
            f"Available values: {available}"
        )

    raise ValueError(
        f"Ambiguous {label_type} {requested!r}: {matches}"
    )


# =============================================================================
# Product configuration
# =============================================================================

def get_product_config(product):
    product = product.upper().strip()

    if product == "ERA5":
        return {
            "product": "ERA5",
            "product_lower": "era5",
            "default_table": os.path.join(
                "corrected_crop_and_era5_alignment_outputs",
                "observations_ERA5temp_ponding_COMMON_SUPPORT_2006_2020.parquet",
            ),
            "table_env": "ERA5_TABLE",
            "cell_col": "era5_cell_id",
            "temp_candidates": [
                "era5_t2m_djf_c",
                "era5_t2m_djf",
            ],
            "default_temp_col": "era5_t2m_djf_c",
            "temp_label": "ERA5 DJF 2 m temperature (°C)",
        }

    if product == "RACMO":
        return {
            "product": "RACMO",
            "product_lower": "racmo",
            "default_table": os.path.join(
                "corrected_crop_and_racmo_alignment_outputs",
                "observations_RACMOtemp_ponding_COMMON_SUPPORT_2006_2020.parquet",
            ),
            "table_env": "RACMO_TABLE",
            "cell_col": "racmo_cell_id",
            "temp_candidates": [
                "racmo_t2m_djf_c",
                "racmo_t2m_djf",
                "t2m_djf_c",
                "t2m",
            ],
            "default_temp_col": "racmo_t2m_djf_c",
            "temp_label": "RACMO DJF 2 m temperature (°C)",
        }

    raise ValueError("PRODUCT must be ERA5 or RACMO.")


def choose_temperature_column(df, config, requested=None):
    if requested and requested in df.columns:
        return requested

    for column in config["temp_candidates"]:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find a temperature column. "
        f"Available columns: {list(df.columns)}"
    )


def choose_region_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"REGION_COL={requested} not found."
            )
        return requested

    candidates = [
        "region",
        "antarctic_region",
        "basin",
        "basin_name",
        "imbie_region",
        "region_name",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError("No region column was found.")


def choose_lon_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"LON_COL={requested} not found."
            )
        return requested

    candidates = [
        "lon",
        "longitude",
        "cell_lon",
        "cell_longitude",
        "era5_lon",
        "era5_longitude",
        "era5_cell_lon",
        "era5_cell_center_lon",
        "racmo_lon",
        "racmo_longitude",
        "racmo_cell_lon",
        "racmo_cell_center_lon",
        "centroid_lon",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Longitude filtering requested, but no longitude "
        "column was found. Set LON_COL explicitly."
    )


# =============================================================================
# Validation and filtering
# =============================================================================

def validate_table(
    df,
    *,
    config,
    temp_col,
    year_min,
    year_max,
    min_coverage_fraction,
):
    d = df.copy()

    required = [
        "shelf",
        "year",
        config["cell_col"],
        temp_col,
        "n_pixels_30m",
        "y_ponded_pixels",
    ]

    missing = [
        column for column in required
        if column not in d.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    d["year"] = pd.to_numeric(
        d["year"],
        errors="coerce",
    )

    for column in [
        temp_col,
        "n_pixels_30m",
        "y_ponded_pixels",
    ]:
        d[column] = pd.to_numeric(
            d[column],
            errors="coerce",
        )

    keep = (
        np.isfinite(d["year"])
        & (d["year"] >= year_min)
        & (d["year"] <= year_max)
        & np.isfinite(d[temp_col])
        & np.isfinite(d["n_pixels_30m"])
        & np.isfinite(d["y_ponded_pixels"])
        & (d["n_pixels_30m"] > 0)
        & (d["y_ponded_pixels"] >= 0)
        & (
            d["y_ponded_pixels"]
            <= d["n_pixels_30m"]
        )
    )

    if "effective_overlap_area_m2" in d.columns:
        keep &= np.isfinite(
            d["effective_overlap_area_m2"]
        )
        keep &= d["effective_overlap_area_m2"] > 0

    if "coverage_fraction_of_shelf_cell" in d.columns:
        keep &= np.isfinite(
            d["coverage_fraction_of_shelf_cell"]
        )
        keep &= (
            d["coverage_fraction_of_shelf_cell"]
            >= min_coverage_fraction
        )

    d = d[keep].copy()

    if d.empty:
        raise RuntimeError(
            "No observations remain after validation."
        )

    d["year"] = d["year"].astype(int)
    d["n_pixels_30m"] = d[
        "n_pixels_30m"
    ].astype("int64")
    d["y_ponded_pixels"] = d[
        "y_ponded_pixels"
    ].astype("int64")

    d["ponding_frac"] = (
        d["y_ponded_pixels"]
        / d["n_pixels_30m"]
    )

    return d


def correct_pine_island_region(
    df,
    region_col,
    corrected_region,
):
    d = df.copy()

    mask = d["shelf"].astype(str).str.contains(
        "Pine Island",
        case=False,
        na=False,
    )

    if mask.any():
        print(
            "[REGION CORRECTION] Pine Island ->",
            corrected_region,
            flush=True,
        )

        d.loc[mask, region_col] = corrected_region

    return d


def filter_region(df, region_col, region):
    matched = resolve_one_label(
        region,
        sorted(df[region_col].astype(str).unique()),
        label_type="region",
    )

    out = df[
        df[region_col].astype(str) == matched
    ].copy()

    if out.empty:
        raise RuntimeError(
            f"No rows remain for region={matched}"
        )

    return out


def filter_lon_range(df, lon_col, lower, upper):
    lon = pd.to_numeric(
        df[lon_col],
        errors="coerce",
    ) % 360.0

    out = df[
        np.isfinite(lon)
        & (lon >= lower)
        & (lon < upper)
    ].copy()

    if out.empty:
        raise RuntimeError(
            f"No rows remain for longitude {lower}–{upper}."
        )

    return out


def apply_case_filter(
    df,
    *,
    case_name,
    region_col,
    lon_col_requested=None,
):
    d = df.copy()
    case_name = case_name.lower().strip()

    if case_name == "all_shelves":
        return d

    if case_name == "all_no_amery":
        return d[
            ~d["shelf"].astype(str).str.contains(
                "Amery",
                case=False,
                na=False,
            )
        ].copy()

    if case_name == "antarctic_peninsula":
        return filter_region(
            d,
            region_col,
            "Antarctic Peninsula",
        )

    if case_name == "amery":
        return d[
            d["shelf"].astype(str).str.contains(
                "Amery",
                case=False,
                na=False,
            )
        ].copy()

    if case_name == "weddell_sea":
        return filter_region(
            d,
            region_col,
            "Weddell Sea",
        )

    if case_name == "ross_sea":
        return filter_region(
            d,
            region_col,
            "Ross Sea",
        )

    if case_name in {
        "east_antarctica_no_amery",
        "east_antarctica_no_amery_lon_0_90",
        "east_antarctica_no_amery_lon_90_180",
    }:
        available = sorted(
            d[region_col].astype(str).unique()
        )

        normalized = {
            normalize_label(value): value
            for value in available
        }

        target = normalize_label(
            "East Antarctica excluding Amery"
        )

        if target in normalized:
            d = d[
                d[region_col].astype(str)
                == normalized[target]
            ].copy()
        else:
            d = filter_region(
                d,
                region_col,
                "East Antarctica",
            )

        d = d[
            ~d["shelf"].astype(str).str.contains(
                "Amery",
                case=False,
                na=False,
            )
        ].copy()

        if case_name.endswith("lon_0_90"):
            lon_col = choose_lon_column(
                d,
                lon_col_requested,
            )
            d = filter_lon_range(
                d,
                lon_col,
                0.0,
                90.0,
            )

        elif case_name.endswith("lon_90_180"):
            lon_col = choose_lon_column(
                d,
                lon_col_requested,
            )
            d = filter_lon_range(
                d,
                lon_col,
                90.0,
                180.0,
            )

        return d

    raise ValueError(
        f"Unsupported CASE_NAME={case_name}"
    )


def standardize_temperature(df, temp_col):
    d = df.copy()

    mean = float(d[temp_col].mean())
    sd = float(d[temp_col].std(ddof=0))

    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0

    d["temp_z"] = (
        d[temp_col] - mean
    ) / sd

    return d, mean, sd


def make_temperature_bins(
    df,
    temp_col,
    *,
    n_bins,
):
    d = df.copy()

    d["_temp_bin"] = pd.qcut(
        d[temp_col],
        q=min(n_bins, len(d)),
        duplicates="drop",
    )

    d["_bin_code"] = d["_temp_bin"].cat.codes
    d = d[d["_bin_code"] >= 0].copy()

    rows = []

    for code, group in d.groupby(
        "_bin_code",
        observed=True,
    ):
        n = float(group["n_pixels_30m"].sum())
        y = float(group["y_ponded_pixels"].sum())

        if n <= 0:
            continue

        rows.append(
            {
                "bin_index": int(code),
                "bin_label": str(
                    group["_temp_bin"].iloc[0]
                ),
                "temperature_mean": float(
                    np.average(
                        group[temp_col],
                        weights=group["n_pixels_30m"],
                    )
                ),
                "temperature_min": float(
                    group[temp_col].min()
                ),
                "temperature_max": float(
                    group[temp_col].max()
                ),
                "n_rows": int(len(group)),
                "n_pixels_30m": int(round(n)),
                "y_ponded_pixels": int(round(y)),
                "observed_frac": y / n,
                "observed_percent": 100.0 * y / n,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("temperature_mean")
        .reset_index(drop=True)
    )


# =============================================================================
# Link functions
# =============================================================================

def inv_cloglog_pm(eta):
    eta = pm.math.clip(
        eta,
        -30.0,
        20.0,
    )

    return 1.0 - pm.math.exp(
        -pm.math.exp(eta)
    )


def inv_cloglog_np(eta):
    eta = np.clip(
        np.asarray(eta, dtype=float),
        -30.0,
        20.0,
    )

    return 1.0 - np.exp(
        -np.exp(eta)
    )


# =============================================================================
# Models
# =============================================================================

def build_binomial_model(
    y,
    n,
    temp_z,
    *,
    dim_name,
):
    coords = {
        dim_name: np.arange(len(y)),
    }

    with pm.Model(coords=coords) as model:

        n_data = pm.Data(
            "n",
            np.asarray(n, dtype="int64"),
            dims=dim_name,
        )

        temp_data = pm.Data(
            "temp_z",
            np.asarray(temp_z, dtype=float),
            dims=dim_name,
        )

        alpha = pm.Normal(
            "alpha",
            mu=-6.0,
            sigma=3.0,
        )

        beta_temp = pm.Normal(
            "beta_temp",
            mu=0.0,
            sigma=1.5,
        )

        eta = alpha + beta_temp * temp_data

        mu = pm.Deterministic(
            "mu",
            inv_cloglog_pm(eta),
            dims=dim_name,
        )

        pm.Binomial(
            "y",
            n=n_data,
            p=mu,
            observed=np.asarray(y, dtype="int64"),
            dims=dim_name,
        )

    return model


def build_beta_binomial_model(
    y,
    n,
    temp_z,
    *,
    dim_name,
    kappa_alpha,
    kappa_beta,
):
    coords = {
        dim_name: np.arange(len(y)),
    }

    with pm.Model(coords=coords) as model:

        n_data = pm.Data(
            "n",
            np.asarray(n, dtype="int64"),
            dims=dim_name,
        )

        temp_data = pm.Data(
            "temp_z",
            np.asarray(temp_z, dtype=float),
            dims=dim_name,
        )

        alpha = pm.Normal(
            "alpha",
            mu=-6.0,
            sigma=3.0,
        )

        beta_temp = pm.Normal(
            "beta_temp",
            mu=0.0,
            sigma=1.5,
        )

        eta = alpha + beta_temp * temp_data

        mu = pm.Deterministic(
            "mu",
            inv_cloglog_pm(eta),
            dims=dim_name,
        )

        kappa = pm.Gamma(
            "kappa",
            alpha=kappa_alpha,
            beta=kappa_beta,
        )

        pm.BetaBinomial(
            "y",
            n=n_data,
            alpha=mu * kappa,
            beta=(1.0 - mu) * kappa,
            observed=np.asarray(y, dtype="int64"),
            dims=dim_name,
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
    random_seed,
    compute_log_likelihood,
):
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={
                "log_likelihood": compute_log_likelihood,
            },
        )

    return trace


# =============================================================================
# Summaries and curves
# =============================================================================

def summarize_trace(trace, model_name):
    rows = []

    variables = [
        "alpha",
        "beta_temp",
    ]

    if "kappa" in trace.posterior:
        variables.append("kappa")

    for variable in variables:
        values = flatten_posterior(
            trace,
            variable,
        )

        quantiles = np.quantile(
            values,
            [0.025, 0.5, 0.975],
        )

        rows.append(
            {
                "model": model_name,
                "parameter": variable,
                "mean": float(np.mean(values)),
                "sd": float(
                    np.std(values, ddof=1)
                ),
                "q025": float(quantiles[0]),
                "median": float(quantiles[1]),
                "q975": float(quantiles[2]),
                "p_gt_0": (
                    float(np.mean(values > 0))
                    if variable == "beta_temp"
                    else np.nan
                ),
            }
        )

    if "kappa" in trace.posterior:
        kappa = flatten_posterior(
            trace,
            "kappa",
        )

        rho = 1.0 / (1.0 + kappa)

        quantiles = np.quantile(
            rho,
            [0.025, 0.5, 0.975],
        )

        rows.append(
            {
                "model": model_name,
                "parameter": (
                    "rho_1_over_1_plus_kappa"
                ),
                "mean": float(np.mean(rho)),
                "sd": float(
                    np.std(rho, ddof=1)
                ),
                "q025": float(quantiles[0]),
                "median": float(quantiles[1]),
                "q975": float(quantiles[2]),
                "p_gt_0": np.nan,
            }
        )

    return pd.DataFrame(rows)


def posterior_curve(
    trace,
    temp_grid,
    *,
    temp_mean,
    temp_sd,
    model_name,
    max_draws,
    seed,
):
    alpha = flatten_posterior(
        trace,
        "alpha",
    )

    beta = flatten_posterior(
        trace,
        "beta_temp",
    )

    indices = thin_draw_indices(
        len(alpha),
        max_draws,
        seed=seed,
    )

    alpha = alpha[indices]
    beta = beta[indices]

    temp_z = (
        np.asarray(temp_grid, dtype=float)
        - temp_mean
    ) / temp_sd

    eta = (
        alpha[:, None]
        + beta[:, None] * temp_z[None, :]
    )

    mu = inv_cloglog_np(eta)

    output = pd.DataFrame(
        {
            "model": model_name,
            "temperature": temp_grid,
            "mu_mean": np.mean(mu, axis=0),
            "mu_q025": np.quantile(
                mu,
                0.025,
                axis=0,
            ),
            "mu_median": np.quantile(
                mu,
                0.500,
                axis=0,
            ),
            "mu_q975": np.quantile(
                mu,
                0.975,
                axis=0,
            ),
        }
    )

    for column in [
        "mu_mean",
        "mu_q025",
        "mu_median",
        "mu_q975",
    ]:
        output[
            column + "_percent"
        ] = 100.0 * output[column]

    return output


# =============================================================================
# Historical fit diagnostics
# =============================================================================

def predicted_mu_at_observations(trace, max_draws, seed):
    alpha = flatten_posterior(
        trace,
        "alpha",
    )

    beta = flatten_posterior(
        trace,
        "beta_temp",
    )

    indices = thin_draw_indices(
        len(alpha),
        max_draws,
        seed=seed,
    )

    return (
        alpha[indices],
        beta[indices],
    )


def evaluate_binned_fit(
    trace,
    binned,
    *,
    max_draws,
    seed,
):
    alpha, beta = predicted_mu_at_observations(
        trace,
        max_draws,
        seed,
    )

    temp_z = binned[
        "temp_z"
    ].to_numpy(dtype=float)

    eta = (
        alpha[:, None]
        + beta[:, None] * temp_z[None, :]
    )

    mu = inv_cloglog_np(eta)

    observed = binned[
        "observed_frac"
    ].to_numpy(dtype=float)

    weights = binned[
        "n_pixels_30m"
    ].to_numpy(dtype=float)

    weights = weights / weights.sum()

    median = np.quantile(
        mu,
        0.5,
        axis=0,
    )

    residual = median - observed

    output = binned.copy()

    output["fit_mean"] = np.mean(
        mu,
        axis=0,
    )

    output["fit_q025"] = np.quantile(
        mu,
        0.025,
        axis=0,
    )

    output["fit_median"] = median

    output["fit_q975"] = np.quantile(
        mu,
        0.975,
        axis=0,
    )

    for column in [
        "fit_mean",
        "fit_q025",
        "fit_median",
        "fit_q975",
    ]:
        output[
            column + "_percent"
        ] = 100.0 * output[column]

    metrics = {
        "pixel_weighted_rmse": float(
            np.sqrt(
                np.sum(
                    weights * residual ** 2
                )
            )
        ),
        "pixel_weighted_mae": float(
            np.sum(
                weights * np.abs(residual)
            )
        ),
        "pixel_weighted_bias": float(
            np.sum(
                weights * residual
            )
        ),
    }

    return output, metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(
    *,
    binned,
    curves,
    out_dir,
    prefix,
    temp_label,
):
    fig, ax = plt.subplots(
        figsize=(9.5, 6.2)
    )

    sizes = binned[
        "n_pixels_30m"
    ].to_numpy(dtype=float)

    if np.nanmax(sizes) > 0:
        sizes = sizes / np.nanmax(sizes)

    sizes = (
        45.0
        + 220.0 * np.sqrt(sizes)
    )

    ax.scatter(
        binned["temperature_mean"],
        binned["observed_percent"],
        s=sizes,
        color=COLOR_OBSERVED,
        edgecolor="white",
        linewidth=0.9,
        label="Observed aggregate bin fractions",
        zorder=8,
    )

    model_styles = [
        (
            "binomial",
            COLOR_BINOMIAL,
            "-",
            "Binomial",
        ),
        (
            "beta_binomial",
            COLOR_BETA_BINOMIAL,
            "--",
            "Beta-binomial",
        ),
    ]

    for (
        model_name,
        color,
        linestyle,
        label,
    ) in model_styles:
        subset = curves[
            curves["model"] == model_name
        ].copy()

        if subset.empty:
            continue

        ax.fill_between(
            subset["temperature"],
            subset["mu_q025_percent"],
            subset["mu_q975_percent"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

        ax.plot(
            subset["temperature"],
            subset["mu_median_percent"],
            color=color,
            linestyle=linestyle,
            linewidth=2.8,
            label=label,
        )

    ax.set_xlabel(temp_label)
    ax.set_ylabel(
        "Ponded 30 m-pixel fraction (%)"
    )

    ax.set_title(
        "Binomial versus beta-binomial "
        "temperature response"
    )

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.35)

    ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.3f")
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=True,
        loc="best",
    )

    fig.tight_layout()

    path = os.path.join(
        out_dir,
        f"{prefix}_binomial_vs_beta_binomial.png",
    )

    savefig(fig, path)


# =============================================================================
# Main
# =============================================================================

def main():
    random_seed = int(
        os.environ.get(
            "RANDOM_SEED",
            "42",
        )
    )

    product = os.environ.get(
        "PRODUCT",
        "ERA5",
    )

    config = get_product_config(
        product
    )

    product_name = config["product"]
    product_lower = config["product_lower"]

    case_name = (
        os.environ.get(
            "CASE_NAME",
            "antarctic_peninsula",
        )
        .strip()
        .lower()
    )

    model_support = (
        os.environ.get(
            "MODEL_SUPPORT",
            "raw_aggregate",
        )
        .strip()
        .lower()
    )

    if model_support not in {
        "row",
        "raw_aggregate",
    }:
        raise ValueError(
            "MODEL_SUPPORT must be 'row' or 'raw_aggregate'."
        )

    table_path = os.environ.get(
        config["table_env"],
        config["default_table"],
    )

    # Important: use RACMO_TABLE for RACMO, not ERA5_TABLE.
    temp_col_requested = os.environ.get(
        "TEMP_COL",
        config["default_temp_col"],
    )

    region_col_requested = (
        os.environ.get(
            "REGION_COL",
            "region",
        )
        .strip()
    )

    lon_col_requested = (
        os.environ.get(
            "LON_COL",
            "",
        )
        .strip()
        or None
    )

    year_min = int(
        os.environ.get(
            "OBS_YEAR_MIN",
            "2006",
        )
    )

    year_max = int(
        os.environ.get(
            "OBS_YEAR_MAX",
            "2020",
        )
    )

    min_coverage = float(
        os.environ.get(
            "MIN_COVERAGE_FRACTION",
            "0",
        )
    )

    pine_island_region = os.environ.get(
        "PINE_ISLAND_REGION_CORRECTION",
        "Amundsen Sea Embayment",
    )

    out_base = os.environ.get(
        "MODEL_OUT_DIR",
        (
            "binomial_vs_beta_binomial_"
            "scenario_models"
        ),
    )

    out_dir = os.path.join(
        out_base,
        (
            f"{product_lower}_"
            f"{case_name}_"
            f"{model_support}_2006_2020"
        ),
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    prefix = (
        f"{product_lower}_"
        f"{case_name}_"
        f"{model_support}"
    )

    draws = int(
        os.environ.get(
            "DRAWS",
            "1000",
        )
    )

    tune = int(
        os.environ.get(
            "TUNE",
            "1000",
        )
    )

    chains = int(
        os.environ.get(
            "CHAINS",
            "4",
        )
    )

    cores = int(
        os.environ.get(
            "CORES",
            str(chains),
        )
    )

    target_accept = float(
        os.environ.get(
            "TARGET_ACCEPT",
            "0.95",
        )
    )

    n_temp_bins = int(
        os.environ.get(
            "N_TEMP_BINS",
            "18",
        )
    )

    max_curve_draws = int(
        os.environ.get(
            "MAX_POSTERIOR_DRAWS_DIAG",
            "1000",
        )
    )

    kappa_alpha = float(
        os.environ.get(
            "KAPPA_ALPHA",
            "2.0",
        )
    )

    kappa_beta = float(
        os.environ.get(
            "KAPPA_BETA",
            "0.1",
        )
    )

    compute_log_likelihood = env_bool(
        "LOG_LIKELIHOOD",
        "0",
    )

    overwrite_trace = env_bool(
        "OVERWRITE_TRACE",
        "0",
    )

    print("\n" + "#" * 100)
    print("BINOMIAL VS BETA-BINOMIAL")
    print("#" * 100)
    print("PRODUCT:", product_name)
    print("CASE_NAME:", case_name)
    print("MODEL_SUPPORT:", model_support)
    print("TABLE:", table_path)
    print("OUT_DIR:", out_dir)
    print("DRAWS:", draws)
    print("TUNE:", tune)
    print("CHAINS:", chains)
    print("CORES:", cores)
    print("#" * 100 + "\n")

    # -------------------------------------------------------------------------
    # Load and filter data
    # -------------------------------------------------------------------------

    raw = read_table(
        table_path
    )

    temp_col = choose_temperature_column(
        raw,
        config,
        temp_col_requested,
    )

    region_col = choose_region_column(
        raw,
        region_col_requested,
    )

    data = validate_table(
        raw,
        config=config,
        temp_col=temp_col,
        year_min=year_min,
        year_max=year_max,
        min_coverage_fraction=min_coverage,
    )

    data = correct_pine_island_region(
        data,
        region_col,
        pine_island_region,
    )

    data = apply_case_filter(
        data,
        case_name=case_name,
        region_col=region_col,
        lon_col_requested=lon_col_requested,
    )

    if data.empty:
        raise RuntimeError(
            "Scenario filter produced an empty dataframe."
        )

    diagnostics_count_table(
        data,
        temp_col=temp_col,
        cell_col=config["cell_col"],
        name=(
            f"{product_name} {case_name}"
        ),
    )

    data, temp_mean, temp_sd = standardize_temperature(
        data,
        temp_col,
    )

    data_path = os.path.join(
        out_dir,
        f"{prefix}_model_dataframe.parquet",
    )

    data.to_parquet(
        data_path,
        index=False,
    )

    # Descriptive bins are always created for plotting.
    binned = make_temperature_bins(
        data,
        temp_col,
        n_bins=n_temp_bins,
    )

    binned["temp_z"] = (
        binned["temperature_mean"]
        - temp_mean
    ) / temp_sd

    binned_path = os.path.join(
        out_dir,
        f"{prefix}_observed_temperature_bins.csv",
    )

    binned.to_csv(
        binned_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Select the fitting support
    # -------------------------------------------------------------------------

    if model_support == "row":
        y = data[
            "y_ponded_pixels"
        ].to_numpy(dtype="int64")

        n = data[
            "n_pixels_30m"
        ].to_numpy(dtype="int64")

        temp_z = data[
            "temp_z"
        ].to_numpy(dtype=float)

        dim_name = "obs"

    else:
        y = binned[
            "y_ponded_pixels"
        ].to_numpy(dtype="int64")

        n = binned[
            "n_pixels_30m"
        ].to_numpy(dtype="int64")

        temp_z = binned[
            "temp_z"
        ].to_numpy(dtype=float)

        dim_name = "bin"

    # -------------------------------------------------------------------------
    # Fit binomial
    # -------------------------------------------------------------------------

    binomial_model = build_binomial_model(
        y,
        n,
        temp_z,
        dim_name=dim_name,
    )

    binomial_trace = sample_model(
        binomial_model,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
        compute_log_likelihood=compute_log_likelihood,
    )

    binomial_trace_path = save_idata_safely(
        binomial_trace,
        os.path.join(
            out_dir,
            f"{prefix}_binomial_trace.nc",
        ),
        overwrite=overwrite_trace,
    )

    # -------------------------------------------------------------------------
    # Fit beta-binomial
    # -------------------------------------------------------------------------

    beta_binomial_model = build_beta_binomial_model(
        y,
        n,
        temp_z,
        dim_name=dim_name,
        kappa_alpha=kappa_alpha,
        kappa_beta=kappa_beta,
    )

    beta_binomial_trace = sample_model(
        beta_binomial_model,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed + 1,
        compute_log_likelihood=compute_log_likelihood,
    )

    beta_binomial_trace_path = save_idata_safely(
        beta_binomial_trace,
        os.path.join(
            out_dir,
            f"{prefix}_beta_binomial_trace.nc",
        ),
        overwrite=overwrite_trace,
    )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------

    summary = pd.concat(
        [
            summarize_trace(
                binomial_trace,
                "binomial",
            ),
            summarize_trace(
                beta_binomial_trace,
                "beta_binomial",
            ),
        ],
        ignore_index=True,
    )

    summary_path = os.path.join(
        out_dir,
        f"{prefix}_model_summary.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("[SAVED]", summary_path)
    print(summary.to_string(index=False))

    az.summary(
        binomial_trace,
        var_names=[
            "alpha",
            "beta_temp",
        ],
    ).to_csv(
        os.path.join(
            out_dir,
            f"{prefix}_binomial_az_summary.csv",
        )
    )

    az.summary(
        beta_binomial_trace,
        var_names=[
            "alpha",
            "beta_temp",
            "kappa",
        ],
    ).to_csv(
        os.path.join(
            out_dir,
            f"{prefix}_beta_binomial_az_summary.csv",
        )
    )

    # -------------------------------------------------------------------------
    # Curves
    # -------------------------------------------------------------------------

    temp_grid = np.linspace(
        float(data[temp_col].min()),
        float(data[temp_col].max()),
        500,
    )

    binomial_curve = posterior_curve(
        binomial_trace,
        temp_grid,
        temp_mean=temp_mean,
        temp_sd=temp_sd,
        model_name="binomial",
        max_draws=max_curve_draws,
        seed=random_seed,
    )

    beta_binomial_curve = posterior_curve(
        beta_binomial_trace,
        temp_grid,
        temp_mean=temp_mean,
        temp_sd=temp_sd,
        model_name="beta_binomial",
        max_draws=max_curve_draws,
        seed=random_seed,
    )

    curves = pd.concat(
        [
            binomial_curve,
            beta_binomial_curve,
        ],
        ignore_index=True,
    )

    curves_path = os.path.join(
        out_dir,
        f"{prefix}_posterior_curves.csv",
    )

    curves.to_csv(
        curves_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Fit diagnostics on the displayed bins
    # -------------------------------------------------------------------------

    binomial_binned_fit, binomial_metrics = evaluate_binned_fit(
        binomial_trace,
        binned,
        max_draws=max_curve_draws,
        seed=random_seed,
    )

    beta_binomial_binned_fit, beta_binomial_metrics = evaluate_binned_fit(
        beta_binomial_trace,
        binned,
        max_draws=max_curve_draws,
        seed=random_seed + 1,
    )

    binomial_binned_fit[
        "model"
    ] = "binomial"

    beta_binomial_binned_fit[
        "model"
    ] = "beta_binomial"

    binned_fits = pd.concat(
        [
            binomial_binned_fit,
            beta_binomial_binned_fit,
        ],
        ignore_index=True,
    )

    binned_fits_path = os.path.join(
        out_dir,
        f"{prefix}_binned_fitted_values.csv",
    )

    binned_fits.to_csv(
        binned_fits_path,
        index=False,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "binomial",
                **binomial_metrics,
            },
            {
                "model": "beta_binomial",
                **beta_binomial_metrics,
            },
        ]
    )

    metrics_path = os.path.join(
        out_dir,
        f"{prefix}_historical_fit_metrics.csv",
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    print("[SAVED]", metrics_path)
    print(metrics.to_string(index=False))

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    plot_comparison(
        binned=binned,
        curves=curves,
        out_dir=out_dir,
        prefix=prefix,
        temp_label=config["temp_label"],
    )

    # -------------------------------------------------------------------------
    # Optional information-criterion comparison
    # -------------------------------------------------------------------------

    information_criteria = {}

    if compute_log_likelihood:
        try:
            loo_binomial = az.loo(
                binomial_trace,
            )

            loo_beta_binomial = az.loo(
                beta_binomial_trace,
            )

            information_criteria = {
                "binomial_elpd_loo": float(
                    loo_binomial.elpd_loo
                ),
                "binomial_p_loo": float(
                    loo_binomial.p_loo
                ),
                "beta_binomial_elpd_loo": float(
                    loo_beta_binomial.elpd_loo
                ),
                "beta_binomial_p_loo": float(
                    loo_beta_binomial.p_loo
                ),
            }

            with open(
                os.path.join(
                    out_dir,
                    f"{prefix}_loo.json",
                ),
                "w",
            ) as file:
                json.dump(
                    information_criteria,
                    file,
                    indent=2,
                )

        except Exception as error:
            print(
                "[WARN] LOO calculation failed:",
                repr(error),
            )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "product": product_name,
        "case_name": case_name,
        "model_support": model_support,
        "input_table": table_path,
        "temperature_column": temp_col,
        "cell_column": config["cell_col"],
        "region_column": region_col,
        "temperature_mean_for_standardization": temp_mean,
        "temperature_sd_for_standardization": temp_sd,
        "n_rows": int(len(data)),
        "n_temperature_bins": int(len(binned)),
        "binning_note": (
            "Temperature bins are descriptive only when MODEL_SUPPORT=row. "
            "When MODEL_SUPPORT=raw_aggregate, the pooled bin counts are the "
            "observations used in both likelihoods."
        ),
        "binomial_model": {
            "formula": (
                "y ~ Binomial(n, mu); "
                "cloglog(mu) = alpha + beta_temp * temp_z"
            ),
            "trace_path": binomial_trace_path,
            "independence_assumption": (
                "Conditional binomial variance assumes independent and "
                "identically distributed Bernoulli trials within each "
                "observation."
            ),
        },
        "beta_binomial_model": {
            "formula": (
                "y ~ BetaBinomial(n, mu*kappa, (1-mu)*kappa); "
                "cloglog(mu) = alpha + beta_temp * temp_z"
            ),
            "kappa_prior": (
                f"Gamma(alpha={kappa_alpha}, beta={kappa_beta})"
            ),
            "trace_path": beta_binomial_trace_path,
        },
        "historical_fit_metrics": {
            "binomial": binomial_metrics,
            "beta_binomial": beta_binomial_metrics,
        },
        "information_criteria": information_criteria,
    }

    metadata_path = os.path.join(
        out_dir,
        f"{prefix}_metadata.json",
    )

    with open(
        metadata_path,
        "w",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=str,
        )

    print("[SAVED]", metadata_path)

    print("\n" + "#" * 100)
    print("[DONE] BINOMIAL VS BETA-BINOMIAL COMPLETE")
    print("#" * 100)
    print("Outputs:", out_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()