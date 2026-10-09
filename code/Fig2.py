#!/usr/bin/env python
# coding: utf-8

"""
Fig. 2 Manuscript Publication Figure

"""
 
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from mpl_toolkits.axes_grid1.inset_locator import mark_inset

 
# Paths
 
CLUSTER_OUT_DIR = (
    "recropped_shelf_spatial_autocorrelation_percolation_outputs"
)

CLUSTER_FIG_DIR = os.path.join(
    CLUSTER_OUT_DIR,
    "publication_supplement_figures",
)

CLUSTER_TABLE_DIR = os.path.join(
    CLUSTER_OUT_DIR,
    "publication_tables",
)

SCENE_TABLE_PATH = os.path.join(
    CLUSTER_TABLE_DIR,
    "largest_cluster_scene_points.csv",
)

KNEE_SUMMARY_PATH = os.path.join(
    CLUSTER_TABLE_DIR,
    "empirical_knee_summary.csv",
)

BAYES_OUT_DIR = (
    "era5_primary_and_peninsula_sensitivity"
)

BAYES_CURVES_PATH = os.path.join(
    BAYES_OUT_DIR,
    "primary_and_peninsula_posterior_curves.csv",
)

BINNED_OBSERVATIONS_PATH = os.path.join(
    BAYES_OUT_DIR,
    "equal_width_all_shelves_observations.csv",
)

PENINSULA_BOOTSTRAP_PATH = os.path.join(
    BAYES_OUT_DIR,
    "peninsula_binomial_shelf_bootstrap_curve.csv",
)

os.makedirs(
    CLUSTER_FIG_DIR,
    exist_ok=True,
)

OUTPUT_BASENAME = os.path.join(
    CLUSTER_FIG_DIR,
    "final_dominant_cluster_and_temperature_response_two_panel",
)

CAPTION_PATH = (
    OUTPUT_BASENAME
    + "_caption.txt"
)

TRANSITION_TEMPERATURE_PATH = (
    OUTPUT_BASENAME
    + "_transition_temperature_summary.csv"
)
 
PRIMARY_MIN_PONDED_PIXELS = 100

X_COLUMN = "ponded_fraction_calculated"

Y_COLUMN = (
    "largest_component_fraction_of_valid_calculated"
)

SHELF_COLUMN = "shelf"

# Shelf-balanced local quantile settings
  
QUANTILES_TO_PLOT = [
    0.50,
    0.75,
    0.90,
]

LOCAL_WINDOW_DEX = 0.35
MIN_LOCAL_POINTS = 20
N_GRID = 400


# Cluster-panel uncertainty

SHOW_68_INTERVAL = True

SHOW_95_INTERVAL = False

SHOW_EXTENSIVE_QUANTILE_ZONE = False

TEMPERATURE_PANEL_X_MIN = -17.0
TEMPERATURE_PANEL_X_MAX = None

TEMPERATURE_PANEL_Y_MIN = 0.0
TEMPERATURE_PANEL_Y_MAX = 17.0

SHOW_TEMPERATURE_BARS = True
  
# Warm-temperature inset

TEMPERATURE_INSET_X_MIN = -7.0
TEMPERATURE_INSET_X_MAX = 0

TEMPERATURE_INSET_Y_MIN = 0.0
TEMPERATURE_INSET_Y_MAX = 2.0

TEMPERATURE_INSET_BOUNDS = [
    0.12,
    0.40,
    0.57,
    0.42,
]

# PNAS figure dimensions
# Standard PNAS widths:
#   8.7 cm  = one column
#   11.4 cm = one and a half columns
#   17.8 cm = two columns
  
FIGURE_WIDTH_CM = 16
FIGURE_HEIGHT_CM = 11

SAVE_DPI = 600
  
# Universal legend

UNIVERSAL_LEGEND_COLUMNS = 3
UNIVERSAL_LEGEND_Y = 0.035
 
# Colors
 
COLOR_CLUSTER_POINTS = "#0072B2"
COLOR_CLUSTER_CURVES = "#D55E00"
COLOR_KNEE = "black"

COLOR_INTERVAL_68 = "0.35"
COLOR_INTERVAL_95 = "0.62"
COLOR_EXTENSIVE_ZONE = "#E69F00"

COLOR_BINNED_OBSERVATIONS = "#6BAED6"
COLOR_EQUAL_ROW = "#56423c"

COLOR_ALL_SHELVES_BB = "#f05b7a"
COLOR_PENINSULA_BINOMIAL = "#6A3D9A"
 
# Plot style
 
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 7.7,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 600,
        "savefig.dpi": SAVE_DPI,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
 

def check_file(
    path,
    description,
):

    if not os.path.isfile(
        path
    ):
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def require_columns(
    dataframe,
    required_columns,
    table_name,
):

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing columns: {missing}"
        )


def get_summary_value(
    summary,
    column,
    fallback=np.nan,
):

    if column not in summary.columns:
        return fallback

    value = pd.to_numeric(
        summary.loc[
            0,
            column,
        ],
        errors="coerce",
    )

    if np.isfinite(
        value
    ):
        return float(
            value
        )

    return fallback


def percent_formatter(
    value,
    position=None,
):

    return f"{100.0 * value:g}%"


def percent_unit_formatter(
    value,
    position=None,
):

    return f"{value:g}%"


def parse_boolean_series(
    series,
):

    if pd.api.types.is_bool_dtype(
        series
    ):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    text = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    output = text.isin(
        [
            "true",
            "t",
            "yes",
            "y",
            "1",
        ]
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_mask = numeric.notna()

    output.loc[
        numeric_mask
    ] = (
        numeric.loc[
            numeric_mask
        ]
        != 0
    )

    return output.astype(
        bool
    )


def savefig_both(
    figure,
    path_base,
):

    png_path = path_base + ".png"
    pdf_path = path_base + ".pdf"

    figure.savefig(
        png_path,
        dpi=SAVE_DPI,
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        facecolor="white",
    )

    print(
        "[SAVED]",
        png_path,
    )

    print(
        "[SAVED]",
        pdf_path,
    )
 

def weighted_quantile(
    values,
    quantile,
    weights,
):
    """Calculate a weighted empirical quantile."""

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(
            values
        )
        & np.isfinite(
            weights
        )
        & (
            weights > 0
        )
    )

    values = values[
        valid
    ]

    weights = weights[
        valid
    ]

    if len(
        values
    ) == 0:
        return np.nan

    order = np.argsort(
        values
    )

    values = values[
        order
    ]

    weights = weights[
        order
    ]

    cumulative = (
        np.cumsum(
            weights
        )
        - 0.5
        * weights
    )

    cumulative = (
        cumulative
        / np.sum(
            weights
        )
    )

    return float(
        np.interp(
            quantile,
            cumulative,
            values,
            left=values[
                0
            ],
            right=values[
                -1
            ],
        )
    )


def shelf_balanced_local_quantile_curve(
    data,
    *,
    quantile,
    fixed_logx_grid,
    x_column=X_COLUMN,
    y_column=Y_COLUMN,
    shelf_column=SHELF_COLUMN,
    window_dex=LOCAL_WINDOW_DEX,
    min_points=MIN_LOCAL_POINTS,
):

    subset = data.loc[
        np.isfinite(
            data[
                x_column
            ]
        )
        & np.isfinite(
            data[
                y_column
            ]
        )
        & (
            data[
                x_column
            ]
            > 0
        )
        & (
            data[
                y_column
            ]
            >= 0
        )
        & data[
            shelf_column
        ].notna()
    ].copy()

    if len(
        subset
    ) < min_points:
        return pd.DataFrame()

    x = subset[
        x_column
    ].to_numpy(
        dtype=float
    )

    y = subset[
        y_column
    ].to_numpy(
        dtype=float
    )

    shelves = (
        subset[
            shelf_column
        ]
        .astype(str)
        .to_numpy()
    )

    logx = np.log10(
        x
    )

    grid = np.asarray(
        fixed_logx_grid,
        dtype=float,
    )

    grid = grid[
        (
            grid
            >= logx.min()
        )
        & (
            grid
            <= logx.max()
        )
    ]

    half_width = (
        window_dex
        / 2.0
    )

    rows = []

    for grid_value in grid:
        local = (
            np.abs(
                logx
                - grid_value
            )
            <= half_width
        )

        if local.sum() >= min_points:
            indices = np.flatnonzero(
                local
            )

        else:
            indices = np.argsort(
                np.abs(
                    logx
                    - grid_value
                )
            )[
                :min_points
            ]

        if len(
            indices
        ) < min_points:
            continue

        local_shelves = shelves[
            indices
        ]

        local_y = y[
            indices
        ]

        (
            unique_shelves,
            inverse,
            counts,
        ) = np.unique(
            local_shelves,
            return_inverse=True,
            return_counts=True,
        )

        weights = (
            1.0
            / counts[
                inverse
            ]
        )

        value = weighted_quantile(
            local_y,
            quantile,
            weights,
        )

        if not np.isfinite(
            value
        ):
            continue

        rows.append(
            {
                "x": (
                    10**grid_value
                ),
                "y": value,
                "quantile": quantile,
                "n_local_scenes": len(
                    indices
                ),
                "n_local_shelves": len(
                    unique_shelves
                ),
            }
        )

    return pd.DataFrame(
        rows
    )

def temperature_at_ponding_fraction(
    curve,
    ponding_fraction,
    *,
    temperature_column="temperature",
    response_column="median_percent",
):

    if not np.isfinite(
        ponding_fraction
    ):
        return np.nan

    target_percent = (
        100.0
        * float(
            ponding_fraction
        )
    )

    temperature = pd.to_numeric(
        curve[
            temperature_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    response = pd.to_numeric(
        curve[
            response_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            temperature
        )
        & np.isfinite(
            response
        )
    )

    temperature = temperature[
        valid
    ]

    response = response[
        valid
    ]

    if len(
        response
    ) < 2:
        return np.nan

    order = np.argsort(
        response
    )

    response = response[
        order
    ]

    temperature = temperature[
        order
    ]

    (
        response,
        unique_indices,
    ) = np.unique(
        response,
        return_index=True,
    )

    temperature = temperature[
        unique_indices
    ]

    if len(
        response
    ) < 2:
        return np.nan

    if (
        target_percent
        < response.min()
        or target_percent
        > response.max()
    ):
        return np.nan

    return float(
        np.interp(
            target_percent,
            response,
            temperature,
        )
    )


def temperature_within_training_support(
    curve,
    temperature_value,
):

    if not np.isfinite(
        temperature_value
    ):
        return pd.NA

    if (
        "within_training_support"
        not in curve.columns
    ):
        return pd.NA

    support = curve.loc[
        curve[
            "within_training_support"
        ]
    ]

    support_temperature = pd.to_numeric(
        support[
            "temperature"
        ],
        errors="coerce",
    )

    support_temperature = support_temperature[
        np.isfinite(
            support_temperature
        )
    ]

    if support_temperature.empty:
        return False

    return bool(
        (
            temperature_value
            >= support_temperature.min()
        )
        and (
            temperature_value
            <= support_temperature.max()
        )
    )


def summarize_transition_temperatures(
    *,
    all_shelves_curve,
    peninsula_curve,
    empirical_knee,
    knee_68_low,
    knee_68_high,
    knee_95_low,
    knee_95_high,
):


    transition_values = {
        "transition": empirical_knee,
        "central_68_low": knee_68_low,
        "central_68_high": knee_68_high,
        "central_95_low": knee_95_low,
        "central_95_high": knee_95_high,
    }

    model_curves = {
        "all_shelves_beta_binomial": all_shelves_curve,
        "antarctic_peninsula_binomial": peninsula_curve,
    }

    rows = []

    for model_name, curve in model_curves.items():
        response_minimum = float(
            np.nanmin(
                curve[
                    "median_percent"
                ]
            )
        )

        response_maximum = float(
            np.nanmax(
                curve[
                    "median_percent"
                ]
            )
        )

        for statistic, fraction in transition_values.items():
            if np.isfinite(
                fraction
            ):
                implied_temperature = (
                    temperature_at_ponding_fraction(
                        curve,
                        fraction,
                    )
                )

                target_percent = (
                    100.0
                    * fraction
                )

                response_range_status = (
                    "inside_median_response_range"
                    if (
                        target_percent
                        >= response_minimum
                        and target_percent
                        <= response_maximum
                    )
                    else "outside_median_response_range"
                )

            else:
                implied_temperature = np.nan
                target_percent = np.nan
                response_range_status = (
                    "transition_fraction_unavailable"
                )

            support_status = (
                temperature_within_training_support(
                    curve,
                    implied_temperature,
                )
            )

            rows.append(
                {
                    "model": model_name,
                    "transition_statistic": statistic,
                    "ponding_fraction": fraction,
                    "ponding_percent": target_percent,
                    "implied_era5_djf_temperature_c": (
                        implied_temperature
                    ),
                    "temperature_within_training_support": (
                        support_status
                    ),
                    "median_response_minimum_percent": (
                        response_minimum
                    ),
                    "median_response_maximum_percent": (
                        response_maximum
                    ),
                    "response_range_status": (
                        response_range_status
                    ),
                    "response_curve_used": (
                        "posterior median"
                    ),
                    "uncertainty_interpretation": (
                        "Transition-fraction uncertainty propagated through "
                        "the posterior median response; response-model "
                        "posterior uncertainty is not jointly propagated."
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def get_transition_temperature(
    summary,
    model,
    statistic,
):

    selected = summary.loc[
        (
            summary[
                "model"
            ]
            == model
        )
        & (
            summary[
                "transition_statistic"
            ]
            == statistic
        ),
        "implied_era5_djf_temperature_c",
    ]

    if selected.empty:
        return np.nan

    value = pd.to_numeric(
        selected.iloc[
            0
        ],
        errors="coerce",
    )

    if np.isfinite(
        value
    ):
        return float(
            value
        )

    return np.nan


def format_temperature(
    value,
):

    if np.isfinite(
        value
    ):
        return f"{value:.3f}°C"

    return "outside modeled median-response range"


def format_temperature_range(
    first_value,
    second_value,
):

    if (
        np.isfinite(
            first_value
        )
        and np.isfinite(
            second_value
        )
    ):
        lower_temperature = min(
            first_value,
            second_value,
        )

        upper_temperature = max(
            first_value,
            second_value,
        )

        return (
            f"{lower_temperature:.3f} to "
            f"{upper_temperature:.3f}°C"
        )

    return (
        f"{format_temperature(first_value)} to "
        f"{format_temperature(second_value)}"
    )


def print_transition_temperature_report(
    *,
    model_label,
    transition_temperature,
    temperature_68_low,
    temperature_68_high,
    temperature_95_low,
    temperature_95_high,
):

    print(
        model_label
    )

    print(
        "  Transition temperature:",
        format_temperature(
            transition_temperature
        ),
    )

    print(
        "  Temperature range from central 68% "
        "transition-fraction bounds:",
        format_temperature_range(
            temperature_68_low,
            temperature_68_high,
        ),
    )

    print(
        "  Temperature range from central 95% "
        "transition-fraction bounds:",
        format_temperature_range(
            temperature_95_low,
            temperature_95_high,
        ),
    )
 
# Temperature-response plotting helpers

def plot_temperature_model_curve(
    axis,
    curve,
    *,
    color,
    interval_alpha,
    outside_support_alpha=None,
    linewidth=2.2,
):

    curve = curve.sort_values(
        "temperature"
    ).copy()

    if (
        "within_training_support"
        not in curve.columns
    ):
        curve[
            "within_training_support"
        ] = True

    inside = curve.loc[
        curve[
            "within_training_support"
        ]
    ]

    outside = curve.loc[
        ~curve[
            "within_training_support"
        ]
    ]

    if not inside.empty:
        axis.fill_between(
            inside[
                "temperature"
            ].to_numpy(
                dtype=float
            ),
            inside[
                "q025_percent"
            ].to_numpy(
                dtype=float
            ),
            inside[
                "q975_percent"
            ].to_numpy(
                dtype=float
            ),
            color=color,
            alpha=interval_alpha,
            linewidth=0,
            zorder=3,
        )

        axis.plot(
            inside[
                "temperature"
            ],
            inside[
                "median_percent"
            ],
            color=color,
            linewidth=linewidth,
            zorder=8,
        )

    if not outside.empty:
        if outside_support_alpha is not None:
            axis.fill_between(
                outside[
                    "temperature"
                ].to_numpy(
                    dtype=float
                ),
                outside[
                    "q025_percent"
                ].to_numpy(
                    dtype=float
                ),
                outside[
                    "q975_percent"
                ].to_numpy(
                    dtype=float
                ),
                color=color,
                alpha=outside_support_alpha,
                linewidth=0,
                zorder=2,
            )

        axis.plot(
            outside[
                "temperature"
            ],
            outside[
                "median_percent"
            ],
            color=color,
            linestyle="--",
            linewidth=(
                linewidth
                * 0.88
            ),
            zorder=8,
        )


def plot_peninsula_bootstrap_interval(
    axis,
    bootstrap_curve,
    *,
    color=COLOR_PENINSULA_BINOMIAL,
    inside_alpha=0.22,
    outside_alpha=0.06,
):
    """Plot Peninsula whole-shelf bootstrap 95% intervals."""

    bootstrap_inside = bootstrap_curve.loc[
        bootstrap_curve[
            "inside_training_support"
        ]
    ].copy()

    bootstrap_outside = bootstrap_curve.loc[
        ~bootstrap_curve[
            "inside_training_support"
        ]
    ].copy()

    if not bootstrap_inside.empty:
        axis.fill_between(
            bootstrap_inside[
                "temperature"
            ].to_numpy(
                dtype=float
            ),
            bootstrap_inside[
                "bootstrap_q025_percent"
            ].to_numpy(
                dtype=float
            ),
            bootstrap_inside[
                "bootstrap_q975_percent"
            ].to_numpy(
                dtype=float
            ),
            color=color,
            alpha=inside_alpha,
            linewidth=0,
            zorder=3,
        )

    if not bootstrap_outside.empty:
        axis.fill_between(
            bootstrap_outside[
                "temperature"
            ].to_numpy(
                dtype=float
            ),
            bootstrap_outside[
                "bootstrap_q025_percent"
            ].to_numpy(
                dtype=float
            ),
            bootstrap_outside[
                "bootstrap_q975_percent"
            ].to_numpy(
                dtype=float
            ),
            color=color,
            alpha=outside_alpha,
            linewidth=0,
            zorder=2,
        )


def plot_peninsula_median(
    axis,
    peninsula_curve,
    *,
    linewidth=2.2,
):

    peninsula_inside = peninsula_curve.loc[
        peninsula_curve[
            "within_training_support"
        ]
    ].copy()

    peninsula_outside = peninsula_curve.loc[
        ~peninsula_curve[
            "within_training_support"
        ]
    ].copy()

    if not peninsula_inside.empty:
        axis.plot(
            peninsula_inside[
                "temperature"
            ],
            peninsula_inside[
                "median_percent"
            ],
            color=COLOR_PENINSULA_BINOMIAL,
            linewidth=linewidth,
            zorder=8,
        )

    if not peninsula_outside.empty:
        axis.plot(
            peninsula_outside[
                "temperature"
            ],
            peninsula_outside[
                "median_percent"
            ],
            color=COLOR_PENINSULA_BINOMIAL,
            linestyle="--",
            linewidth=(
                linewidth
                * 0.88
            ),
            zorder=8,
        )
 
# Read cluster-analysis outputs
 
check_file(
    SCENE_TABLE_PATH,
    "Cluster scene table",
)

check_file(
    KNEE_SUMMARY_PATH,
    "Empirical-knee summary",
)

scene_data = pd.read_csv(
    SCENE_TABLE_PATH
)

knee_summary = pd.read_csv(
    KNEE_SUMMARY_PATH
)

if knee_summary.empty:
    raise ValueError(
        "The empirical-knee summary table is empty."
    )

require_columns(
    scene_data,
    [
        X_COLUMN,
        Y_COLUMN,
        SHELF_COLUMN,
    ],
    "largest_cluster_scene_points.csv",
)

scene_data[
    X_COLUMN
] = pd.to_numeric(
    scene_data[
        X_COLUMN
    ],
    errors="coerce",
)

scene_data[
    Y_COLUMN
] = pd.to_numeric(
    scene_data[
        Y_COLUMN
    ],
    errors="coerce",
)

if (
    "n_ponded_pixels"
    in scene_data.columns
):
    scene_data[
        "n_ponded_pixels"
    ] = pd.to_numeric(
        scene_data[
            "n_ponded_pixels"
        ],
        errors="coerce",
    )

    scene_data = scene_data.loc[
        scene_data[
            "n_ponded_pixels"
        ]
        >= PRIMARY_MIN_PONDED_PIXELS
    ].copy()

plot_data = scene_data.loc[
    np.isfinite(
        scene_data[
            X_COLUMN
        ]
    )
    & np.isfinite(
        scene_data[
            Y_COLUMN
        ]
    )
    & (
        scene_data[
            X_COLUMN
        ]
        > 0
    )
    & (
        scene_data[
            Y_COLUMN
        ]
        >= 0
    )
    & scene_data[
        SHELF_COLUMN
    ].notna()
].copy()

if plot_data.empty:
    raise RuntimeError(
        "No cluster observations passed the plotting filters."
    )
 
# Read empirical-knee estimates
 
empirical_knee = get_summary_value(
    knee_summary,
    "empirical_knee",
)

knee_68_low = get_summary_value(
    knee_summary,
    "bootstrap_68_low",
)

knee_68_high = get_summary_value(
    knee_summary,
    "bootstrap_68_high",
)

knee_95_low = get_summary_value(
    knee_summary,
    "bootstrap_95_low",
)

knee_95_high = get_summary_value(
    knee_summary,
    "bootstrap_95_high",
)

extensive_zone_low = get_summary_value(
    knee_summary,
    "extensive_zone_low",
)

extensive_zone_high = get_summary_value(
    knee_summary,
    "extensive_zone_high",
)

if not np.isfinite(
    empirical_knee
):
    raise ValueError(
        "Could not read the empirical knee from "
        "empirical_knee_summary.csv."
    )

 
# Calculate shelf-balanced local quantile curves

x_values = plot_data[
    X_COLUMN
].to_numpy(
    dtype=float
)

fixed_logx_grid = np.linspace(
    np.log10(
        x_values.min()
    ),
    np.log10(
        x_values.max()
    ),
    N_GRID,
)

cluster_curves = {}

for quantile in QUANTILES_TO_PLOT:
    curve = shelf_balanced_local_quantile_curve(
        plot_data,
        quantile=quantile,
        fixed_logx_grid=fixed_logx_grid,
    )

    if curve.empty:
        warnings.warn(
            "Could not calculate the "
            f"{quantile:.2f} local quantile curve."
        )

    cluster_curves[
        quantile
    ] = curve


cluster_curve_styles = {
    0.50: {
        "linestyle": "-",
        "linewidth": 2.0,
    },

    0.75: {
        "linestyle": "--",
        "linewidth": 2.0,
    },

    0.90: {
        "linestyle": ":",
        "linewidth": 2.2,
    },
}

# Read temperature-response outputs
 
check_file(
    BAYES_CURVES_PATH,
    "Bayesian posterior curves",
)

check_file(
    BINNED_OBSERVATIONS_PATH,
    "Binned observed temperature response",
)

bayesian_curves = pd.read_csv(
    BAYES_CURVES_PATH
)

binned_observations = pd.read_csv(
    BINNED_OBSERVATIONS_PATH
)

require_columns(
    bayesian_curves,
    [
        "model_key",
        "temperature",
        "median_percent",
        "q025_percent",
        "q975_percent",
    ],
    "primary_and_peninsula_posterior_curves.csv",
)

require_columns(
    binned_observations,
    [
        "bin_center",
        "bin_width",
        "pixel_pooled_percent",
        "temperature_mean_equal_row",
        "equal_row_mean_percent",
    ],
    "equal_width_all_shelves_observations.csv",
)

numeric_curve_columns = [
    "temperature",
    "median_percent",
    "q025_percent",
    "q975_percent",
]

for column in numeric_curve_columns:
    bayesian_curves[
        column
    ] = pd.to_numeric(
        bayesian_curves[
            column
        ],
        errors="coerce",
    )

numeric_bin_columns = [
    "bin_center",
    "bin_width",
    "pixel_pooled_percent",
    "temperature_mean_equal_row",
    "equal_row_mean_percent",
]

for column in numeric_bin_columns:
    binned_observations[
        column
    ] = pd.to_numeric(
        binned_observations[
            column
        ],
        errors="coerce",
    )
  
# Select model curves
  

model_key_text = (
    bayesian_curves[
        "model_key"
    ]
    .astype("string")
    .str.lower()
)

all_shelves_curve = bayesian_curves.loc[
    model_key_text.str.contains(
        "all_shelves",
        na=False,
    )
].copy()

peninsula_curve = bayesian_curves.loc[
    model_key_text.str.contains(
        "peninsula",
        na=False,
    )
].copy()

if all_shelves_curve.empty:
    raise RuntimeError(
        "Could not identify the all-shelves Beta-Binomial curve."
    )

if peninsula_curve.empty:
    raise RuntimeError(
        "Could not identify the Peninsula Binomial curve."
    )

all_shelves_curve = (
    all_shelves_curve
    .sort_values(
        "temperature"
    )
    .reset_index(
        drop=True
    )
)

peninsula_curve = (
    peninsula_curve
    .sort_values(
        "temperature"
    )
    .reset_index(
        drop=True
    )
)
  
# Identify model training support

if (
    "within_training_support"
    in all_shelves_curve.columns
):
    all_shelves_curve[
        "within_training_support"
    ] = parse_boolean_series(
        all_shelves_curve[
            "within_training_support"
        ]
    )
else:
    all_shelves_curve[
        "within_training_support"
    ] = True


if (
    "within_training_support"
    in peninsula_curve.columns
):
    peninsula_curve[
        "within_training_support"
    ] = parse_boolean_series(
        peninsula_curve[
            "within_training_support"
        ]
    )

else:
    peninsula_curve[
        "within_training_support"
    ] = True

# Read Peninsula whole-shelf bootstrap interval
  
peninsula_bootstrap = None

if os.path.isfile(
    PENINSULA_BOOTSTRAP_PATH
):
    peninsula_bootstrap = pd.read_csv(
        PENINSULA_BOOTSTRAP_PATH
    )

    required_bootstrap_columns = [
        "temperature",
        "bootstrap_q025_percent",
        "bootstrap_median_percent",
        "bootstrap_q975_percent",
    ]

    require_columns(
        peninsula_bootstrap,
        required_bootstrap_columns,
        "peninsula_binomial_shelf_bootstrap_curve.csv",
    )

    for column in required_bootstrap_columns:
        peninsula_bootstrap[
            column
        ] = pd.to_numeric(
            peninsula_bootstrap[
                column
            ],
            errors="coerce",
        )

    if (
        "inside_training_support"
        in peninsula_bootstrap.columns
    ):
        peninsula_bootstrap[
            "inside_training_support"
        ] = parse_boolean_series(
            peninsula_bootstrap[
                "inside_training_support"
            ]
        )

    else:
        peninsula_bootstrap[
            "inside_training_support"
        ] = True

    peninsula_bootstrap = (
        peninsula_bootstrap
        .sort_values(
            "temperature"
        )
        .reset_index(
            drop=True
        )
    )

else:
    warnings.warn(
        "The Peninsula shelf-bootstrap curve was not found. "
        "The posterior interval from the Binomial model will be used."
    )
 

transition_temperature_summary = summarize_transition_temperatures(
    all_shelves_curve=all_shelves_curve,
    peninsula_curve=peninsula_curve,
    empirical_knee=empirical_knee,
    knee_68_low=knee_68_low,
    knee_68_high=knee_68_high,
    knee_95_low=knee_95_low,
    knee_95_high=knee_95_high,
)

transition_temperature_summary.to_csv(
    TRANSITION_TEMPERATURE_PATH,
    index=False,
)

print(
    "[SAVED]",
    TRANSITION_TEMPERATURE_PATH,
)

all_shelves_transition_temperature = get_transition_temperature(
    transition_temperature_summary,
    "all_shelves_beta_binomial",
    "transition",
)

all_shelves_68_low_temperature = get_transition_temperature(
    transition_temperature_summary,
    "all_shelves_beta_binomial",
    "central_68_low",
)

all_shelves_68_high_temperature = get_transition_temperature(
    transition_temperature_summary,
    "all_shelves_beta_binomial",
    "central_68_high",
)

all_shelves_95_low_temperature = get_transition_temperature(
    transition_temperature_summary,
    "all_shelves_beta_binomial",
    "central_95_low",
)

all_shelves_95_high_temperature = get_transition_temperature(
    transition_temperature_summary,
    "all_shelves_beta_binomial",
    "central_95_high",
)

peninsula_transition_temperature = get_transition_temperature(
    transition_temperature_summary,
    "antarctic_peninsula_binomial",
    "transition",
)

peninsula_68_low_temperature = get_transition_temperature(
    transition_temperature_summary,
    "antarctic_peninsula_binomial",
    "central_68_low",
)

peninsula_68_high_temperature = get_transition_temperature(
    transition_temperature_summary,
    "antarctic_peninsula_binomial",
    "central_68_high",
)

peninsula_95_low_temperature = get_transition_temperature(
    transition_temperature_summary,
    "antarctic_peninsula_binomial",
    "central_95_low",
)

peninsula_95_high_temperature = get_transition_temperature(
    transition_temperature_summary,
    "antarctic_peninsula_binomial",
    "central_95_high",
)

# Temperature ranges and uncertainty
 
available_temperature_minimum = float(
    np.nanmin(
        [
            binned_observations[
                "bin_center"
            ].min(),
            all_shelves_curve[
                "temperature"
            ].min(),
            peninsula_curve[
                "temperature"
            ].min(),
        ]
    )
)

available_temperature_maximum = float(
    np.nanmax(
        [
            binned_observations[
                "bin_center"
            ].max(),
            all_shelves_curve[
                "temperature"
            ].max(),
            peninsula_curve[
                "temperature"
            ].max(),
        ]
    )
)

temperature_panel_x_maximum = (
    TEMPERATURE_PANEL_X_MAX
    if TEMPERATURE_PANEL_X_MAX is not None
    else available_temperature_maximum
)

temperature_inset_x_maximum = (
    TEMPERATURE_INSET_X_MAX
    if TEMPERATURE_INSET_X_MAX is not None
    else available_temperature_maximum
)

all_shelves_interval_maximum = float(
    np.nanmax(
        all_shelves_curve[
            "q975_percent"
        ]
    )
)

if peninsula_bootstrap is not None:
    peninsula_interval_maximum = float(
        np.nanmax(
            peninsula_bootstrap[
                "bootstrap_q975_percent"
            ]
        )
    )

    peninsula_interval_description = (
        "whole-shelf bootstrap"
    )

else:
    peninsula_interval_maximum = float(
        np.nanmax(
            peninsula_curve[
                "q975_percent"
            ]
        )
    )

    peninsula_interval_description = (
        "posterior"
    )


print(
    "=" * 78
)

print(
    "Final dominant-cluster and temperature-response two-panel figure"
)

print(
    "=" * 78
)

print(
    "Cluster observations:",
    len(
        plot_data
    ),
)

print(
    "Cluster shelves:",
    plot_data[
        SHELF_COLUMN
    ].nunique(),
)

print(
    "Empirical knee:",
    f"{100.0 * empirical_knee:.4f}%",
)

if np.all(
    np.isfinite(
        [
            knee_68_low,
            knee_68_high,
        ]
    )
):
    print(
        "Central 68% shelf-bootstrap interval:",
        (
            f"{100.0 * knee_68_low:.4f}%–"
            f"{100.0 * knee_68_high:.4f}%"
        ),
    )

if np.all(
    np.isfinite(
        [
            knee_95_low,
            knee_95_high,
        ]
    )
):
    print(
        "Central 95% shelf-bootstrap interval:",
        (
            f"{100.0 * knee_95_low:.4f}%–"
            f"{100.0 * knee_95_high:.4f}%"
        ),
    )

print(
    "Available modeled temperature range:",
    (
        f"{available_temperature_minimum:.3f} to "
        f"{available_temperature_maximum:.3f}°C"
    ),
)

print(
    "Displayed Panel B temperature range:",
    (
        f"{TEMPERATURE_PANEL_X_MIN:.3f} to "
        f"{temperature_panel_x_maximum:.3f}°C"
    ),
)

print(
    "Displayed Panel B response range:",
    (
        f"{TEMPERATURE_PANEL_Y_MIN:.1f} to "
        f"{TEMPERATURE_PANEL_Y_MAX:.1f}%"
    ),
)

print(
    "All-shelves 95% interval maximum:",
    f"{all_shelves_interval_maximum:.3f}%",
)

print(
    f"Peninsula {peninsula_interval_description} 95% interval maximum:",
    f"{peninsula_interval_maximum:.3f}%",
)

if (
    all_shelves_interval_maximum
    > TEMPERATURE_PANEL_Y_MAX
    or peninsula_interval_maximum
    > TEMPERATURE_PANEL_Y_MAX
):
    print(
    "Model output larger than observed temperature range"
    )

print(
    "\nModel-implied ERA5 DJF temperature at the cluster transition"
)

print(
    "-" * 78
)

print_transition_temperature_report(
    model_label=(
        "All-shelves Beta-Binomial relationship"
    ),
    transition_temperature=(
        all_shelves_transition_temperature
    ),
    temperature_68_low=(
        all_shelves_68_low_temperature
    ),
    temperature_68_high=(
        all_shelves_68_high_temperature
    ),
    temperature_95_low=(
        all_shelves_95_low_temperature
    ),
    temperature_95_high=(
        all_shelves_95_high_temperature
    ),
)

print_transition_temperature_report(
    model_label=(
        "Antarctic Peninsula Binomial relationship"
    ),
    transition_temperature=(
        peninsula_transition_temperature
    ),
    temperature_68_low=(
        peninsula_68_low_temperature
    ),
    temperature_68_high=(
        peninsula_68_high_temperature
    ),
    temperature_95_low=(
        peninsula_95_low_temperature
    ),
    temperature_95_high=(
        peninsula_95_high_temperature
    ),
)

print(
    "=" * 78
)
 
# Create PNAS two-column-width figure
 

CM_TO_INCH = (
    1.0
    / 2.54
)

figure, (
    cluster_axis,
    temperature_axis,
) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(
        FIGURE_WIDTH_CM
        * CM_TO_INCH,
        FIGURE_HEIGHT_CM
        * CM_TO_INCH,
    ),
)

figure.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.29,
    top=0.965,
    wspace=0.20,
)


if (
    SHOW_95_INTERVAL
    and np.all(
        np.isfinite(
            [
                knee_95_low,
                knee_95_high,
            ]
        )
    )
):
    cluster_axis.axvspan(
        knee_95_low,
        knee_95_high,
        color=COLOR_INTERVAL_95,
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )

if (
    SHOW_68_INTERVAL
    and np.all(
        np.isfinite(
            [
                knee_68_low,
                knee_68_high,
            ]
        )
    )
):
    cluster_axis.axvspan(
        knee_68_low,
        knee_68_high,
        color=COLOR_INTERVAL_68,
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )

if (
    SHOW_EXTENSIVE_QUANTILE_ZONE
    and np.all(
        np.isfinite(
            [
                extensive_zone_low,
                extensive_zone_high,
            ]
        )
    )
):
    cluster_axis.axvspan(
        extensive_zone_low,
        extensive_zone_high,
        color=COLOR_EXTENSIVE_ZONE,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )


  
# Cluster observations
  

cluster_axis.scatter(
    plot_data[
        X_COLUMN
    ],
    plot_data[
        Y_COLUMN
    ],
    s=24,
    alpha=0.36,
    color=COLOR_CLUSTER_POINTS,
    edgecolor="white",
    linewidth=0.27,
    rasterized=True,
    zorder=2,
)


  
# Shelf-balanced local quantile curves
  

for quantile in QUANTILES_TO_PLOT:
    curve = cluster_curves[
        quantile
    ]

    if curve.empty:
        continue

    style = cluster_curve_styles[
        quantile
    ]

    cluster_axis.plot(
        curve[
            "x"
        ],
        curve[
            "y"
        ],
        color=COLOR_CLUSTER_CURVES,
        linestyle=style[
            "linestyle"
        ],
        linewidth=style[
            "linewidth"
        ],
        zorder=4,
    )


  
# Empirical transition
  

cluster_axis.axvline(
    empirical_knee,
    color=COLOR_KNEE,
    linestyle="-",
    linewidth=1.8,
    zorder=6,
)

cluster_axis.set_xscale(
    "log"
)

cluster_x_minimum = float(
    plot_data[
        X_COLUMN
    ].min()
)

cluster_x_maximum = float(
    plot_data[
        X_COLUMN
    ].max()
)

cluster_axis.set_xlim(
    cluster_x_minimum
    / 1.15,
    cluster_x_maximum
    * 1.15,
)

cluster_axis.set_xlabel(
    "Ponded fraction (%)"
)

cluster_axis.set_ylabel(
    "Largest connected cluster / shelf pixels (%)"
)

cluster_axis.xaxis.set_major_formatter(
    mticker.FuncFormatter(
        percent_formatter
    )
)

cluster_axis.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        percent_formatter
    )
)

cluster_axis.set_ylim(
    bottom=0
)

cluster_axis.grid(
    True,
    which="major",
    color="0.86",
    linewidth=0.7,
)

cluster_axis.grid(
    True,
    which="minor",
    color="0.93",
    linewidth=0.45,
)

cluster_axis.spines[
    "top"
].set_visible(
    False
)

cluster_axis.spines[
    "right"
].set_visible(
    False
)

cluster_axis.text(
    0.015,
    0.975,
    "A",
    transform=cluster_axis.transAxes,
    ha="left",
    va="top",
    fontsize=11,
    fontweight="bold",
    zorder=200,
)

valid_equal_row = (
    np.isfinite(
        binned_observations[
            "temperature_mean_equal_row"
        ]
    )
    & np.isfinite(
        binned_observations[
            "equal_row_mean_percent"
        ]
    )
)

valid_bars = (
    np.isfinite(
        binned_observations[
            "bin_center"
        ]
    )
    & np.isfinite(
        binned_observations[
            "bin_width"
        ]
    )
    & np.isfinite(
        binned_observations[
            "pixel_pooled_percent"
        ]
    )
)
 
# Binned pixel-pooled observations

if SHOW_TEMPERATURE_BARS:
    temperature_axis.bar(
        binned_observations.loc[
            valid_bars,
            "bin_center",
        ],
        binned_observations.loc[
            valid_bars,
            "pixel_pooled_percent",
        ],
        width=(
            0.90
            * binned_observations.loc[
                valid_bars,
                "bin_width",
            ]
        ),
        color=COLOR_BINNED_OBSERVATIONS,
        edgecolor="#2171B5",
        linewidth=0.4,
        alpha=0.35,
        zorder=1,
    )


  
# All-shelves model
  

plot_temperature_model_curve(
    temperature_axis,
    all_shelves_curve,
    color=COLOR_ALL_SHELVES_BB,
    interval_alpha=0.16,
    outside_support_alpha=0.05,
    linewidth=2.0,
)

if peninsula_bootstrap is not None:
    plot_peninsula_bootstrap_interval(
        temperature_axis,
        peninsula_bootstrap,
        color=COLOR_PENINSULA_BINOMIAL,
        inside_alpha=0.22,
        outside_alpha=0.06,
    )

else:
    temperature_axis.fill_between(
        peninsula_curve[
            "temperature"
        ].to_numpy(
            dtype=float
        ),
        peninsula_curve[
            "q025_percent"
        ].to_numpy(
            dtype=float
        ),
        peninsula_curve[
            "q975_percent"
        ].to_numpy(
            dtype=float
        ),
        color=COLOR_PENINSULA_BINOMIAL,
        alpha=0.20,
        linewidth=0,
        zorder=3,
    )
 
# Peninsula median

plot_peninsula_median(
    temperature_axis,
    peninsula_curve,
    linewidth=2.0,
)
  
# Equal-row empirical bin means

temperature_axis.scatter(
    binned_observations.loc[
        valid_equal_row,
        "temperature_mean_equal_row",
    ],
    binned_observations.loc[
        valid_equal_row,
        "equal_row_mean_percent",
    ],
    color=COLOR_EQUAL_ROW,
    marker="s",
    s=27,
    edgecolor="white",
    linewidth=0.55,
    zorder=20,
)
  
# Panel B formatting
  
temperature_axis.set_xlim(
    TEMPERATURE_PANEL_X_MIN,
    temperature_panel_x_maximum,
)

temperature_axis.set_ylim(
    TEMPERATURE_PANEL_Y_MIN,
    TEMPERATURE_PANEL_Y_MAX,
)

temperature_axis.set_xlabel(
    "ERA5 DJF 2 m temperature (°C)"
)

temperature_axis.set_ylabel(
    "Ponded fraction (%)"
)

temperature_axis.yaxis.set_major_locator(
    mticker.MultipleLocator(
        2
    )
)

temperature_axis.yaxis.set_minor_locator(
    mticker.MultipleLocator(
        1
    )
)

temperature_axis.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        percent_unit_formatter
    )
)

temperature_axis.grid(
    True,
    which="major",
    color="0.87",
    linewidth=0.65,
    zorder=0,
)

temperature_axis.grid(
    True,
    which="minor",
    axis="y",
    color="0.94",
    linewidth=0.4,
    zorder=0,
)

temperature_axis.spines[
    "top"
].set_visible(
    False
)

temperature_axis.spines[
    "right"
].set_visible(
    False
)

temperature_axis.text(
    0.015,
    0.975,
    "B",
    transform=temperature_axis.transAxes,
    ha="left",
    va="top",
    fontsize=11,
    fontweight="bold",
    zorder=200,
)

# Panel B warm-temperature inset
 
temperature_inset = temperature_axis.inset_axes(
    TEMPERATURE_INSET_BOUNDS
)

temperature_inset.set_facecolor(
    "white"
)

temperature_inset.patch.set_alpha(
    0.96
)

# Inset observations
  
if SHOW_TEMPERATURE_BARS:
    temperature_inset.bar(
        binned_observations.loc[
            valid_bars,
            "bin_center",
        ],
        binned_observations.loc[
            valid_bars,
            "pixel_pooled_percent",
        ],
        width=(
            0.90
            * binned_observations.loc[
                valid_bars,
                "bin_width",
            ]
        ),
        color=COLOR_BINNED_OBSERVATIONS,
        edgecolor="#2171B5",
        linewidth=0.30,
        alpha=0.30,
        zorder=1,
    )
  
# Inset all-shelves model
  
plot_temperature_model_curve(
    temperature_inset,
    all_shelves_curve,
    color=COLOR_ALL_SHELVES_BB,
    interval_alpha=0.16,
    outside_support_alpha=0.05,
    linewidth=1.6,
)
 
# Inset Peninsula uncertainty

if peninsula_bootstrap is not None:
    plot_peninsula_bootstrap_interval(
        temperature_inset,
        peninsula_bootstrap,
        color=COLOR_PENINSULA_BINOMIAL,
        inside_alpha=0.22,
        outside_alpha=0.06,
    )

else:
    temperature_inset.fill_between(
        peninsula_curve[
            "temperature"
        ].to_numpy(
            dtype=float
        ),
        peninsula_curve[
            "q025_percent"
        ].to_numpy(
            dtype=float
        ),
        peninsula_curve[
            "q975_percent"
        ].to_numpy(
            dtype=float
        ),
        color=COLOR_PENINSULA_BINOMIAL,
        alpha=0.20,
        linewidth=0,
        zorder=3,
    ) 
# Inset Peninsula median

plot_peninsula_median(
    temperature_inset,
    peninsula_curve,
    linewidth=1.6,
)

temperature_inset.scatter(
    binned_observations.loc[
        valid_equal_row,
        "temperature_mean_equal_row",
    ],
    binned_observations.loc[
        valid_equal_row,
        "equal_row_mean_percent",
    ],
    color=COLOR_EQUAL_ROW,
    marker="s",
    s=18,
    edgecolor="white",
    linewidth=0.4,
    zorder=20,
)

temperature_inset.set_xlim(
    TEMPERATURE_INSET_X_MIN,
    temperature_inset_x_maximum,
)

temperature_inset.set_ylim(
    TEMPERATURE_INSET_Y_MIN,
    TEMPERATURE_INSET_Y_MAX,
)

temperature_inset.yaxis.set_major_locator(
    mticker.MultipleLocator(
        1
    )
)

temperature_inset.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        percent_unit_formatter
    )
)

temperature_inset.tick_params(
    axis="both",
    labelsize=6,
    length=2.2,
    pad=1.5,
)

temperature_inset.grid(
    True,
    color="0.88",
    linewidth=0.45,
    alpha=0.75,
)

for spine in temperature_inset.spines.values():
    spine.set_visible(
        True
    )

    spine.set_color(
        "0.40"
    )

    spine.set_linewidth(
        0.65
    )

mark_inset(
    temperature_axis,
    temperature_inset,
    loc1=1,
    loc2=3,
    fc="none",
    ec="0.45",
    linewidth=0.65,
)

cluster_quantile_legend_handle = (
    Line2D(
        [0],
        [0],
        color=COLOR_CLUSTER_CURVES,
        linestyle="-",
        linewidth=2.0,
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_CLUSTER_CURVES,
        linestyle="--",
        linewidth=2.0,
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_CLUSTER_CURVES,
        linestyle=":",
        linewidth=2.1,
    ),
)

# Combine each fitted response with its corresponding uncertainty envelope.
all_shelves_model_handle = (
    Patch(
        facecolor=COLOR_ALL_SHELVES_BB,
        edgecolor="none",
        alpha=0.16,
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_ALL_SHELVES_BB,
        linestyle="-",
        linewidth=2.0,
    ),
)

peninsula_model_handle = (
    Patch(
        facecolor=COLOR_PENINSULA_BINOMIAL,
        edgecolor="none",
        alpha=0.22,
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_PENINSULA_BINOMIAL,
        linestyle="-",
        linewidth=2.0,
    ),
)

# Combine the empirical threshold line and bootstrap band.
threshold_handle = (
    Patch(
        facecolor=COLOR_INTERVAL_68,
        edgecolor="none",
        alpha=0.14,
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_KNEE,
        linestyle="-",
        linewidth=1.8,
    ),
)

universal_legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=COLOR_CLUSTER_POINTS,
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=6,
        alpha=0.75,
    ),

    cluster_quantile_legend_handle,

    threshold_handle,

    Patch(
        facecolor=COLOR_BINNED_OBSERVATIONS,
        edgecolor="#2171B5",
        linewidth=0.4,
        alpha=0.35,
    ),

    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markerfacecolor=COLOR_EQUAL_ROW,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=6,
    ),

    all_shelves_model_handle,

    peninsula_model_handle,
]

universal_legend_labels = [
    "Shelf-year observations",
    "Shelf-balanced quantiles",
    "Connected-ponding threshold",
    "Pixel-pooled observations",
    "Equal-row observations",
    "All-Shelves model",
    "Peninsula-trained model",
]

universal_legend = figure.legend(
    handles=universal_legend_handles,
    labels=universal_legend_labels,
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        UNIVERSAL_LEGEND_Y,
    ),
    bbox_transform=figure.transFigure,
    ncol=3,
    frameon=True,
    framealpha=0.98,
    facecolor="white",
    edgecolor="0.70",
    fontsize=7,
    handlelength=2.4,
    columnspacing=1.2,
    handletextpad=0.55,
    labelspacing=0.45,
    borderpad=0.50,
    handler_map={
        tuple: HandlerTuple(
            ndivide=None,
            pad=0.20,
        )
    },
)

universal_legend.set_zorder(300)


savefig_both(
    figure,
    OUTPUT_BASENAME,
)

plt.show()