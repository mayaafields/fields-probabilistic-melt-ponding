#!/bin/bash
#SBATCH --job-name=ponding_temp_models
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ponding_temp_models_%j.out
#SBATCH --error=logs/ponding_temp_models_%j.err

set -euo pipefail

mkdir -p logs

echo "======================================================================"
echo "MANUSCRIPT TEMPERATURE-RESPONSE MODEL RUN"
echo "1. All-Shelves temperature-only Beta-Binomial"
echo "2. Five-shelf Antarctic Peninsula temperature-only Binomial"
echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Allocated memory: ${SLURM_MEM_PER_NODE:-unknown}"
echo "Temporary directory: ${TMPDIR:-/tmp}"
echo "======================================================================"


# =============================================================================
# 1. Python environment
# =============================================================================

# Uncomment and edit one environment setup method.

# ---------------------------------------------------------------------------
# Conda
# ---------------------------------------------------------------------------

# source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# conda activate antarctic-pymc

# ---------------------------------------------------------------------------
# Micromamba
# ---------------------------------------------------------------------------

# eval "$(micromamba shell hook --shell bash)"
# micromamba activate antarctic-pymc

# ---------------------------------------------------------------------------
# Environment modules
# ---------------------------------------------------------------------------

# module purge
# module load python/3.11


# =============================================================================
# 2. Prevent nested thread oversubscription
# =============================================================================

# PyMC runs four chains concurrently. Each chain receives one numerical-
# library thread, matching the four CPUs allocated by SLURM.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

# Place compiled PyTensor files on the job-local temporary filesystem.
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID}"

# Noninteractive plotting and immediate log output.
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1


# =============================================================================
# 3. Script
# =============================================================================

# Change this if the public/repository filename is different.
export MODEL_SCRIPT="fit_manuscript_temperature_models.py"


# =============================================================================
# 4. Input data
# =============================================================================

# Preferred input: processed row-level Parquet table containing one record per
# shelf × ERA5 cell × year.
export MODEL_DATAFRAME="/path/to/observations_ERA5temp_ponding_COMMON_SUPPORT_2006_2020.parquet"

# If MODEL_DATAFRAME is a GeoPackage instead, use:
#
# export
export MODEL_DATAFRAME="/raid01/mafields/tas/MODELS_filtered/ssp585/jupyter/common_support_era5_melt_augmented_with_mali_floor10kPa/common_support_era5_melt_with_mali_floor10kPa.gpkg"

export MODEL_DATA_LAYER="era5_cell_years_with_mali"


# =============================================================================
# 5. Output
# =============================================================================

# Stable production output directory.
export MODEL_OUT_DIR="results/manuscript_temperature_models"

# To retain each run separately, use:
#
# export MODEL_OUT_DIR="results/manuscript_temperature_models_${SLURM_JOB_ID}"


# =============================================================================
# 6. Observation period and filtering
# =============================================================================

export START_YEAR=2006
export END_YEAR=2020

# Retain all records with valid positive spatial support.
export MIN_COVERAGE_FRACTION=0


# =============================================================================
# 7. Model specification
# =============================================================================

# Beta-Binomial concentration prior:
#
#     kappa ~ Gamma(shape=2, rate=0.1)
#
export KAPPA_ALPHA=2.0
export KAPPA_BETA=0.1

# The earlier case-filtering workflow standardized temperature separately
# after selecting each training sample. "peninsula" reproduces that behavior.
#
# Set to "all_shelves" only if the final manuscript Peninsula model used the
# all-shelf calibration mean and standard deviation.
export PENINSULA_SCALING="peninsula"


# =============================================================================
# 8. Full posterior sampling
# =============================================================================

export CHAINS=4
export CORES=4

# Manuscript production settings: 2,000 tuning iterations and 2,000 retained
# draws per chain.
export DRAWS=2000
export TUNE=2000

export TARGET_ACCEPT=0.95
export MAX_TREEDPTH=12
export RANDOM_SEED=42

# Retain pointwise log likelihood for native-training-sample LOO diagnostics.
# The LOO values for the two final models should not be directly compared
# because the models use different training samples.
export LOG_LIKELIHOOD=1

# Do not silently replace an existing trace.
export OVERWRITE_TRACES=0


# =============================================================================
# 9. Posterior curves and fitted-period diagnostics
# =============================================================================

# Draws used for posterior curves and row-level summaries. This does not
# change the number of MCMC draws used to fit the models.
export MAX_PREDICTION_DRAWS=1000

# Reduce this value if row-level post-processing exceeds available memory.
export PREDICTION_CHUNK_SIZE=5000

# Descriptive quantile bins used for empirical plots.
export N_TEMP_BINS=18

# Bins of fitted risk used for calibration plots.
export CALIBRATION_BIN_COUNT=10

# Common physical-temperature grid for plotting both models.
export CURVE_TEMP_MIN=-18
export CURVE_TEMP_MAX=3
export CURVE_POINTS=600


# =============================================================================
# 10. Connected-ponding thresholds
# =============================================================================

# Fractions, not percentages.
#
# Central estimate: 0.8718%
# Central 68% interval: approximately 0.72%–1.26%
#
export THRESHOLD_CENTRAL=0.008718
export THRESHOLD_LOWER_68=0.0072
export THRESHOLD_UPPER_68=0.0126


# =============================================================================
# 11. Validate environment and inputs
# =============================================================================

if [[ ! -f "${MODEL_SCRIPT}" ]]; then
    echo "ERROR: Python model script does not exist:"
    echo "  ${MODEL_SCRIPT}"
    exit 1
fi

if [[ ! -f "${MODEL_DATAFRAME}" ]]; then
    echo "ERROR: Model dataframe does not exist:"
    echo "  ${MODEL_DATAFRAME}"
    exit 1
fi

echo
echo "Python executable: $(command -v python)"
python --version

echo
echo "Configuration:"
echo "  Script: ${MODEL_SCRIPT}"
echo "  Input: ${MODEL_DATAFRAME}"

if [[ -n "${MODEL_DATA_LAYER:-}" ]]; then
    echo "  Input layer: ${MODEL_DATA_LAYER}"
fi

echo "  Output: ${MODEL_OUT_DIR}"
echo "  Years: ${START_YEAR}-${END_YEAR}"
echo "  Peninsula scaling: ${PENINSULA_SCALING}"
echo "  Chains: ${CHAINS}"
echo "  Cores: ${CORES}"
echo "  Draws per chain: ${DRAWS}"
echo "  Tuning iterations per chain: ${TUNE}"
echo "  Total retained draws: $((CHAINS * DRAWS))"
echo "  Target acceptance: ${TARGET_ACCEPT}"
echo "  Maximum tree depth: ${MAX_TREEDPTH}"
echo "  Log likelihood: ${LOG_LIKELIHOOD}"
echo "  Kappa prior: Gamma(shape=${KAPPA_ALPHA}, rate=${KAPPA_BETA})"
echo "  Threshold: ${THRESHOLD_CENTRAL}"
echo "  Threshold 68% interval: ${THRESHOLD_LOWER_68}-${THRESHOLD_UPPER_68}"
echo


# =============================================================================
# 12. Record software environment
# =============================================================================

mkdir -p "${MODEL_OUT_DIR}"

python - <<'PY'
import platform

print("Python:", platform.python_version())

packages = [
    "arviz",
    "matplotlib",
    "numpy",
    "pandas",
    "pymc",
    "pytensor",
]

for package_name in packages:
    try:
        package = __import__(package_name)
        print(
            f"{package_name}:",
            getattr(package, "__version__", "unknown"),
        )
    except Exception as error:
        print(f"{package_name}: unavailable ({error!r})")
PY

python -m pip freeze \
    > "${MODEL_OUT_DIR}/python_environment_${SLURM_JOB_ID}.txt"


# =============================================================================
# 13. Run
# =============================================================================

START_SECONDS=$(date +%s)

srun \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    python -u "${MODEL_SCRIPT}"

END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))


# =============================================================================
# 14. Completion report
# =============================================================================

echo
echo "======================================================================"
echo "RUN COMPLETE"
echo "======================================================================"
echo "End: $(date)"
echo "Elapsed seconds: ${ELAPSED_SECONDS}"

printf "Elapsed time: %02d:%02d:%02d\n" \
    $((ELAPSED_SECONDS / 3600)) \
    $(((ELAPSED_SECONDS % 3600) / 60)) \
    $((ELAPSED_SECONDS % 60))

echo "Output: ${MODEL_OUT_DIR}"
echo
echo "Expected trace files:"
echo "  ${MODEL_OUT_DIR}/traces/all_shelves_beta_binomial.nc"
echo "  ${MODEL_OUT_DIR}/traces/peninsula_binomial.nc"
echo
echo "Expected metadata:"
echo "  ${MODEL_OUT_DIR}/analysis_metadata.json"
echo "======================================================================"