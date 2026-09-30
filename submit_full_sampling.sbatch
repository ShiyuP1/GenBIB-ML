#!/usr/bin/env bash
#SBATCH --job-name=genbib-full
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%2
#SBATCH --output=genbib-full_%A_%a.out
#SBATCH --error=genbib-full_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/oscar/home/speng44/data/mucoll/speng44/bib_gen_mother_muon
GENBIB_DIR="$PROJECT_ROOT/GenBIB-ML"
MODEL_ROOT="$PROJECT_ROOT/new_diffusion_norm1"

PAPER1_CODE_ROOT="$PROJECT_ROOT/paper1interface"
PYTHON=/oscar/home/speng44/work/venv_paper1_test/bin/python

D_LAYERS=2048,2048,2048,2048,2048
DIM_T=1024
NUM_TIMESTEPS=1000
SAMPLE_BATCH_SIZE=4096
NORMALIZATION=quantile
SCHEDULER=cosine
SEED=42
MAX_REJECTION_ROUNDS=100
OVERSAMPLE=1
OUTPUT_NAME=generated_samples_full

COLLECTIONS=(VBC VEC ITBC ITEC OTBC OTEC)
COLLECTION=${COLLECTIONS[$SLURM_ARRAY_TASK_ID]}

mapfile -t MODEL_MATCHES < <(
    find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${COLLECTION}_TABDDPM_*" -print
)
if test "${#MODEL_MATCHES[@]}" -ne 1
then
    printf 'ERROR: expected one %s model under %s; found %s\n' \
        "$COLLECTION" "$MODEL_ROOT" "${#MODEL_MATCHES[@]}" >&2
    printf '%s\n' "${MODEL_MATCHES[@]}" >&2
    exit 1
fi

MODEL_DIR=${MODEL_MATCHES[0]}
SAMPLE_DIR="$MODEL_DIR/sample"
CONDITIONS_FILE="$SAMPLE_DIR/full_conditions.npy"
OUTPUT_FILE="$SAMPLE_DIR/${OUTPUT_NAME}.npy"
mkdir -p "$SAMPLE_DIR"

"$PYTHON" "$GENBIB_DIR/sample.py" \
    --collection "$COLLECTION" \
    --model-dir "$MODEL_DIR" \
    --paper1-code-root "$PAPER1_CODE_ROOT" \
    --conditions-file "$CONDITIONS_FILE" \
    --output-file "$OUTPUT_FILE" \
    --device cuda \
    --seed "$SEED" \
    --max-rejection-rounds "$MAX_REJECTION_ROUNDS" \
    --d-layers "$D_LAYERS" \
    --dim-t "$DIM_T" \
    --num-timesteps "$NUM_TIMESTEPS" \
    --sample-batch-size "$SAMPLE_BATCH_SIZE" \
    --normalization "$NORMALIZATION" \
    --scheduler "$SCHEDULER" \
    --oversample "$OVERSAMPLE"
