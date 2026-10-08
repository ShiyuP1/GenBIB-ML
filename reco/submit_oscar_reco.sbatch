#!/bin/bash
#SBATCH --job-name=genbib-reco
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shiyu_peng@brown.edu

set -euo pipefail

PROJECT_ROOT=/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_muon
GENBIB_DIR="$PROJECT_ROOT/GenBIB-ML"
IMAGE=/oscar/home/speng44/data/mucoll/mucoll-sim-ubuntu24:v3.0.sif
BENCHMARK_DIR="$GENBIB_DIR/mucoll-benchmarks"
BENCHMARK_COMMIT=298d68ac9466f21387a03727b0a85902984fa394

INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the directory containing the six sample files}"
NUM_EVENTS="${NUM_EVENTS:-1}"
WRITE_FRACTION="${WRITE_FRACTION:-1.0}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
RANDOM_SEED="${RANDOM_SEED:-12345}"

INPUT_DIR="$(readlink -f "$INPUT_DIR")"
RECO_ROOT="$(dirname "$INPUT_DIR")"

for path in "$PROJECT_ROOT" "$GENBIB_DIR" "$IMAGE" "$BENCHMARK_DIR" "$INPUT_DIR"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
done

actual_benchmark_commit="$(git -C "$BENCHMARK_DIR" rev-parse HEAD)"
if [[ "$actual_benchmark_commit" != "$BENCHMARK_COMMIT" ]]; then
    echo "Unexpected mucoll-benchmarks commit: $actual_benchmark_commit" >&2
    echo "Expected: $BENCHMARK_COMMIT" >&2
    exit 1
fi

collections=(VBC VEC ITBC ITEC OTBC OTEC)
for collection in "${collections[@]}"; do
    if [[ ! -f "$INPUT_DIR/${collection}_sample.npy" ]]; then
        echo "Missing required sample: $INPUT_DIR/${collection}_sample.npy" >&2
        exit 1
    fi
done

echo "Input directory:  $INPUT_DIR"
echo "Reco root:        $RECO_ROOT"
echo "Events:           $NUM_EVENTS"
echo "Write fraction:   $WRITE_FRACTION"
echo "Sample mode:      $SAMPLE_MODE"
echo "Random seed:      $RANDOM_SEED"
echo "Image:            $IMAGE"
echo "Benchmark commit: $actual_benchmark_commit"

apptainer exec --bind "$PROJECT_ROOT:$PROJECT_ROOT" --bind "$RECO_ROOT:$RECO_ROOT" "$IMAGE" \
    bash "$GENBIB_DIR/reco/run_reco.sh" \
    "$BENCHMARK_DIR" "$INPUT_DIR" "$RECO_ROOT" "$WRITE_FRACTION" \
    "$SAMPLE_MODE" "$RANDOM_SEED" "$NUM_EVENTS"
