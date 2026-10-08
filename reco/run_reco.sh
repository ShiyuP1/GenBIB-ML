#!/bin/bash
set -e

if [ "$#" -ne 7 ]; then
    echo "Usage: $0 BENCHMARK_DIR INPUT_DIR RECO_ROOT WRITE_FRACTION SAMPLE_MODE RANDOM_SEED NUM_EVENTS" >&2
    exit 1
fi

BENCHMARK_DIR="$1"
INPUT_DIR="$2"
RECO_ROOT="$3"
WRITE_FRACTION="$4"
SAMPLE_MODE="$5"
RANDOM_SEED="$6"
NUM_EVENTS="$7"

RECO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIGI_DIR="$RECO_ROOT/digi"
TRACK_DIR="$RECO_ROOT/reco"
PLOT_DIR="$RECO_ROOT/plot"
ASSIGNED_DIR="$DIGI_DIR/assigned_ID"
GEOMAP_PATH="$DIGI_DIR/maia_cellid_sensor_geometry.npz"
EDM4HEP_INPUT="$DIGI_DIR/generated_hits.edm4hep.root"
DIGI_OUTPUT="$DIGI_DIR/digi_output.edm4hep.root"
RECO_OUTPUT="$TRACK_DIR/reco_output.edm4hep.root"

for path in "$BENCHMARK_DIR" "$INPUT_DIR" "$RECO_DIR/tracker_reco_override.py"; do
    if [ ! -e "$path" ]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
done

for input in "$INPUT_DIR"/{VBC,VEC,ITBC,ITEC,OTBC,OTEC}_sample.npy; do
    if [ ! -f "$input" ]; then
        echo "Missing required sample: $input" >&2
        exit 1
    fi
done

for output in "$EDM4HEP_INPUT" "$DIGI_OUTPUT" "$RECO_OUTPUT"; do
    if [ -e "$output" ]; then
        echo "Refusing to overwrite existing output: $output" >&2
        exit 1
    fi
done

if [ -d "$ASSIGNED_DIR" ] && find "$ASSIGNED_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty assigned-ID directory: $ASSIGNED_DIR" >&2
    exit 1
fi
mkdir -p "$ASSIGNED_DIR" "$TRACK_DIR" "$PLOT_DIR"

source /opt/setup_mucoll.sh
source <(sed 's/\r$//' "$BENCHMARK_DIR/setup_config.sh") "$BENCHMARK_DIR" MAIA_v0

python3 "$RECO_DIR/assign_actual_cellid.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$ASSIGNED_DIR" \
    --input-format 9col \
    --write-fraction "$WRITE_FRACTION" \
    --sample-mode "$SAMPLE_MODE" \
    --random-seed "$RANDOM_SEED" \
    --geomap-path "$GEOMAP_PATH"

python3 "$RECO_DIR/numpy_to_edm4hep.py" \
    --samples-path "$ASSIGNED_DIR" \
    --output-path "$EDM4HEP_INPUT" \
    --num-events "$NUM_EVENTS"

cd "$DIGI_DIR"
k4run "$RECO_DIR/tracker_reco_override.py" --stage digi -n "$NUM_EVENTS" \
    --inputFiles "$EDM4HEP_INPUT" --outputFile "$DIGI_OUTPUT"
cd "$TRACK_DIR"
k4run "$RECO_DIR/tracker_reco_override.py" --stage reco -n "$NUM_EVENTS" \
    --inputFiles "$DIGI_OUTPUT" --outputFile "$RECO_OUTPUT"

python3 - "$RECO_OUTPUT" <<'PY'
from podio import root_io
import sys

event = next(root_io.Reader(sys.argv[1]).get("events"))
names = ["SiTracks", "SiTracksPreFit", "SiTracks_Refitted", "SelectedTracks"]
counts = {name: len(event.get(name)) for name in names}
print("Track collections:", counts)
if counts["SelectedTracks"] == 0:
    raise RuntimeError("SelectedTracks is empty")
PY

echo "Reco pipeline complete: $RECO_OUTPUT"
