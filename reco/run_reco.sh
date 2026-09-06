#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif}"
BENCHMARK_DIR="${BENCHMARK_DIR:-/path/to/mucoll-benchmarks}"
INPUT_FILE="${INPUT_FILE:-/path/to/generated_hits.edm4hep.root}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/reco_output}"
NUM_EVENTS="${NUM_EVENTS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for path in "${IMAGE}" "${BENCHMARK_DIR}" "${INPUT_FILE}" "${SCRIPT_DIR}/tracker_reco_override.py"; do
    if [ ! -e "${path}" ]; then
        echo "Missing required path: ${path}" >&2
        exit 1
    fi
done

INPUT_DIR="$(cd "$(dirname "${INPUT_FILE}")" && pwd)"
INPUT_NAME="$(basename "${INPUT_FILE}")"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

apptainer exec --bind "${SCRIPT_DIR}:/work/genbib-reco,${BENCHMARK_DIR}:/work/mucoll-benchmarks,${INPUT_DIR}:/work/input,${OUTPUT_DIR}:/work/output" "${IMAGE}" bash -lc '
    set -euo pipefail
    set +e +u +o pipefail
    source /opt/setup_mucoll.sh 2>/dev/null || true

    STACK_SETUP="$(find /opt/spack/opt/spack -name setup.sh -path "*mucoll-stack*" 2>/dev/null | head -n1)"
    if [ -n "${STACK_SETUP}" ]; then
        source "${STACK_SETUP}"
    fi

    SETUP_CONFIG="$(mktemp)"
    sed "s/\r$//" /work/mucoll-benchmarks/setup_config.sh > "${SETUP_CONFIG}"
    source "${SETUP_CONFIG}" /work/mucoll-benchmarks MAIA_v0

    if [ ! -f "${MUCOLL_GEO:-}" ]; then
        CONFIG_PATH=/work/mucoll-benchmarks/configs/MAIAConfig
        CONFIG_PACKAGE_PATH="${CONFIG_PATH}/MAIAConfig"
        MUCOLL_GEO="$(find /opt/spack/opt/spack -path "*/k4geo*/share/k4geo/MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml" 2>/dev/null | head -n1)"
        MUCOLL_MATMAP="$(find /opt/spack/opt/spack -path "*/share/*/data/MAIA_v0_material.json" 2>/dev/null | head -n1)"
        export MUCOLL_GEOM_NAME=MAIA_v0
        export MUCOLL_GEO
        export MUCOLL_MATMAP
        export MUCOLL_CONFIG="${CONFIG_PATH}"
        export MUCOLL_CONFIG_NAME=MAIAConfig
        export PYTHONPATH="${CONFIG_PACKAGE_PATH}:${CONFIG_PATH}:/work/mucoll-benchmarks/common:${PYTHONPATH:-}"
    fi

    set -euo pipefail
    cd /work/output
    k4run /work/genbib-reco/tracker_reco_override.py --stage digi -n "$2" --inputFiles "/work/input/$1" --outputFile /work/output/digi_output.edm4hep.root
    k4run /work/genbib-reco/tracker_reco_override.py --stage reco -n "$2" --inputFiles /work/output/digi_output.edm4hep.root --outputFile /work/output/reco_output.edm4hep.root
' _ "${INPUT_NAME}" "${NUM_EVENTS}"

echo "Reconstruction output: ${OUTPUT_DIR}/reco_output.edm4hep.root"
