# Tracker reconstruction

This workflow converts the six Paper 1 tracker-hit arrays to EDM4hep and runs MAIA v3 tracker reconstruction.

The reconstruction uses the following settings:

- `IsStrip = False`
- `ForceHitsOntoSurface = True`
- CKF tracking enabled
- duplicate removal enabled
- output track collection: `SiTracks`

1. Clone GenBIB-ML and the tested benchmark version.

   ```bash
   git clone https://github.com/ShiyuP1/GenBIB-ML.git
   cd GenBIB-ML
   git clone --recurse-submodules https://github.com/MuonColliderSoft/mucoll-benchmarks.git
   git -C mucoll-benchmarks checkout 298d68ac9466f21387a03727b0a85902984fa394
   git -C mucoll-benchmarks submodule update --init --recursive
   ```

2. Use the MAIA v3 image. Set `IMAGE` to an existing SIF file or pull it with Apptainer.

   ```bash
   apptainer pull mucoll-sim-ubuntu24_v3.0-amd64.sif docker://ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
   ```

3. Put the generated arrays in one directory using these names.

   ```text
   tabddpm_VBC_samples.npy
   tabddpm_VEC_samples.npy
   tabddpm_ITBC_samples.npy
   tabddpm_ITEC_samples.npy
   tabddpm_OTBC_samples.npy
   tabddpm_OTEC_samples.npy
   ```

4. Assign MAIA CellIDs and convert the arrays to EDM4hep. Replace the three paths at the top before running the commands.

   ```bash
   IMAGE=/path/to/mucoll-sim-ubuntu24_v3.0-amd64.sif
   INPUT_NPY_DIR=/path/to/generated_arrays
   OUTPUT_DIR=/path/to/output

   mkdir -p "$OUTPUT_DIR/assigned_ID"
   apptainer exec --bind "$PWD:/work/genbib,$INPUT_NPY_DIR:/work/input,$OUTPUT_DIR:/work/output" "$IMAGE" bash -lc 'source /opt/setup_mucoll.sh && cd /work/genbib/mucoll-benchmarks && source setup_config.sh /work/genbib/mucoll-benchmarks MAIA_v0 && python3 /work/genbib/reco/assign_actual_cellid.py --input-dir /work/input --output-dir /work/output/assigned_ID --input-format 9col --geomap-path /work/output/maia_cellid_sensor_geometry.npz && python3 /work/genbib/reco/numpy_to_edm4hep.py --samples-path /work/output/assigned_ID --output-path /work/output/generated_hits.edm4hep.root --num-events 1'
   ```

5. Set the four paths and run tracker reconstruction.

   ```bash
   export IMAGE=/path/to/mucoll-sim-ubuntu24_v3.0-amd64.sif
   export BENCHMARK_DIR="$PWD/mucoll-benchmarks"
   export INPUT_FILE=/path/to/output/generated_hits.edm4hep.root
   export OUTPUT_DIR=/path/to/reco_output
   bash reco/run_reco.sh
   ```

   The reconstructed tracks are written to `reco_output.edm4hep.root` in the selected output directory.
