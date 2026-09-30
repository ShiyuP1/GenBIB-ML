# GenBIB-ML

Generate Paper 1-aligned tracker BIB samples and run tracker reconstruction.

## Full sampling on Oscar

`sample.py` uses the Paper 1 TabDDPM sampler, inverse geometry transform,
endcap z snapping, and material-map rejection. Its output columns are:

```text
logE, time, r, phi, z, side, layer, module, sensor
```

By default, `sample.py` reconstructs the complete condition sequence directly
from each model's `dataset/y_train.npy`, `dataset/y_val.npy`, and
`y_lookup.npy`. This mirrors the normal Paper 1 train-to-evaluation flow and
requires no separately prepared condition file. An optional
`--conditions-file` can still provide an external Paper 1 context array with
columns `system_id, side, layer, module, sensor`.

Set `PROJECT_ROOT`, `MODEL_ROOT`, and `OUTPUT_NAME` near the top of
`submit_full_sampling.sbatch`, then submit:

```bash
sbatch submit_full_sampling.sbatch
```

The Paper 1 interface is expected at `PROJECT_ROOT/paper1interface`.

Each full sample is written inside its model directory:

```text
VBC_TABDDPM_.../sample/generated_samples_full.npy
VEC_TABDDPM_.../sample/generated_samples_full.npy
ITBC_TABDDPM_.../sample/generated_samples_full.npy
ITEC_TABDDPM_.../sample/generated_samples_full.npy
OTBC_TABDDPM_.../sample/generated_samples_full.npy
OTEC_TABDDPM_.../sample/generated_samples_full.npy
```

The same script supports norm1 and norm42 by changing `MODEL_ROOT`. All model
and sampling parameters are explicit in the sbatch file. `OUTPUT_NAME` selects
the output filename and permits intentional replacement or side-by-side runs.
After 100 rejection rounds, any conditions that still fail the material mask
are written to a separate `*_dropped_conditions.npy` file and omitted from the
generated sample.

Tracker reconstruction instructions are in [`reco/README.md`](reco/README.md).
