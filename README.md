# GenBIB-ML

Generate Paper 1-aligned tracker BIB samples and run tracker reconstruction.

## Full sampling on Oscar

`sample.py` uses the Paper 1 TabDDPM sampler, inverse geometry transform,
endcap z snapping, and material-map rejection. Its output columns are:

```text
logE, time, r, phi, z, side, layer, module, sensor
```

Before submission, prepare one Paper 1 condition array in each model directory:

```text
VBC_TABDDPM_.../sample/full_conditions.npy
VEC_TABDDPM_.../sample/full_conditions.npy
ITBC_TABDDPM_.../sample/full_conditions.npy
ITEC_TABDDPM_.../sample/full_conditions.npy
OTBC_TABDDPM_.../sample/full_conditions.npy
OTEC_TABDDPM_.../sample/full_conditions.npy
```

Each array has columns `system_id, side, layer, module, sensor`. Repeated rows
request repeated samples with the same detector condition.

Set `MODEL_ROOT` near the top of `submit_full_sampling.sbatch`, then submit:

```bash
sbatch submit_full_sampling.sbatch
```

Each full sample is written inside its model directory:

```text
VBC_TABDDPM_.../sample/generated_samples_full.npy
VEC_TABDDPM_.../sample/generated_samples_full.npy
ITBC_TABDDPM_.../sample/generated_samples_full.npy
ITEC_TABDDPM_.../sample/generated_samples_full.npy
OTBC_TABDDPM_.../sample/generated_samples_full.npy
OTEC_TABDDPM_.../sample/generated_samples_full.npy
```

The same script supports norm1 and norm42 by changing only `MODEL_ROOT`. All
model and sampling parameters are explicit in the sbatch file. The sampler
exits with an error instead of saving an incomplete sample.

Tracker reconstruction instructions are in [`reco/README.md`](reco/README.md).
