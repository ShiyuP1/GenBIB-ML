# GenBIB-ML

Generate Paper 1-aligned tracker BIB samples and run tracker reconstruction.

## Full sampling on Oscar

`sample.py` uses the Paper 1 TabDDPM sampler, inverse geometry transform,
endcap z snapping, and material-map rejection. Its output columns are:

```text
logE, time, r, phi, z, side, layer, module, sensor
```

Before submission, prepare one Paper 1 condition array per collection:

```text
full_conditions/VBC_conditions.npy
full_conditions/VEC_conditions.npy
full_conditions/ITBC_conditions.npy
full_conditions/ITEC_conditions.npy
full_conditions/OTBC_conditions.npy
full_conditions/OTEC_conditions.npy
```

Each array has columns `system_id, side, layer, module, sensor`. Repeated rows
request repeated samples with the same detector condition.

Set `MODEL_ROOT` and `OUTPUT_DIR` near the top of
`submit_full_sampling.sbatch`, then submit:

```bash
sbatch submit_full_sampling.sbatch
```

The six outputs are written as:

```text
tabddpm_VBC_samples.npy
tabddpm_VEC_samples.npy
tabddpm_ITBC_samples.npy
tabddpm_ITEC_samples.npy
tabddpm_OTBC_samples.npy
tabddpm_OTEC_samples.npy
```

The same script supports norm1 and norm42 by changing `MODEL_ROOT` and
`OUTPUT_DIR`. All model and sampling parameters are explicit in the sbatch
file. The sampler exits with an error instead of saving an incomplete sample.

Tracker reconstruction instructions are in [`reco/README.md`](reco/README.md).
