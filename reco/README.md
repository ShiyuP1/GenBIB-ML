# Tracker reconstruction

Uses `mucoll-sim-ubuntu24:v3.0`, `mucoll-benchmarks` commit
`298d68ac9466f21387a03727b0a85902984fa394`, and the Paper1-aligned tracker
settings.

## 1. Prepare samples

```text
sample/
├── VBC_sample.npy
├── VEC_sample.npy
├── ITBC_sample.npy
├── ITEC_sample.npy
├── OTBC_sample.npy
└── OTEC_sample.npy
```

## 2. Prepare the benchmark

```bash
git clone https://github.com/MuonColliderSoft/mucoll-benchmarks.git
git -C mucoll-benchmarks checkout 298d68ac9466f21387a03727b0a85902984fa394
git -C mucoll-benchmarks submodule update --init --recursive
```

## 3. Run

```bash
sbatch --export=ALL,INPUT_DIR=/path/to/reco/sample reco/submit_oscar_reco.sbatch
```

## 4. Outputs

```text
reco/
├── sample/
├── digi/
├── reco/
│   └── reco_output.edm4hep.root
└── plot/
```

The job runs one event by default and refuses to overwrite existing outputs.
It uses `IsStrip=False`, `ForceHitsOntoSurface=True`, at least eight pre-fit
hits, `pT >= 0.5`, and refit reduced chi2 <= 3.
