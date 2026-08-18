"""Generate Paper 1 TabDDPM samples for one detector collection."""

import gc
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


# Edit this section.
COLLECTION = "ITBC"
PAPER1_CODE_ROOT = Path(
    "/oscar/data/mleblan6/mucoll/speng44/bib_gen_model/"
    "ddpm_outputs/tabddpm/local_phi/paper1-inference"
)
MODEL_DIR = Path(
    "/oscar/data/mleblan6/mucoll/speng44/bib_gen_model/"
    "ddpm_outputs/tabddpm/local_phi/"
    "ITBC_TABDDPM_local_phi_cond-side-layer-module-sensor_"
    "t1000_s300000_h4096x4096x4096x4096x4096x4096_dim2048_b4096"
)
CONDITIONS_FILE = None  # Path("conditions.npy") or None for a 16-row smoke test
OUTPUT_FILE = Path("generated_samples.npy")
OVERSAMPLE = 1
SEED = 8
MAX_REJECTION_ROUNDS = 10000
DEVICE = "auto"  # "auto", "cuda", or "cpu"


COLLECTIONS = {
    "VBC": (1, "VertexBarrelCollection"),
    "VEC": (2, "VertexEndcapCollection"),
    "ITBC": (3, "InnerTrackerBarrelCollection"),
    "ITEC": (4, "InnerTrackerEndcapCollection"),
    "OTBC": (5, "OuterTrackerBarrelCollection"),
    "OTEC": (6, "OuterTrackerEndcapCollection"),
}

FEATURE_ORDER = [0, 4, 1, 2, 3, 6, 7, 8, 9]
FEATURE_INDICES = {
    "r": 2,
    "phi": 3,
    "z": 4,
    "side": 5,
    "layer": 6,
    "module": 7,
    "sensor": 8,
}


def require_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_directory(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_paper1(paper1_root):
    tabddpm_root = paper1_root / "diffusion" / "tabddpm_official"
    scripts_dir = tabddpm_root / "scripts"
    sampler_path = require_file(scripts_dir / "sample.py", "Paper 1 TabDDPM sampler")

    for path in (scripts_dir, tabddpm_root, paper1_root):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)

    spec = importlib.util.spec_from_file_location("paper1_tabddpm_sample", sampler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from helpers.data_transforms import inverse_geometry_transform
    from helpers.flow import build_xy_z_lookup, snap_z_to_detector_xy
    from helpers.material_map import apply_material_map_hybrid

    return (
        module.sample,
        inverse_geometry_transform,
        build_xy_z_lookup,
        snap_z_to_detector_xy,
        apply_material_map_hybrid,
    )


def load_conditions(path, system_id, y_lookup):
    if path is None:
        conditions4 = y_lookup[: min(16, len(y_lookup))]
        return np.column_stack(
            [np.full(len(conditions4), system_id, dtype=np.int64), conditions4]
        )

    conditions = np.load(require_file(path, "conditions file"))
    if conditions.ndim != 2 or conditions.shape[1] != 5:
        raise ValueError(
            "Conditions must have shape (N, 5): "
            "system_id, side, layer, module, sensor"
        )
    if not np.all(np.isfinite(conditions)) or not np.all(conditions == np.rint(conditions)):
        raise ValueError("Conditions must contain finite integer values")

    conditions = conditions.astype(np.int64)
    wrong_system = conditions[:, 0] != system_id
    if np.any(wrong_system):
        found = np.unique(conditions[wrong_system, 0]).tolist()
        raise ValueError(f"Expected system_id {system_id}; found {found}")
    return conditions


def map_conditions_to_classes(conditions, y_lookup):
    class_by_condition = {tuple(row): index for index, row in enumerate(y_lookup)}
    class_ids = np.empty(len(conditions), dtype=np.int64)
    missing = []

    for index, row in enumerate(conditions[:, 1:]):
        key = tuple(row)
        if key not in class_by_condition:
            missing.append(conditions[index].tolist())
        else:
            class_ids[index] = class_by_condition[key]

    if missing:
        raise ValueError(f"Conditions not present in y_lookup.npy: {missing[:10]}")
    return class_ids


def build_endcap_z_lookup(
    model_dir,
    y_lookup,
    collection_name,
    inverse_geometry_transform,
    build_xy_z_lookup,
):
    reference_parts = []
    dataset_dir = model_dir / "dataset"

    for split in ("train", "val"):
        features = np.load(
            require_file(dataset_dir / f"X_num_{split}.npy", f"dataset X_num_{split}.npy")
        ).astype(np.float32)
        class_ids = np.load(
            require_file(dataset_dir / f"y_{split}.npy", f"dataset y_{split}.npy")
        ).astype(np.int64).reshape(-1)
        local_hits = np.column_stack([features, y_lookup[class_ids]]).astype(np.float32)
        reference_parts.append(
            inverse_geometry_transform(
                local_hits,
                "local_phi",
                collection_name,
                FEATURE_ORDER,
            )
        )

    reference_hits = np.concatenate(reference_parts)
    z_lookup = build_xy_z_lookup(
        reference_hits,
        FEATURE_INDICES["side"],
        FEATURE_INDICES["layer"],
        FEATURE_INDICES["r"],
        FEATURE_INDICES["phi"],
        FEATURE_INDICES["z"],
    )
    del reference_parts, reference_hits
    gc.collect()
    return z_lookup


def main():
    if COLLECTION not in COLLECTIONS:
        raise ValueError(f"COLLECTION must be one of: {', '.join(COLLECTIONS)}")
    if OVERSAMPLE < 1:
        raise ValueError("OVERSAMPLE must be at least 1")

    system_id, collection_name = COLLECTIONS[COLLECTION]
    paper1_root = require_directory(PAPER1_CODE_ROOT, "Paper 1 repository")
    model_dir = require_directory(MODEL_DIR, "model directory")
    model_path = require_file(model_dir / "model.pt", "model.pt")
    config_path = require_file(model_dir / "run_config.json", "run_config.json")
    lookup_path = require_file(model_dir / "y_lookup.npy", "y_lookup.npy")
    info_path = require_file(model_dir / "dataset" / "info.json", "dataset/info.json")

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    with info_path.open(encoding="utf-8") as file:
        dataset_info = json.load(file)

    if config["BASIS"] != "local_phi" or config["Y_MODE"] != "cond":
        raise ValueError("This interface requires a local_phi, condition-trained model")

    y_lookup = np.load(lookup_path).astype(np.int64)
    if y_lookup.ndim != 2 or y_lookup.shape[1] != 4:
        raise ValueError(f"Expected y_lookup.npy shape (N, 4); found {y_lookup.shape}")

    conditions = load_conditions(CONDITIONS_FILE, system_id, y_lookup)
    class_ids = map_conditions_to_classes(conditions, y_lookup)
    conditions = np.repeat(conditions, OVERSAMPLE, axis=0)
    class_ids = np.repeat(class_ids, OVERSAMPLE)

    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    (
        tabddpm_sample,
        inverse_geometry_transform,
        build_xy_z_lookup,
        snap_z_to_detector_xy,
        apply_material_map_hybrid,
    ) = load_paper1(paper1_root)

    model_params = {
        "num_classes": int(config["n_classes"]),
        "is_y_cond": bool(config["is_y_cond"]),
        "rtdl_params": {
            "d_layers": [int(value) for value in config["D_LAYERS"].split(",")],
            "dropout": 0.0,
        },
        "dim_t": int(config["DIM_T"]),
    }
    transform_config = {
        "seed": int(config["SEED"]),
        "normalization": config["NORMALIZATION"],
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }

    output_path = OUTPUT_FILE.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z_lookup = None
    if "Endcap" in collection_name:
        z_lookup = build_endcap_z_lookup(
            model_dir,
            y_lookup,
            collection_name,
            inverse_geometry_transform,
            build_xy_z_lookup,
        )

    num_features = int(dataset_info["n_num_features"])
    output = np.empty((len(class_ids), num_features + 4), dtype=np.float32)
    unfilled = np.ones(len(class_ids), dtype=bool)

    print(f"Collection: {collection_name}")
    print(f"Conditions: {len(class_ids):,}")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory(prefix="genbib_tabddpm_", dir=output_path.parent) as work_dir:
        sample_job = {
            "parent_dir": work_dir,
            "real_data_path": str(model_dir / "dataset"),
            "batch_size": int(config["SAMPLE_BATCH_SIZE"]),
            "model_type": "mlp",
            "model_params": model_params,
            "model_path": str(model_path),
            "num_timesteps": int(config["NUM_TIMESTEPS"]),
            "gaussian_loss_type": "mse",
            "scheduler": config["SCHEDULER"],
            "T_dict": transform_config,
            "num_numerical_features": num_features,
            "disbalance": None,
            "device": device,
            "change_val": False,
        }

        for round_id in range(MAX_REJECTION_ROUNDS):
            remaining = np.flatnonzero(unfilled)
            if len(remaining) == 0:
                break

            requested_ids = class_ids[remaining]
            print(
                f"Round {round_id + 1}: {len(remaining):,}/"
                f"{len(class_ids):,} remaining",
                flush=True,
            )
            tabddpm_sample(
                **sample_job,
                num_samples=len(remaining),
                seed=SEED + round_id,
                y_to_sample=requested_ids,
            )

            generated_features = np.load(Path(work_dir) / "X_num_train.npy").astype(
                np.float32
            )
            returned_ids = np.load(Path(work_dir) / "y_train.npy").astype(
                np.int64
            ).reshape(-1)
            if not np.array_equal(returned_ids, requested_ids):
                raise RuntimeError("TabDDPM changed the requested condition order")

            generated_conditions = y_lookup[returned_ids]
            generated_hits = np.column_stack(
                [generated_features, generated_conditions]
            ).astype(np.float32)
            generated_hits = inverse_geometry_transform(
                generated_hits,
                "local_phi",
                collection_name,
                FEATURE_ORDER,
            )

            if z_lookup is not None:
                generated_hits[:, FEATURE_INDICES["z"]] = snap_z_to_detector_xy(
                    generated_hits[:, FEATURE_INDICES["r"]],
                    generated_hits[:, FEATURE_INDICES["phi"]],
                    generated_hits[:, FEATURE_INDICES["side"]],
                    generated_hits[:, FEATURE_INDICES["layer"]],
                    z_lookup,
                )

            mask = np.asarray(
                apply_material_map_hybrid(
                    {collection_name: generated_hits},
                    None,
                    collection_name,
                    FEATURE_INDICES,
                ),
                dtype=bool,
            )
            passing_slots = remaining[mask]
            output[passing_slots] = generated_hits[mask]
            unfilled[passing_slots] = False

            del generated_features, returned_ids, generated_conditions, generated_hits, mask
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if np.any(unfilled):
        failed_path = output_path.with_name(f"{output_path.stem}_unfilled_conditions.npy")
        np.save(failed_path, conditions[unfilled])
        raise RuntimeError(
            f"{unfilled.sum()} conditions remain after {MAX_REJECTION_ROUNDS} rounds; "
            f"saved to {failed_path}"
        )

    np.save(output_path, output)
    print(f"Saved {output.shape} to {output_path}")


if __name__ == "__main__":
    main()


