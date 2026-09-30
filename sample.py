"""Generate Paper 1-aligned TabDDPM samples for one tracker collection."""

import argparse
import gc
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one Paper 1-aligned sample per supplied condition row."
    )
    parser.add_argument("--collection", required=True, choices=COLLECTIONS)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--paper1-code-root", required=True, type=Path)
    parser.add_argument("--conditions-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-rejection-rounds", required=True, type=int)
    parser.add_argument("--d-layers", required=True)
    parser.add_argument("--dim-t", required=True, type=int)
    parser.add_argument("--num-timesteps", required=True, type=int)
    parser.add_argument("--sample-batch-size", required=True, type=int)
    parser.add_argument(
        "--normalization",
        required=True,
        choices=("quantile", "standard", "minmax"),
    )
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--oversample", default=1, type=int)
    return parser.parse_args()


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


def parse_layers(value):
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--d-layers must be a comma-separated list of integers") from error
    if not layers or any(width <= 0 for width in layers):
        raise ValueError("--d-layers must contain positive integers")
    return layers


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


def load_conditions(path, system_id, y_lookup, oversample):
    conditions = np.load(require_file(path, "conditions file"), allow_pickle=False)
    if conditions.ndim != 2 or conditions.shape[1] != 5:
        raise ValueError(
            "Conditions must have shape (N, 5): "
            "system_id, side, layer, module, sensor"
        )
    if len(conditions) == 0:
        raise ValueError("Conditions file is empty")
    if not np.all(np.isfinite(conditions)) or not np.all(conditions == np.rint(conditions)):
        raise ValueError("Conditions must contain finite integer values")

    conditions = conditions.astype(np.int64)
    wrong_system = conditions[:, 0] != system_id
    if np.any(wrong_system):
        found = np.unique(conditions[wrong_system, 0]).tolist()
        raise ValueError(f"Expected system_id {system_id}; found {found}")

    class_by_condition = {tuple(row.tolist()): index for index, row in enumerate(y_lookup)}
    class_ids = np.empty(len(conditions), dtype=np.int64)
    missing = []
    for index, row in enumerate(conditions[:, 1:]):
        key = tuple(row.tolist())
        class_id = class_by_condition.get(key)
        if class_id is None:
            missing.append(conditions[index].tolist())
        else:
            class_ids[index] = class_id
    if missing:
        raise ValueError(f"Conditions not present in y_lookup.npy: {missing[:10]}")

    if oversample < 1:
        raise ValueError("--oversample must be at least 1")
    if oversample > 1:
        conditions = np.repeat(conditions, oversample, axis=0)
        class_ids = np.repeat(class_ids, oversample)
    return conditions, class_ids


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
            require_file(dataset_dir / f"X_num_{split}.npy", f"X_num_{split}.npy"),
            allow_pickle=False,
        ).astype(np.float32)
        class_ids = np.load(
            require_file(dataset_dir / f"y_{split}.npy", f"y_{split}.npy"),
            allow_pickle=False,
        ).astype(np.int64).reshape(-1)
        if len(features) != len(class_ids):
            raise ValueError(f"Feature/condition length mismatch in {split} split")
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


def save_atomic(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}.",
            suffix=".npy",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, array)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    args = parse_args()
    system_id, collection_name = COLLECTIONS[args.collection]
    paper1_root = require_directory(args.paper1_code_root, "Paper 1 repository")
    model_dir = require_directory(args.model_dir, "model directory")
    model_path = require_file(model_dir / "model.pt", "model.pt")
    lookup_path = require_file(model_dir / "y_lookup.npy", "y_lookup.npy")
    dataset_dir = require_directory(model_dir / "dataset", "dataset directory")

    y_lookup = np.load(lookup_path, allow_pickle=False).astype(np.int64)
    if y_lookup.ndim != 2 or y_lookup.shape[1] != 4:
        raise ValueError(f"Expected y_lookup.npy shape (N, 4); found {y_lookup.shape}")
    conditions, class_ids = load_conditions(
        args.conditions_file,
        system_id,
        y_lookup,
        args.oversample,
    )

    train_features = np.load(
        require_file(dataset_dir / "X_num_train.npy", "X_num_train.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if train_features.ndim != 2:
        raise ValueError(f"Expected a 2D training array; found {train_features.shape}")
    num_features = int(train_features.shape[1])
    if num_features != 5:
        raise ValueError(f"Expected five generated features; found {num_features}")
    del train_features

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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
        "num_classes": int(len(y_lookup)),
        "is_y_cond": True,
        "rtdl_params": {
            "d_layers": parse_layers(args.d_layers),
            "dropout": 0.0,
        },
        "dim_t": int(args.dim_t),
    }
    transform_config = {
        "seed": int(args.seed),
        "normalization": args.normalization,
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }

    output_path = args.output_file.expanduser().resolve()
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

    output = np.empty((len(class_ids), num_features + 4), dtype=np.float32)
    unfilled = np.ones(len(class_ids), dtype=bool)

    print(f"Collection: {collection_name}")
    print(f"Conditions: {len(class_ids):,}")
    print(f"Classes: {len(y_lookup):,}")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory(prefix="genbib_tabddpm_", dir=output_path.parent) as work_dir:
        sample_job = {
            "parent_dir": work_dir,
            "real_data_path": str(dataset_dir),
            "batch_size": int(args.sample_batch_size),
            "model_type": "mlp",
            "model_params": model_params,
            "model_path": str(model_path),
            "num_timesteps": int(args.num_timesteps),
            "gaussian_loss_type": "mse",
            "scheduler": args.scheduler,
            "T_dict": transform_config,
            "num_numerical_features": num_features,
            "disbalance": None,
            "device": device,
            "change_val": False,
        }

        for round_id in range(args.max_rejection_rounds):
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
                seed=args.seed + round_id,
                y_to_sample=requested_ids,
            )

            generated_features = np.load(
                Path(work_dir) / "X_num_train.npy",
                allow_pickle=False,
            ).astype(np.float32)
            returned_ids = np.load(
                Path(work_dir) / "y_train.npy",
                allow_pickle=False,
            ).astype(np.int64).reshape(-1)
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
            if mask.shape != (len(remaining),):
                raise RuntimeError(
                    f"Material mask returned shape {mask.shape}; expected {(len(remaining),)}"
                )
            passing_slots = remaining[mask]
            output[passing_slots] = generated_hits[mask]
            unfilled[passing_slots] = False
            print(
                f"Round {round_id + 1}: accepted={len(passing_slots):,}, "
                f"remaining={int(unfilled.sum()):,}",
                flush=True,
            )

            del generated_features, returned_ids, generated_conditions, generated_hits, mask
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    dropped_rows = int(unfilled.sum())
    if dropped_rows:
        dropped_path = output_path.with_name(
            f"{output_path.stem}_dropped_conditions.npy"
        )
        np.save(dropped_path, conditions[unfilled])
        keep = ~unfilled
        output = output[keep]
        conditions = conditions[keep]
        print(
            f"WARNING: dropped {dropped_rows:,} conditions after "
            f"{args.max_rejection_rounds} rounds; saved them to {dropped_path}",
            flush=True,
        )

    if len(output) == 0:
        raise RuntimeError("No generated rows passed the material mask")

    expected_shape = (len(conditions), 9)
    if output.shape != expected_shape:
        raise RuntimeError(f"Output shape {output.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Output contains non-finite values")
    output_conditions = np.rint(output[:, 5:9]).astype(np.int64)
    if not np.array_equal(output_conditions, conditions[:, 1:5]):
        raise RuntimeError("Output conditions do not match the requested conditions")

    save_atomic(output_path, output)
    print(f"Saved {output.shape} to {output_path}")
    print(f"Dropped rows: {dropped_rows:,}")
    print("FINAL RESULT: PASS")


if __name__ == "__main__":
    main()
