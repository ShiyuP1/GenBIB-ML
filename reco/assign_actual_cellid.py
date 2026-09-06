#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


NINE_COL_COLLECTIONS = [
    ("tabddpm_VBC_samples", "VertexBarrelCollection_SimTrackerHit_conditional_reco9_0", 1),
    ("tabddpm_VEC_samples", "VertexEndcapCollection_SimTrackerHit_conditional_reco9_0", 2),
    ("tabddpm_ITBC_samples", "InnerTrackerBarrelCollection_SimTrackerHit_conditional_reco9_0", 3),
    ("tabddpm_ITEC_samples", "InnerTrackerEndcapCollection_SimTrackerHit_conditional_reco9_0", 4),
    ("tabddpm_OTBC_samples", "OuterTrackerBarrelCollection_SimTrackerHit_conditional_reco9_0", 5),
    ("tabddpm_OTEC_samples", "OuterTrackerEndcapCollection_SimTrackerHit_conditional_reco9_0", 6),
]

WITH_SYSTEM_COLLECTIONS = [
    ("VertexBarrelCollection_SimTrackerHit_conditional_reco9_0", 1),
    ("VertexEndcapCollection_SimTrackerHit_conditional_reco9_0", 2),
    ("InnerTrackerBarrelCollection_SimTrackerHit_conditional_reco9_0", 3),
    ("InnerTrackerEndcapCollection_SimTrackerHit_conditional_reco9_0", 4),
    ("OuterTrackerBarrelCollection_SimTrackerHit_conditional_reco9_0", 5),
    ("OuterTrackerEndcapCollection_SimTrackerHit_conditional_reco9_0", 6),
]

# Reference lookup is not required for normal CellID assignment.
REF_H5_PATH: Path | None = None

STATUS_NONE = 0
STATUS_UNIQUE = 1
STATUS_MULTI = 2
STATUS_RESCUED = 3

# Numerical tolerance for sensor boundaries and xyz round-trip noise.
FP_EPS_MM = 1e-6


@dataclass
class SensorGeometry:
    cellids: np.ndarray
    systems: np.ndarray
    centers: np.ndarray
    axes: np.ndarray
    half_lengths: np.ndarray
    shape_names: np.ndarray
    trd2_params: np.ndarray
    trees: dict[int, cKDTree]
    tree_indices: dict[int, np.ndarray]


@dataclass
class ReferenceLookup:
    tree: cKDTree
    cellids: np.ndarray


def decode_cellid(cellid: int) -> tuple[int, int, int, int, int]:
    raw_side = (int(cellid) >> 5) & 0x3
    side = raw_side - 4 if raw_side >= 2 else raw_side
    return (
        int(cellid) & 0x1F,
        side,
        (int(cellid) >> 7) & 0x3F,
        (int(cellid) >> 13) & 0x7FF,
        (int(cellid) >> 24) & 0xFF,
    )


def load_sensor_geometry(path: Path | str) -> SensorGeometry:
    data = np.load(path, allow_pickle=False)
    systems = data["systems"].astype(np.int16)
    centers = data["centers_mm"].astype(np.float64)

    trees: dict[int, cKDTree] = {}
    tree_indices: dict[int, np.ndarray] = {}
    for system in np.unique(systems):
        system_id = int(system)
        indices = np.flatnonzero(systems == system_id)
        trees[system_id] = cKDTree(centers[indices])
        tree_indices[system_id] = indices

    return SensorGeometry(
        cellids=data["cellids"].astype(np.int64),
        systems=systems,
        centers=centers,
        axes=data["axes"].astype(np.float64),
        half_lengths=data["half_lengths_mm"].astype(np.float64),
        shape_names=data["shape_names"].astype(str),
        trd2_params=data["trd2_params_mm"].astype(np.float64),
        trees=trees,
        tree_indices=tree_indices,
    )


def context_center_cm(context):
    point = context.localToWorld(array("d", [0.0, 0.0, 0.0]))
    return (float(point.X()), float(point.Y()), float(point.Z()))


def local_to_master(matrix, xyz_cm):
    src = array("d", xyz_cm)
    dst = array("d", [0.0, 0.0, 0.0])
    matrix.LocalToMaster(src, dst)
    return np.array(dst, dtype=np.float64)


def shape_half_lengths_cm(shape):
    if shape is None:
        return np.nan, np.nan, np.nan, False
    if hasattr(shape, "ComputeBBox"):
        shape.ComputeBBox()

    vals = []
    ok = True
    for name in ("GetDX", "GetDY", "GetDZ"):
        if hasattr(shape, name):
            vals.append(float(getattr(shape, name)()))
        else:
            vals.append(np.nan)
            ok = False
    return vals[0], vals[1], vals[2], ok


def trd2_params_mm(shape):
    if shape is None or not hasattr(shape, "GetDx1"):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    return (
        float(shape.GetDx1()) * 10.0,
        float(shape.GetDx2()) * 10.0,
        float(shape.GetDy1()) * 10.0,
        float(shape.GetDy2()) * 10.0,
        float(shape.GetDz()) * 10.0,
    )


def build_sensor_geometry(path: Path) -> None:
    import dd4hep
    import ROOT

    if "MUCOLL_GEO" not in os.environ:
        raise RuntimeError("MUCOLL_GEO is not set")

    path.parent.mkdir(parents=True, exist_ok=True)
    detector = dd4hep.Detector.getInstance()
    detector.fromXML(os.environ["MUCOLL_GEO"])
    geom = ROOT.gGeoManager
    nav = geom.GetCurrentNavigator()
    vm = detector.volumeManager()

    by_cellid: dict[int, str] = {}
    for system in range(1, 7):
        subdetector = vm.subdetector(system)
        for pair in subdetector.ptr().volumes:
            cellid = int(pair.first)
            if decode_cellid(cellid)[0] != system:
                continue
            if cellid in by_cellid:
                continue
            # Use the sensitive placement center only to recover its TGeo path.
            x, y, z = context_center_cm(pair.second)
            if not nav.FindNode(x, y, z):
                continue
            found_path = geom.GetPath()
            # Keep only paths deep enough to identify an individual sensor.
            if found_path != "/world_volume_1" and found_path.count("/") >= 3:
                by_cellid[cellid] = found_path

    cellids = []
    systems = []
    centers = []
    axes = []
    half_lengths = []
    shape_names = []
    trd2_params = []
    bad_half_lengths = 0

    for cellid, found_path in sorted(by_cellid.items()):
        if not geom.cd(found_path):
            continue

        matrix = geom.GetCurrentMatrix()
        node = geom.GetCurrentNode()
        volume = node.GetVolume() if node else None
        shape = volume.GetShape() if volume else None

        # Store the sensor frame in global coordinates for fast containment tests.
        center_cm = local_to_master(matrix, (0.0, 0.0, 0.0))
        axis = np.vstack(
            [
                local_to_master(matrix, (1.0, 0.0, 0.0)) - center_cm,
                local_to_master(matrix, (0.0, 1.0, 0.0)) - center_cm,
                local_to_master(matrix, (0.0, 0.0, 1.0)) - center_cm,
            ]
        )
        norms = np.linalg.norm(axis, axis=1)
        axis = axis / np.where(norms[:, None] == 0.0, 1.0, norms[:, None])

        system = decode_cellid(cellid)[0]
        dx, dy, dz, half_lengths_ok = shape_half_lengths_cm(shape)
        if not half_lengths_ok:
            bad_half_lengths += 1
        cellids.append(cellid)
        systems.append(system)
        centers.append(center_cm * 10.0)
        axes.append(axis)
        half_lengths.append(np.array([dx, dy, dz], dtype=np.float64) * 10.0)
        shape_names.append(shape.ClassName() if shape else "")
        trd2_params.append(np.array(trd2_params_mm(shape), dtype=np.float64))

    if not cellids:
        raise RuntimeError("No sensitive sensor geometry was found")

    np.savez_compressed(
        path,
        cellids=np.array(cellids, dtype=np.int64),
        systems=np.array(systems, dtype=np.int16),
        centers_mm=np.vstack(centers).astype(np.float64),
        axes=np.stack(axes).astype(np.float64),
        half_lengths_mm=np.vstack(half_lengths).astype(np.float64),
        shape_names=np.array(shape_names, dtype=str),
        trd2_params_mm=np.vstack(trd2_params).astype(np.float64),
    )
    print(f"built sensor geometry: {path} ({len(cellids)} sensors)")
    if bad_half_lengths:
        print(f"warning: {bad_half_lengths} sensors have incomplete shape half lengths")


def load_or_build_sensor_geometry(path: Path) -> SensorGeometry:
    if not path.exists():
        build_sensor_geometry(path)
    return load_sensor_geometry(path)


XYZ_KEY_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])


def xyz_keys(xyz):
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    return xyz.view(XYZ_KEY_DTYPE).reshape(-1)


def load_h5_reference_lookup(path: Path, collection: str) -> ReferenceLookup:
    import pandas as pd

    xyz_chunks = []
    cellid_chunks = []
    where = f'collection == "{collection}"'
    for chunk in pd.read_hdf(path, key="df", where=where, columns=["x", "y", "z", "cellid0"], chunksize=500_000):
        xyz_chunks.append(chunk[["x", "y", "z"]].to_numpy(dtype=np.float64))
        cellid_chunks.append(chunk["cellid0"].to_numpy(dtype=np.int64))
    if not xyz_chunks:
        return build_reference_lookup(np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64))
    return build_reference_lookup(np.concatenate(xyz_chunks), np.concatenate(cellid_chunks))


def build_reference_lookup(xyz: np.ndarray, cellids: np.ndarray) -> ReferenceLookup:
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    cellids = np.asarray(cellids, dtype=np.int64)

    keys = xyz_keys(xyz)
    order = np.argsort(keys, order=("x", "y", "z"))
    keys = keys[order]
    xyz = xyz[order]
    cellids = cellids[order]

    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    stops = np.r_[starts[1:], len(keys)]
    keep = np.ones(len(starts), dtype=bool)
    for i in np.flatnonzero((stops - starts) > 1):
        start = starts[i]
        stop = stops[i]
        if np.any(cellids[start:stop] != cellids[start]):
            keep[i] = False

    kept = starts[keep]
    return ReferenceLookup(tree=cKDTree(xyz[kept]), cellids=cellids[kept])


def lookup_reference_cellids(ref: ReferenceLookup, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Nearest neighbor within FP_EPS_MM -- r/phi round trips aren't bit exact.
    dist, index = ref.tree.query(xyz, k=1, workers=-1)
    found = dist <= FP_EPS_MM
    cellids = np.full(len(xyz), -1, dtype=np.int64)
    cellids[found] = ref.cellids[index[found]]
    return cellids, found


# def pack_cellid(system: int, side: int, layer: int, module: int, sensor: int) -> int:
#     return (
#         (system & 0x1F)
#         | ((side & 0x3) << 5)
#         | ((layer & 0x3F) << 7)
#         | ((module & 0x7FF) << 13)
#         | ((sensor & 0xFF) << 24)
#     )
def decode_cellids(cellids: np.ndarray) -> np.ndarray:
    cellids = cellids.astype(np.int64, copy=False)
    raw_side = (cellids >> 5) & 0x3
    side = np.where(raw_side >= 2, raw_side - 4, raw_side)
    return np.column_stack(
        (
            cellids & 0x1F,
            side,
            (cellids >> 7) & 0x3F,
            (cellids >> 13) & 0x7FF,
            (cellids >> 24) & 0xFF,
        )
    ).astype(np.int64)


def _candidate_indices(xyz, system, geom, k):
    base = geom.tree_indices[system]
    kk = min(k, len(base))
    _dist, local_indices = geom.trees[system].query(xyz, k=kk, workers=-1)
    local_indices = np.atleast_2d(local_indices)
    if local_indices.shape[0] != len(xyz):
        local_indices = local_indices.T
    return base[local_indices]


def _local_coordinates(xyz, candidates, geom):
    diff = xyz[:, None, :] - geom.centers[candidates]
    return np.einsum("nkij,nkj->nki", geom.axes[candidates], diff)


def _effective_half_lengths(local, candidates, geom):
    half = geom.half_lengths[candidates]
    hx = half[:, :, 0].copy()
    hy = half[:, :, 1].copy()
    hz = half[:, :, 2].copy()

    trd2 = geom.shape_names[candidates] == "TGeoTrd2"
    if np.any(trd2):
        rows, cols = np.where(trd2)
        params = geom.trd2_params[candidates[rows, cols]]
        dz = params[:, 4]
        frac = np.clip((local[rows, cols, 2] / np.where(dz == 0.0, 1.0, dz) + 1.0) * 0.5, 0.0, 1.0)
        hx[rows, cols] = params[:, 0] * (1.0 - frac) + params[:, 1] * frac
        hy[rows, cols] = params[:, 2] * (1.0 - frac) + params[:, 3] * frac
        hz[rows, cols] = dz

    return hx, hy, hz


def find_contained_sensors(
    xyz: np.ndarray,
    system: int,
    geom: SensorGeometry,
    k: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = _candidate_indices(xyz, system, geom, k)
    local = _local_coordinates(xyz, candidates, geom)
    hx, hy, hz = _effective_half_lengths(local, candidates, geom)
    inside = (
        (np.abs(local[:, :, 0]) <= hx + FP_EPS_MM)
        & (np.abs(local[:, :, 1]) <= hy + FP_EPS_MM)
        & (np.abs(local[:, :, 2]) <= hz + FP_EPS_MM)
    )
    return candidates, inside


def _classify(
    xyz: np.ndarray,
    system: int,
    geom: SensorGeometry,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates, inside = find_contained_sensors(xyz, system, geom, k)
    n_inside = inside.sum(axis=1)

    status = np.full(len(xyz), STATUS_NONE, dtype=np.int8)
    assigned = np.full(len(xyz), -1, dtype=np.int64)

    unique_rows = np.flatnonzero(n_inside == 1)
    if len(unique_rows):
        first = np.argmax(inside[unique_rows], axis=1)
        assigned[unique_rows] = geom.cellids[candidates[unique_rows, first]]
        status[unique_rows] = STATUS_UNIQUE

    status[n_inside > 1] = STATUS_MULTI

    multi_rows = np.flatnonzero(n_inside > 1)
    if len(multi_rows):
        cands = candidates[multi_rows]
        local = _local_coordinates(xyz[multi_rows], cands, geom)
        hx, hy, hz = _effective_half_lengths(local, cands, geom)
        nx = np.abs(local[:, :, 0]) / (hx + 1e-12)
        ny = np.abs(local[:, :, 1]) / (hy + 1e-12)
        nz = np.abs(local[:, :, 2]) / (hz + 1e-12)
        score = np.maximum.reduce([nx, ny, nz])
        best = np.argmin(score, axis=1)
        assigned[multi_rows] = geom.cellids[cands[np.arange(len(multi_rows)), best]]
        status[multi_rows] = STATUS_MULTI

    return assigned, status, candidates, inside


def rphi9_to_columns(rows: np.ndarray, system_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loge = rows[:, 0].astype(np.float64)
    time = rows[:, 1].astype(np.float64)
    radius = rows[:, 2].astype(np.float64)
    phi = rows[:, 3].astype(np.float64)
    z = rows[:, 4].astype(np.float64)
    edep = np.exp(loge)
    xyz = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    systems = np.full(len(rows), system_id, dtype=np.int16)
    head = np.column_stack((edep, xyz, time))
    return head, xyz, systems


def rphi10_to_columns(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loge = rows[:, 0].astype(np.float64)
    radius = rows[:, 1].astype(np.float64)
    phi = rows[:, 2].astype(np.float64)
    z = rows[:, 3].astype(np.float64)
    time = rows[:, 4].astype(np.float64)
    xyz = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    systems = rows[:, 5].astype(np.int16)
    head = np.column_stack((np.exp(loge), xyz, time))
    return head, xyz, systems


def global10_to_columns(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head = rows[:, :5].astype(np.float64)
    xyz = head[:, 1:4]
    systems = rows[:, 5].astype(np.int16)
    return head, xyz, systems


def select_rows(rows: np.ndarray, fraction: float, mode: str, seed: int) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--write-fraction must be in (0, 1]")
    n_select = max(1, int(len(rows) * fraction))
    if n_select >= len(rows):
        return rows
    if mode == "first":
        return rows[:n_select]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n_select, replace=False)
    idx.sort()
    return rows[idx]


def collection_specs(input_format: str) -> list[tuple[str, str, int]]:
    if input_format in {"9col", "rphi10"}:
        return NINE_COL_COLLECTIONS
    return [(name, name, system_id) for name, system_id in WITH_SYSTEM_COLLECTIONS]


def convert_rows(
    rows: np.ndarray,
    input_format: str,
    system_id: int,
    geom: SensorGeometry,
    k: int,
    ref_lookup: ReferenceLookup | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if input_format in {"9col", "rphi9"}:
        head, xyz, systems = rphi9_to_columns(rows, system_id)
    elif input_format == "rphi10":
        head, xyz, systems = rphi10_to_columns(rows)
    else:
        head, xyz, systems = global10_to_columns(rows)

    assigned = np.full(len(rows), -1, dtype=np.int64)
    status = np.full(len(rows), STATUS_NONE, dtype=np.int8)

    for system in np.unique(systems):
        part = np.flatnonzero(systems == system)
        assigned_part, status_part, _candidates, _inside = _classify(xyz[part], int(system), geom, k)
        assigned[part] = assigned_part
        status[part] = status_part

    lookup_rescued = 0
    if ref_lookup is not None:
        ref_cellids, found = lookup_reference_cellids(ref_lookup, xyz)

        unique_rows = np.flatnonzero(status == STATUS_UNIQUE)
        mismatch = unique_rows[found[unique_rows] & (ref_cellids[unique_rows] != assigned[unique_rows])]
        assigned[mismatch] = ref_cellids[mismatch]
        status[mismatch] = STATUS_RESCUED

        problem_rows = np.flatnonzero((status == STATUS_NONE) | (status == STATUS_MULTI))
        rescued_rows = np.empty(0, dtype=np.int64)
        if len(problem_rows):
            rescued_rows = problem_rows[found[problem_rows]]
            assigned[rescued_rows] = ref_cellids[rescued_rows]
            status[rescued_rows] = STATUS_RESCUED

        lookup_rescued = len(mismatch) + len(rescued_rows)

    valid = (status == STATUS_UNIQUE) | (status == STATUS_RESCUED) | (status == STATUS_MULTI)
    labels = decode_cellids(assigned[valid]).astype(np.float64)
    output = np.column_stack((head[valid], labels)) if len(labels) else np.empty((0, 10), dtype=np.float64)
    counts = {
        "written_rows": int(np.count_nonzero(valid)),
        "invalid_rows": int(np.count_nonzero(status == STATUS_NONE)),
        "multi_assigned_rows": int(np.count_nonzero(status == STATUS_MULTI)),
        "rescued_rows": int(np.count_nonzero(status == STATUS_RESCUED)),
        "lookup_rescued_rows": int(lookup_rescued),
    }
    return output, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-format", choices=["9col", "10col", "rphi10"], default="9col")
    parser.add_argument("--write-fraction", type=float, default=1.0)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--geomap-path", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geom = load_or_build_sensor_geometry(args.geomap_path)
    print(f"loaded sensor geometry: {args.geomap_path}")

    summary_rows = []
    for i, (input_stem, output_stem, system_id) in enumerate(collection_specs(args.input_format)):
        input_path = input_dir / f"{input_stem}.npy"
        output_path = output_dir / f"{output_stem}.npy"

        rows = np.load(input_path, mmap_mode="r")
        selected = select_rows(rows, args.write_fraction, args.sample_mode, args.random_seed + i)
        ref_lookup = None
        if REF_H5_PATH is not None:
            h5_collection = output_stem.removesuffix("_SimTrackerHit_conditional_reco9_0")
            ref_lookup = load_h5_reference_lookup(REF_H5_PATH, h5_collection)
        output, counts = convert_rows(
            selected,
            args.input_format,
            system_id,
            geom,
            args.k,
            ref_lookup,
        )
        np.save(output_path, output)

        summary = {
            "collection": output_stem,
            "input_file": str(input_path),
            "source_rows": len(rows),
            "selected_rows": len(selected),
            "written_rows": counts["written_rows"],
            "unresolved_rows": counts["invalid_rows"],
            "multi_assigned_rows": counts["multi_assigned_rows"],
            "lookup_rescued_rows": counts["lookup_rescued_rows"],
        }
        summary_rows.append(summary)
        print(
            f"{output_stem}: source={len(rows)} selected={len(selected)} "
            f"written={counts['written_rows']} multi_assigned={counts['multi_assigned_rows']} "
            f"unresolved={counts['invalid_rows']}"
        )

    summary_path = output_dir / "assign_actual_cellid_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
