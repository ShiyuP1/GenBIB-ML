#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import edm4hep
import numpy as np
import podio
import podio.root_io


TRACKER_CELL_ID_ENCODING = "system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8"
CALO_CELL_ID_ENCODING = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16"

TRACKER_COLLECTIONS = [
    ("VertexBarrelCollection_SimTrackerHit_conditional_reco9_0", "VertexBarrelCollection"),
    ("VertexEndcapCollection_SimTrackerHit_conditional_reco9_0", "VertexEndcapCollection"),
    ("InnerTrackerBarrelCollection_SimTrackerHit_conditional_reco9_0", "InnerTrackerBarrelCollection"),
    ("InnerTrackerEndcapCollection_SimTrackerHit_conditional_reco9_0", "InnerTrackerEndcapCollection"),
    ("OuterTrackerBarrelCollection_SimTrackerHit_conditional_reco9_0", "OuterTrackerBarrelCollection"),
    ("OuterTrackerEndcapCollection_SimTrackerHit_conditional_reco9_0", "OuterTrackerEndcapCollection"),
]

EMPTY_CALO_COLLECTIONS = [
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
    "YokeBarrelCollection",
    "YokeEndcapCollection",
]


def pack_cell_id(system: int, side: int, layer: int, module: int, sensor: int) -> int:
    side_bits = int(side) & 0x3
    return (
        (int(system) & 0x1F)
        | (side_bits << 5)
        | ((int(layer) & 0x3F) << 7)
        | ((int(module) & 0x7FF) << 13)
        | ((int(sensor) & 0xFF) << 24)
    )


def fraction_slice(array: np.ndarray, fraction: float, mode: str, seed: int) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--write-fraction must be in (0, 1]")
    n = max(1, int(len(array) * fraction))
    if n >= len(array):
        return array
    if mode == "first":
        return array[:n]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(array), size=n, replace=False)
    idx.sort()
    return array[idx]


def write_metadata(writer: podio.root_io.Writer) -> None:
    metadata = podio.Frame()
    for _, coll_name in TRACKER_COLLECTIONS:
        metadata.put_parameter(f"{coll_name}__CellIDEncoding", TRACKER_CELL_ID_ENCODING)
    for coll_name in EMPTY_CALO_COLLECTIONS:
        metadata.put_parameter(f"{coll_name}__CellIDEncoding", CALO_CELL_ID_ENCODING)
    writer.write_frame(metadata, "metadata")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-events", type=int, default=1)
    parser.add_argument("--write-fraction", type=float, default=1.0)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--random-seed", type=int, default=12345)
    args = parser.parse_args()

    samples_path = Path(args.samples_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = []
    for i, (file_stem, coll_name) in enumerate(TRACKER_COLLECTIONS):
        path = samples_path / f"{file_stem}.npy"
        rows = np.load(path, mmap_mode="r")
        rows = fraction_slice(rows, args.write_fraction, args.sample_mode, args.random_seed + i)
        loaded.append((rows, coll_name))
        print(f"{coll_name}: selected={len(rows)}")

    writer = podio.root_io.Writer(str(output_path))
    write_metadata(writer)

    for evt_num in range(args.num_events):
        frame = podio.Frame()

        header = edm4hep.EventHeaderCollection()
        h = header.create()
        h.setEventNumber(evt_num)
        h.setRunNumber(0)
        frame.put(header, "EventHeader")
        frame.put(edm4hep.MCParticleCollection(), "MCParticles")

        for coll_name in EMPTY_CALO_COLLECTIONS:
            frame.put(edm4hep.SimCalorimeterHitCollection(), coll_name)

        for rows, coll_name in loaded:
            start = evt_num * len(rows) // args.num_events
            stop = (evt_num + 1) * len(rows) // args.num_events
            coll = edm4hep.SimTrackerHitCollection()
            for row in rows[start:stop]:
                edep, x, y, z, time, system, side, layer, module, sensor = row
                hit = coll.create()
                hit.setEDep(float(edep))
                hit.setTime(float(time))
                hit.setPosition(edm4hep.Vector3d(float(x), float(y), float(z)))
                hit.setCellID(pack_cell_id(int(system), int(side), int(layer), int(module), int(sensor)))
            frame.put(coll, coll_name)
            print(f"event {evt_num} {coll_name}: wrote={stop - start}")

        writer.write_frame(frame, "events")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
