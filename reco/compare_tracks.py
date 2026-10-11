#!/usr/bin/env python3
"""Plot track distributions from one or more EDM4hep reconstruction files."""

import json
import os
from collections import defaultdict
from math import atan, log, pi, tan
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from podio import root_io


BFIELD_T = 5.0
CELL_ID_ENCODING = "system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8"
PROJECT_ROOT = Path("/oscar/home/speng44/data/mucoll/speng44/bib_gen_mother_muon")
TRACK_COLLECTION = "SelectedTracks"
OUTPUT_DIR = PROJECT_ROOT / "new_diffusion_norm1/reco/plot/data_vs_norm1"
TRACK_SETS = [
    ("Data", PROJECT_ROOT / "data/norm1/reco/reco_output.edm4hep.root"),
    (
        "Norm1",
        PROJECT_ROOT
        / "new_diffusion_norm1/reco/reco/reco_output.edm4hep.root",
    ),
]
SYSTEM_KEYS = {
    1: "vxd_barrel_nhits",
    2: "vxd_endcap_nhits",
    3: "it_barrel_nhits",
    4: "it_endcap_nhits",
    5: "ot_barrel_nhits",
    6: "ot_endcap_nhits",
}
SYSTEM_NAMES = {
    1: "VXD Barrel",
    2: "VXD Endcap",
    3: "IT Barrel",
    4: "IT Endcap",
    5: "OT Barrel",
    6: "OT Endcap",
}
PLOT_SPECS = [
    ("vxd_barrel_nhits", np.arange(0, 9) - 0.5, "VXD barrel hits", "VXD Barrel Hits per Track", "linear", "linear"),
    ("vxd_endcap_nhits", np.arange(0, 9) - 0.5, "VXD endcap hits", "VXD Endcap Hits per Track", "linear", "linear"),
    ("it_barrel_nhits", np.arange(0, 6) - 0.5, "IT barrel hits", "IT Barrel Hits per Track", "linear", "linear"),
    ("it_endcap_nhits", np.arange(0, 9) - 0.5, "IT endcap hits", "IT Endcap Hits per Track", "linear", "linear"),
    ("ot_barrel_nhits", np.arange(0, 5) - 0.5, "OT barrel hits", "OT Barrel Hits per Track", "linear", "linear"),
    ("ot_endcap_nhits", np.arange(0, 6) - 0.5, "OT endcap hits", "OT Endcap Hits per Track", "linear", "linear"),
    ("total_nhits", np.arange(0, 25) - 0.5, "Total hits", "Total Hits per Track", "linear", "linear"),
    ("eta", np.linspace(-3, 3, 61), "Track $\\eta$", "Track $\\eta$", "linear", "linear"),
    ("phi", np.linspace(-pi, pi, 65), "Track $\\phi$ [rad]", "Track $\\phi$", "linear", "linear"),
    ("chi2_reduced", np.linspace(0, 5, 6), "$\\chi^2/Ndf$", "Track $\\chi^2/Ndf$", "linear", "linear"),
    ("d0", np.linspace(-5, 5, 51), "$d_0$ [mm]", "Track $d_0$", "linear", "linear"),
    ("z0", np.linspace(-25, 25, 101), "$z_0$ [mm]", "Track $z_0$", "linear", "linear"),
    ("pt", np.logspace(-1, 4, 61), "Track $p_T$ [GeV]", "Track $p_T$", "log", "log"),
]


def parse_encoding(encoding):
    fields = {}
    for token in encoding.split(","):
        name, offset, width = token.split(":")
        fields[name] = int(offset), abs(int(width))
    return fields


ENCODING_FIELDS = parse_encoding(CELL_ID_ENCODING)


def decode_field(cell_id, name):
    offset, width = ENCODING_FIELDS[name]
    return (int(cell_id) >> offset) & ((1 << width) - 1)


def eta_from_tan_lambda(tan_lambda):
    theta = pi / 2.0 - atan(tan_lambda)
    return -log(tan(max(theta / 2.0, 1e-10)))


def physics_state(track):
    states = list(track.getTrackStates())
    by_location = {int(state.location): state for state in states}
    ordered = [by_location[location] for location in (1, 2, 3) if location in by_location]
    ordered.extend(state for state in states if state not in ordered)
    return next(
        (state for state in ordered if state.omega != 0.0 and np.isfinite(state.omega)),
        states[0] if states else None,
    )


def collect_tracks(path, label, track_collection):
    keys = (*SYSTEM_KEYS.values(), "total_nhits", "eta", "phi", "chi2_reduced", "d0", "z0", "pt")
    values = {key: [] for key in keys}
    layer_hits = {system_id: defaultdict(int) for system_id in SYSTEM_KEYS}

    for event_index, frame in enumerate(root_io.Reader(str(path)).get("events")):
        available = list(frame.getAvailableCollections())
        if track_collection not in available:
            raise RuntimeError(
                f"{path}: event {event_index} has no {track_collection}; available={available}"
            )

        for track in frame.get(track_collection):
            ndf = track.getNdf()
            state = physics_state(track)
            if ndf <= 0 or state is None:
                continue

            hit_counts = {key: 0 for key in SYSTEM_KEYS.values()}
            for hit in track.getTrackerHits():
                system_id = decode_field(hit.getCellID(), "system")
                if system_id in SYSTEM_KEYS:
                    hit_counts[SYSTEM_KEYS[system_id]] += 1
                    layer = decode_field(hit.getCellID(), "layer")
                    layer_hits[system_id][layer] += 1

            for key, count in hit_counts.items():
                values[key].append(count)
            values["total_nhits"].append(sum(hit_counts.values()))
            values["eta"].append(eta_from_tan_lambda(state.tanLambda))
            values["phi"].append(state.phi)
            values["chi2_reduced"].append(track.getChi2() / float(ndf))
            values["d0"].append(state.D0)
            values["z0"].append(state.Z0)
            values["pt"].append(0.3 * BFIELD_T / (abs(state.omega) * 1e3))

    arrays = {key: np.asarray(items) for key, items in values.items()}
    print(f"{label}: {len(arrays['total_nhits']):,} {track_collection} tracks")
    return arrays, layer_hits


def plot_histogram(
    axis,
    ratio_axis,
    track_sets,
    key,
    bins,
    xlabel,
    title,
    xscale,
    yscale,
):
    histograms = []
    for label, arrays, _ in track_sets:
        values = arrays[key]
        values = values[np.isfinite(values)]
        if len(values):
            counts = np.histogram(values, bins=bins)[0] / float(len(values))
            histograms.append((label, counts, len(values)))
            axis.stairs(counts, bins, linewidth=1.5, label=f"{label} (N={len(values):,})")

    axis.set(ylabel="Fraction of tracks", title=title)
    axis.set_xscale(xscale)
    axis.set_yscale(yscale)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    axis.legend()

    if ratio_axis is None:
        axis.set_xlabel(xlabel)
        return

    reference = histograms[0][1]
    for label, counts, _ in histograms:
        ratio = np.divide(
            counts,
            reference,
            out=np.full_like(counts, np.nan),
            where=reference > 0,
        )
        ratio_axis.stairs(ratio, bins, linewidth=1.5, label=label)
    ratio_axis.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ratio_axis.set(xlabel=xlabel, ylabel=f"Ratio to\n{histograms[0][0]}")
    ratio_axis.set_xscale(xscale)
    ratio_axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)


def save_histograms(track_sets, output_dir):
    with PdfPages(output_dir / "all_plots.pdf") as combined:
        for key, bins, xlabel, title, xscale, yscale in PLOT_SPECS:
            if len(track_sets) == 1:
                figure, axis = plt.subplots(figsize=(6, 4))
                ratio_axis = None
            else:
                figure, (axis, ratio_axis) = plt.subplots(
                    2,
                    1,
                    figsize=(6, 5),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                figure.subplots_adjust(hspace=0.05)
            plot_histogram(
                axis,
                ratio_axis,
                track_sets,
                key,
                bins,
                xlabel,
                title,
                xscale,
                yscale,
            )
            figure.tight_layout()
            figure.savefig(output_dir / f"{key}.pdf")
            figure.savefig(output_dir / f"{key}.png", dpi=180)
            combined.savefig(figure)
            plt.close(figure)


def save_layer_plot(track_sets, output_dir):
    figure, axes = plt.subplots(3, 2, figsize=(11, 12))
    width = 0.7 / len(track_sets)

    for axis, (system_id, system_name) in zip(axes.flat, SYSTEM_NAMES.items()):
        layers = sorted(
            set().union(*(layer_hits[system_id] for _, _, layer_hits in track_sets))
        )
        x = np.asarray(layers)

        for index, (label, arrays, layer_hits) in enumerate(track_sets):
            n_tracks = len(arrays["total_nhits"])
            averages = np.asarray(
                [layer_hits[system_id].get(layer, 0) / n_tracks for layer in layers]
                if n_tracks else np.zeros(len(layers))
            )
            offset = (index - (len(track_sets) - 1) / 2.0) * width
            axis.bar(x + offset, averages, width=width, label=label, alpha=0.8)

        axis.set(xlabel="Layer", ylabel="Average hits / track", title=system_name)
        axis.set_xticks(x)
        axis.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "average_hits_per_layer.pdf")
    figure.savefig(output_dir / "average_hits_per_layer.png", dpi=180)
    plt.close(figure)


def main():
    missing = [path for _, path in TRACK_SETS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reconstruction inputs: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    track_sets = [
        (label, *collect_tracks(path, label, TRACK_COLLECTION))
        for label, path in TRACK_SETS
    ]

    save_histograms(track_sets, OUTPUT_DIR)
    save_layer_plot(track_sets, OUTPUT_DIR)

    summary = {
        "track_collection": TRACK_COLLECTION,
        "inputs": {label: str(path.resolve()) for label, path in TRACK_SETS},
        "track_counts": {
            label: len(arrays["total_nhits"])
            for label, arrays, _ in track_sets
        },
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
