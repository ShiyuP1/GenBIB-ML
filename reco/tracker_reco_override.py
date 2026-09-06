#!/usr/bin/env python3
import sys


def pop_stage():
    if "--stage" not in sys.argv:
        raise RuntimeError("Missing required argument: --stage digi|reco")
    idx = sys.argv.index("--stage")
    try:
        stage = sys.argv[idx + 1]
    except IndexError as exc:
        raise RuntimeError("Missing value for --stage") from exc
    del sys.argv[idx:idx + 2]
    if stage not in ("digi", "reco"):
        raise RuntimeError("Stage must be digi or reco")
    return stage


def tracker_digi_algs(args):
    from TrackerDigi.tracking_vertex import VXDBarrel_cfg, VXDEndcap_cfg
    from TrackerDigi.tracking_inner import ITBarrel_cfg, ITEndcap_cfg
    from TrackerDigi.tracking_outer import OTBarrel_cfg, OTEndcap_cfg

    algs = [
        VXDBarrel_cfg(args),
        VXDEndcap_cfg(args),
        ITBarrel_cfg(args),
        ITEndcap_cfg(args),
        OTBarrel_cfg(args),
        OTEndcap_cfg(args),
    ]
    for alg in algs:
        alg.IsStrip = False
        alg.ForceHitsOntoSurface = True
    return algs


def make_digi_alg_list(args):
    from Common.event_counter import event_counter_cfg

    return [event_counter_cfg()] + tracker_digi_algs(args)


def make_reco_alg_list(args):
    from Common.event_counter import event_counter_cfg
    from Tracking.mergers import mergehits_cfg, mergehitsrelations_cfg

    algs = [
        event_counter_cfg(),
        mergehits_cfg(args),
        mergehitsrelations_cfg(args),
    ]

    # v3 tracker-only CKF reconstruction.
    from Tracking.CKF_tracking import CKFTracker_cfg, deduper_cfg
    deduper = deduper_cfg()
    deduper.OutputTrackCollectionName = ["SiTracks"]
    algs += [CKFTracker_cfg(args), deduper]

    return algs


stage = pop_stage()

if stage == "digi":
    from digi_args import get_digi_args
    from Common.steering import build_application

    args = get_digi_args()
    build_application(
        args,
        make_digi_alg_list(args),
        input_files=["bib_flow.edm4hep.root"],
        output_file="digi_output.edm4hep.root",
        histo_file="digi_histograms.root",
    )
else:
    from reco_args import get_reco_args
    from Common.steering import build_application

    args = get_reco_args()
    build_application(
        args,
        make_reco_alg_list(args),
        input_files=["digi_output.edm4hep.root"],
        output_file="reco_output.edm4hep.root",
        histo_file="reco_histograms.root",
    )
