"""CLI: `python -m bench list` / `python -m bench run [names...]`."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from .harness import detect_device
from .registry import all_benches, get
from . import benches as _benches  # noqa: F401  side-effect registration


def _env_snapshot(device: str) -> dict:
    snap = {
        "device": device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        snap["torch"] = torch.__version__
        if torch.cuda.is_available():
            snap["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import ultralytics
        snap["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    try:
        import cv2
        snap["opencv"] = cv2.__version__
    except Exception:
        pass
    return snap


def _cmd_list(_):
    bs = all_benches()
    if not bs:
        print("no benches registered")
        return 0
    width = max(len(n) for n in bs)
    for name, spec in sorted(bs.items()):
        print(f"  {name:<{width}}  {spec.description}")
    return 0


def _cmd_run(args):
    names = args.names if args.names else sorted(all_benches().keys())
    missing = [n for n in names if get(n) is None]
    if missing:
        print(f"unknown bench(es): {', '.join(missing)}", file=sys.stderr)
        return 2
    device = detect_device(args.device)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    env = _env_snapshot(device)
    print(f"env: {env}")
    kwargs = dict(
        device=device,
        iters=args.iters,
        warmup=args.warmup,
        frames=args.frames,
        imgsz=args.imgsz,
        video=args.video,
        model=args.model,
    )
    exit_code = 0
    for n in names:
        spec = get(n)
        try:
            report = spec.fn(**kwargs)
            print(report.pretty())
            if out_dir:
                path = out_dir / f"{ts}_{n}{('_' + args.tag) if args.tag else ''}.json"
                payload = {"env": env, "tag": args.tag, "report": report.to_dict()}
                path.write_text(json.dumps(payload, indent=2))
                print(f"  -> {path}")
        except Exception as e:
            exit_code = 1
            print(f"\n!! bench {n} failed: {e}", file=sys.stderr)
            traceback.print_exc()
    return exit_code


def _cmd_compare(args):
    """Compare two JSON result files written by `run --tag before` / `--tag after`."""
    a = json.loads(Path(args.before).read_text())
    b = json.loads(Path(args.after).read_text())
    ra = a["report"]
    rb = b["report"]
    if ra["name"] != rb["name"]:
        print(f"warning: bench names differ ({ra['name']} vs {rb['name']})")
    by_label_a = {s["label"]: s for s in ra["samples"]}
    by_label_b = {s["label"]: s for s in rb["samples"]}
    labels = [s["label"] for s in ra["samples"]] + [l for l in (s["label"] for s in rb["samples"]) if l not in by_label_a]
    print(f"\n=== compare: {ra['name']} ===")
    print(f"before tag: {a.get('tag', '-')}  after tag: {b.get('tag', '-')}")
    print(f"  {'label':<28} {'before ms':>10} {'after ms':>10}   {'delta':>8}")
    print("  " + "-" * 62)
    for lbl in labels:
        sa = by_label_a.get(lbl)
        sb = by_label_b.get(lbl)
        before_ms = sa["mean_s"] * 1000 if sa else None
        after_ms = sb["mean_s"] * 1000 if sb else None
        before_str = f"{before_ms:.2f}" if before_ms is not None else "-"
        after_str = f"{after_ms:.2f}" if after_ms is not None else "-"
        if sa and sb and sb["mean_s"] > 0:
            speedup_str = f"{sa['mean_s'] / sb['mean_s']:>7.2f}x"
        else:
            speedup_str = "   (missing)"
        print(f"  {lbl:<28} {before_str:>10} {after_str:>10}   {speedup_str}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench", description="FunGen perf bench harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="List registered benches")
    pl.set_defaults(handler=_cmd_list)

    pr = sub.add_parser("run", help="Run one or more benches (default: all)")
    pr.add_argument("names", nargs="*")
    pr.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    pr.add_argument("--iters", type=int, default=50)
    pr.add_argument("--warmup", type=int, default=5)
    pr.add_argument("--frames", type=int, default=100, help="frames per async/parallel bench")
    pr.add_argument("--imgsz", type=int, default=640)
    pr.add_argument("--video", default=None, help="optional real video path for end-to-end benches")
    pr.add_argument("--model", default="models/FunGen-12s-pov-1.1.0.pt")
    pr.add_argument("--out", default="bench_results", help="directory to write JSON reports into; '' to disable")
    pr.add_argument("--tag", default="", help="label attached to output filenames and metadata")
    pr.set_defaults(handler=_cmd_run)

    pc = sub.add_parser("compare", help="Compare two JSON report files from `run`")
    pc.add_argument("before")
    pc.add_argument("after")
    pc.set_defaults(handler=_cmd_compare)

    args = p.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
