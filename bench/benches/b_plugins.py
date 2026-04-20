"""Bench every funscript plugin on a realistic synthesised script.

Prints per-plugin wall-clock ms and points in/out so we can spot
plugins that are O(n^2), allocate excessively, or drop a ton of
points unexpectedly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _synth_fs(n_points: int, duration_ms: int = 600_000):
    from funscript.multi_axis_funscript import MultiAxisFunscript
    fs = MultiAxisFunscript(fps=60)
    ts = np.linspace(0, duration_ms, n_points).astype(np.int64)
    # Sinusoid with drift and noise; typical tracker output.
    phase = np.linspace(0, 80 * np.pi, n_points)
    drift = np.linspace(0, 10, n_points)
    noise = np.random.default_rng(0).normal(0, 2, n_points)
    pos = np.clip(50 + 40 * np.sin(phase) + drift + noise, 0, 100).astype(int)
    for i in range(n_points):
        fs.primary_actions.append({"at": int(ts[i]), "pos": int(pos[i])})
    fs._invalidate_cache('primary')
    return fs


def _discover_plugins():
    from funscript.plugins.base_plugin import plugin_registry
    from funscript.plugins.plugin_loader import PluginLoader
    loader = PluginLoader()
    loader.load_builtin_plugins()
    return plugin_registry.list_plugins()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    print(f"synthesising funscript with {args.points} points...")
    from funscript.plugins.base_plugin import plugin_registry
    _discover_plugins()
    plist = plugin_registry.list_plugins()
    if plist and isinstance(plist[0], dict):
        names = sorted(p.get('name') or p.get('internal_name') or str(p) for p in plist)
    else:
        names = sorted(plist)
    print(f"discovered {len(names)} plugins: {', '.join(names)}\n")

    print(f"{'plugin':<35s}  {'pts_in':>6s}  {'pts_out':>7s}  {'ms':>8s}  note")
    rows = []
    for name in names:
        plugin = plugin_registry.get_plugin(name)
        if plugin is None:
            continue
        fs = _synth_fs(args.points)
        pts_in = len(fs.primary_actions)
        note = ""
        t0 = time.perf_counter()
        try:
            out = plugin.transform(fs, axis='primary')
        except Exception as e:
            note = f"ERROR: {e!r}"
            elapsed = time.perf_counter() - t0
            print(f"{name:<35s}  {pts_in:6d}  {'-':>7s}  {elapsed*1000:8.2f}  {note}")
            continue
        elapsed = time.perf_counter() - t0
        # Plugin may return modified fs or operate in-place.
        target = out if out is not None else fs
        pts_out = len(target.primary_actions)
        print(f"{name:<35s}  {pts_in:6d}  {pts_out:7d}  {elapsed*1000:8.2f}  {note}")
        rows.append({"name": name, "pts_in": pts_in, "pts_out": pts_out,
                     "ms": elapsed * 1000.0, "note": note})

    out_dir = Path("bench_results")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_plugins_n{args.points}.json"
    with open(path, "w") as f:
        json.dump({"points": args.points, "rows": rows}, f, indent=2)
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
