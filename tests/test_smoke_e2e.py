"""End-to-end smoke test: run main.py CLI on a tiny clip and assert it produces a funscript.

Skipped unless a real test video is available. Locally we look for
~/Downloads/test_koogar_extra_short_A.mp4; CI can override with
FUNGEN_E2E_VIDEO=/path/to/clip.mp4.

The test does not assume a specific output location: FunGen saves to the
configured `output_folder_path` (per-video subfolder, with `_t1_raw` suffix on
the basename), and may also save a copy next to the video. We glob for any
funscript whose name embeds the video stem.

Marked `integration` and `slow` so a fast `pytest -m "not slow"` skips it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_VIDEO = Path.home() / "Downloads" / "test_koogar_extra_short_A.mp4"


def _resolve_test_video() -> Path | None:
    env = os.environ.get("FUNGEN_E2E_VIDEO")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    return DEFAULT_TEST_VIDEO if DEFAULT_TEST_VIDEO.exists() else None


def _resolve_output_folder() -> Path:
    settings_path = REPO_ROOT / "settings.json"
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text())
            cfg = data.get("output_folder_path")
            if cfg:
                return Path(cfg).expanduser()
        except Exception:
            pass
    return REPO_ROOT / "output"


@pytest.mark.integration
@pytest.mark.slow
def test_cli_produces_funscript(tmp_path: Path) -> None:
    src = _resolve_test_video()
    if src is None:
        pytest.skip(
            "no test video; set FUNGEN_E2E_VIDEO or place a clip at "
            f"{DEFAULT_TEST_VIDEO}"
        )

    work_video = tmp_path / src.name
    shutil.copy2(src, work_video)
    stem = work_video.stem

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        str(work_video),
        "--quiet",
        "--overwrite",
        "--no-autotune",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert proc.returncode == 0, (
        f"main.py exited {proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )

    output_folder = _resolve_output_folder()
    candidates: list[Path] = []
    candidates.extend(tmp_path.rglob(f"{stem}*.funscript"))
    if output_folder.exists():
        candidates.extend(output_folder.rglob(f"{stem}*.funscript"))

    candidates = [p for p in candidates if not p.name.endswith(".bak")]

    assert candidates, (
        f"no funscript produced for {stem}. searched tmp_path={tmp_path} and "
        f"output_folder={output_folder}.\n"
        f"stdout tail:\n{proc.stdout[-2000:]}"
    )
    biggest = max(candidates, key=lambda p: p.stat().st_size)
    assert biggest.stat().st_size > 0, f"funscript empty: {biggest}"
