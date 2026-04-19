"""Resource path resolution.

Single source of truth for locating bundled resources (models, assets, fonts,
patterns, bin tools, logs). Robust to running from source, from Cython-compiled
extension modules, or from a frozen (Nuitka/PyInstaller) binary -- in each of
those environments __file__ either points at a non-Python artifact or is unset,
so callers must not compute resource paths via __file__ themselves.

Resolution order:
  1. FUNGEN_ROOT env var (useful for installer staging / CI / tests)
  2. Frozen build: directory containing sys.executable
  3. Source / Cython build: walk up from this file until we find the app root
     marker (main.py + assets/ both present)
  4. Fallback: current working directory
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def _find_app_root() -> Path:
    env = os.environ.get("FUNGEN_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p

    if _is_frozen():
        return Path(sys.executable).resolve().parent

    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()

    for candidate in (start, *start.parents):
        if (candidate / "main.py").is_file() and (candidate / "assets").is_dir():
            return candidate

    return Path.cwd()


APP_ROOT: Path = _find_app_root()

ASSETS_DIR: Path = APP_ROOT / "assets"
MODELS_DIR: Path = APP_ROOT / "models"
CONFIG_DIR: Path = APP_ROOT / "config"
LOGS_DIR: Path = APP_ROOT / "logs"
BIN_DIR: Path = APP_ROOT / "bin"
PATTERNS_DIR: Path = APP_ROOT / "patterns"

BRANDING_DIR: Path = ASSETS_DIR / "branding"
FONTS_DIR: Path = ASSETS_DIR / "fonts"
SPLASH_FRAMES_DIR: Path = ASSETS_DIR / "splash_frames"

LOGO_PATH: Path = BRANDING_DIR / "logo.png"
ICON_FONT_PATH: Path = FONTS_DIR / "icons.ttf"
SUPPORT_BADGE_PATH: Path = BRANDING_DIR / "support_badge.png"


def asset(*parts: str) -> Path:
    return ASSETS_DIR.joinpath(*parts)


def model(*parts: str) -> Path:
    return MODELS_DIR.joinpath(*parts)


def app_log_file(name: str = "fungen.log") -> Path:
    return LOGS_DIR / name
