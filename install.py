#!/usr/bin/env python3
"""
FunGen installer. Stdlib-only.

Designed to be invoked three ways, all interchangeable:
  1. From a clone:  python install.py
  2. From a shim:   install.bat / install.sh   (bootstraps uv, then runs us)
  3. From a URL:    uv run --no-project --python 3.11 https://.../install.py

In modes 2 and 3, the script may be running from a temp cache (uv downloads
remote scripts there). It locates or clones the FunGen repo into the user's
current working directory before doing anything else.

Builds a self-contained .venv inside the project, picks the right torch wheel
channel for the GPU it finds, then asks (once) whether to clean up any leftover
miniconda FunGen env from older installs.

Re-runnable: blowing away .venv and re-running always lands on a known-good
environment for the current machine.
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WIN = platform.system() == "Windows"
REPO_HTTPS = "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator.git"
REPO_ZIP = "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/archive/refs/heads/main.zip"


def _find_or_clone_project() -> Path:
    """Return the FunGen project root, cloning the repo into cwd/FunGen if missing."""
    # 1. Are we inside a FunGen checkout? (locally-invoked python install.py)
    here = Path(__file__).parent.resolve()
    if (here / "requirements" / "base.txt").exists():
        return here

    # 2. Did the user cd into a FunGen checkout? (uv-from-URL invocation)
    cwd = Path.cwd().resolve()
    if (cwd / "requirements" / "base.txt").exists():
        return cwd

    # 3. Has a previous run already cloned ./FunGen/?
    sub = cwd / "FunGen"
    if (sub / "requirements" / "base.txt").exists():
        return sub

    # 4. Nothing here. Clone the repo into ./FunGen/.
    print()
    print(f"== Fetching FunGen source into {sub} ==")
    sub.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_HTTPS, str(sub)], check=True)
    else:
        print("  git not found, downloading source archive instead...")
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "fungen.zip"
            urllib.request.urlretrieve(REPO_ZIP, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            extracted = next(p for p in Path(tmp).iterdir()
                             if p.is_dir() and p.name.startswith("FunGen"))
            shutil.move(str(extracted), str(sub))
        print("  NOTE: installed without git, so the in-app updater will not work.")
        print("        To enable updates later: install git, remove the FunGen folder,")
        print("        and re-run install.")
    if not (sub / "requirements" / "base.txt").exists():
        sys.exit(f"FunGen source did not arrive at {sub}; aborting.")
    return sub


ROOT = _find_or_clone_project()
VENV = ROOT / ".venv"
PY_BIN = VENV / ("Scripts/python.exe" if WIN else "bin/python")
CONDA_PROMPTED_MARKER = ROOT / ".fungen_conda_prompted"


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(*args, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(a) for a in args)}", flush=True)
    return subprocess.run(list(args), check=True, **kwargs)


def _print_section(title: str) -> None:
    print()
    print(f"== {title} ==")


# ─── uv bootstrap ───────────────────────────────────────────────────────────

def ensure_uv() -> Path:
    """Find or install uv. Falls back to `pip install uv` if astral.sh is unreachable."""
    existing = shutil.which("uv")
    if existing:
        return Path(existing)

    _print_section("Installing uv (one-time, ~15 MB)")
    try:
        if WIN:
            _run("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-Command", "irm https://astral.sh/uv/install.ps1 | iex")
        else:
            _run("sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  astral.sh installer unreachable ({exc}); trying pip fallback...")
        _pip_install_uv_fallback()

    uv = shutil.which("uv")
    if not uv:
        # uv installer drops the binary in standard locations the new shell hasn't picked up yet.
        for cand in (
            Path.home() / ".local" / "bin" / ("uv.exe" if WIN else "uv"),
            Path.home() / ".cargo" / "bin" / ("uv.exe" if WIN else "uv"),
        ):
            if cand.exists():
                return cand
        sys.exit("Failed to locate uv after install. Install manually: pip install uv")
    return Path(uv)


def _pip_install_uv_fallback() -> None:
    for py in ("python3", "python", "py"):
        if not _have(py):
            continue
        try:
            _run(py, "-m", "pip", "install", "--user", "uv")
            return
        except subprocess.CalledProcessError:
            continue
    sys.exit("Could not install uv via curl or pip. Check your internet connection.")


# ─── GPU channel detection ──────────────────────────────────────────────────

def detect_channel() -> str:
    """Return the requirements/<channel>.txt name suffix to install."""
    nvidia_cap = _nvidia_compute_cap()
    if nvidia_cap is not None:
        return "cuda_blackwell" if nvidia_cap >= 12.0 else "cuda_stable"

    nvidia_name = _nvidia_gpu_name()
    if nvidia_name:
        if re.search(r"RTX\s*50\d{2}|Blackwell", nvidia_name, re.IGNORECASE):
            return "cuda_blackwell"
        if re.search(r"(RTX|GTX|Quadro|Tesla|A\d{2,}|H\d{2,}|L\d{2,})",
                     nvidia_name, re.IGNORECASE):
            return "cuda_stable"

    if platform.system() == "Linux" and (_have("rocm-smi") or Path("/opt/rocm").exists()):
        return "rocm"

    if platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        return "mps"

    return "cpu"


def _nvidia_compute_cap() -> float | None:
    """Highest CUDA compute capability across attached NVIDIA GPUs, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    caps: list[float] = []
    for line in out.splitlines():
        try:
            caps.append(float(line.strip()))
        except ValueError:
            continue  # "N/A" on very old drivers
    return max(caps) if caps else None


def _nvidia_gpu_name() -> str | None:
    """First attached NVIDIA GPU name, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    name = out.strip().split("\n", 1)[0].strip()
    return name or None


# ─── env build ──────────────────────────────────────────────────────────────

def build_env(uv: Path, channel: str) -> None:
    base_lock = ROOT / "requirements" / "base.txt"
    channel_lock = ROOT / "requirements" / f"{channel}.txt"
    if not base_lock.exists():
        sys.exit(f"Missing requirements file: {base_lock}")
    if not channel_lock.exists():
        sys.exit(f"Missing requirements file: {channel_lock}")

    _print_section(f"Creating Python 3.11 environment at {VENV}")
    if VENV.exists():
        shutil.rmtree(VENV)
    # --seed installs pip into the venv. Required because the runtime
    # dependency checker shells out to `python -m pip install ...` to
    # auto-install missing packages after a `git pull`; without --seed,
    # the venv has no pip and that call fails with "No module named pip".
    _run(str(uv), "venv", "--python", "3.11", "--seed", str(VENV))

    _print_section(f"Installing dependencies (channel: {channel})")
    # unsafe-best-match: uv default (first-index) refuses to look at PyPI when
    # a GPU lock file declares --index-url to the pytorch index, so packages
    # like imgui (PyPI-only) fail to resolve. Pip behaves like best-match by
    # default; this restores that for uv.
    _run(str(uv), "pip", "install", "--python", str(PY_BIN),
         "--index-strategy", "unsafe-best-match",
         "-r", str(base_lock), "-r", str(channel_lock))


# ─── conda cleanup prompt (one-shot) ────────────────────────────────────────

def offer_conda_cleanup() -> None:
    if CONDA_PROMPTED_MARKER.exists():
        return

    conda_root = Path.home() / "miniconda3"
    fungen_env = conda_root / "envs" / "FunGen"
    if not fungen_env.exists():
        return

    envs_dir = conda_root / "envs"
    other_envs: list[Path] = []
    if envs_dir.exists():
        other_envs = [p for p in envs_dir.iterdir() if p.is_dir() and p.name != "FunGen"]

    _print_section("Legacy miniconda environment detected")
    print(f"Found a previous FunGen conda env at: {fungen_env}")

    if other_envs:
        print(f"Your miniconda also has {len(other_envs)} other env(s) — those won't be touched.")
        print()
        print("  [y] Remove just the FunGen conda env  (frees ~2 GB)")
        print("  [n] Keep it as a fallback              (default)")
        choice = _ask("Choice [y/N]: ", default="n")
        if choice == "y":
            print(f"  Removing {fungen_env}...")
            shutil.rmtree(fungen_env, ignore_errors=True)
            print("  Done. Other conda envs and miniconda itself are untouched.")
    else:
        size_gb = _dir_size_gb(conda_root)
        print("If miniconda was installed specifically for FunGen (no other envs"
              " in it), you may safely remove it.")
        print()
        print("  [1] Remove just the FunGen conda env   (keeps miniconda for future use)")
        print(f"  [2] Remove all of miniconda            (frees ~{size_gb:.1f} GB)")
        print("  [n] Keep everything                    (default)")
        choice = _ask("Choice [1/2/n]: ", default="n")
        if choice == "1":
            print(f"  Removing {fungen_env}...")
            shutil.rmtree(fungen_env, ignore_errors=True)
        elif choice == "2":
            print(f"  Removing {conda_root}...")
            shutil.rmtree(conda_root, ignore_errors=True)

    CONDA_PROMPTED_MARKER.touch()


def _ask(prompt: str, default: str) -> str:
    if not sys.stdin or not sys.stdin.isatty():
        return default
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    return ans or default


def _dir_size_gb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / (1024 ** 3)


# ─── ffmpeg via system package manager (non-fatal) ──────────────────────────

def ensure_ffmpeg() -> None:
    """Try to install ffmpeg via the OS package manager. Non-fatal: prints a
    manual-install hint and returns if no automated path works."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    _print_section("ffmpeg not found, attempting install")
    sys_os = platform.system()
    try:
        if sys_os == "Windows":
            if shutil.which("winget"):
                _run("winget", "install", "-e", "--id", "Gyan.FFmpeg",
                     "--silent", "--accept-source-agreements",
                     "--accept-package-agreements")
            else:
                print("  winget not found. Install ffmpeg manually:")
                print("    https://www.gyan.dev/ffmpeg/builds/  (download the 'release essentials' zip,")
                print("    extract, and add the bin/ folder to your system PATH).")
                return
        elif sys_os == "Darwin":
            if shutil.which("brew"):
                _run("brew", "install", "ffmpeg")
            else:
                print("  Homebrew not found. Install ffmpeg manually:  brew install ffmpeg")
                print("  (install Homebrew from https://brew.sh first if you don't have it).")
                return
        elif sys_os == "Linux":
            for mgr_check, args in (
                ("apt", ["sudo", "apt", "install", "-y", "ffmpeg"]),
                ("dnf", ["sudo", "dnf", "install", "-y", "ffmpeg"]),
                ("pacman", ["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"]),
            ):
                if shutil.which(mgr_check):
                    _run(*args)
                    break
            else:
                print("  no supported package manager found; install ffmpeg via your distro's tools.")
                return
        else:
            print(f"  unsupported OS ({sys_os}); install ffmpeg manually.")
            return
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg install failed: {e}")
        print("  FunGen will still launch but video features will not work until ffmpeg is on PATH.")


# ─── mpv via system package manager (non-fatal) ─────────────────────────────

def ensure_mpv() -> None:
    """Try to install mpv via the OS package manager. Non-fatal: prints a
    manual-install hint and returns if no automated path works.
    FunGen spawns `mpv` as a subprocess for fullscreen review playback."""
    if shutil.which("mpv"):
        return

    _print_section("mpv not found, attempting install")
    sys_os = platform.system()
    try:
        if sys_os == "Windows":
            if shutil.which("winget"):
                # shinchiro.mpv is the canonical winget package that installs
                # mpv.exe to PATH (other packages like mpv.net ship as mpvnet.exe
                # which FunGen's IPC bridge cannot find).
                _run("winget", "install", "-e", "--id", "shinchiro.mpv",
                     "--silent", "--accept-source-agreements",
                     "--accept-package-agreements")
            else:
                print("  winget not found. Install mpv manually:  https://mpv.io/installation/")
                return
        elif sys_os == "Darwin":
            if shutil.which("brew"):
                _run("brew", "install", "mpv")
            else:
                print("  Homebrew not found. Install mpv manually:  brew install mpv")
                print("  (install Homebrew from https://brew.sh first if you don't have it).")
                return
        elif sys_os == "Linux":
            for mgr_check, args in (
                ("apt", ["sudo", "apt", "install", "-y", "mpv"]),
                ("dnf", ["sudo", "dnf", "install", "-y", "mpv"]),
                ("pacman", ["sudo", "pacman", "-S", "--noconfirm", "mpv"]),
            ):
                if shutil.which(mgr_check):
                    _run(*args)
                    break
            else:
                print("  no supported package manager found; install mpv via your distro's tools.")
                return
        else:
            print(f"  unsupported OS ({sys_os}); install mpv manually.")
            return
    except subprocess.CalledProcessError as e:
        print(f"  mpv install failed: {e}")
        print("  FunGen will still launch but fullscreen video playback will not work.")


# ─── per-OS launch hint ─────────────────────────────────────────────────────

def print_launch_hint() -> None:
    sys_os = platform.system()
    print()
    print("=" * 60)
    print("Done. Launch FunGen with:")
    if sys_os == "Windows":
        print("  Double-click  launch.bat")
    elif sys_os == "Darwin":
        print("  Double-click  launch.command   (Finder)")
        print("  Or run        ./launch.sh      (Terminal)")
    else:
        print("  ./launch.sh")
    print("=" * 60)


# ─── entry point ────────────────────────────────────────────────────────────

def main() -> None:
    _print_section("FunGen installer")
    print(f"  platform: {platform.system()} {platform.machine()}")
    print(f"  project:  {ROOT}")

    uv = ensure_uv()
    print(f"  uv:       {uv}")

    channel = detect_channel()
    print(f"  channel:  {channel}")

    build_env(uv, channel)
    ensure_ffmpeg()
    ensure_mpv()
    offer_conda_cleanup()
    print_launch_hint()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nAborted.")
