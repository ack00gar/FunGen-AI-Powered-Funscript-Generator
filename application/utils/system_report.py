"""Centralized system report for troubleshooting and bug reports."""

import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple


def _anonymize_path(path: str) -> str:
    """Replace the user home directory with ~ to avoid leaking usernames."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def _run_cmd(cmd: List[str], timeout: int = 5) -> str:
    """Run a command and return stripped stdout, or empty string on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _get_git_info() -> Dict[str, str]:
    """Get git branch and short commit hash."""
    info = {}
    # Find the repo root (this file lives in application/utils/)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    git_dir = os.path.join(repo_root, ".git")
    if not os.path.isdir(git_dir):
        return info
    info["branch"] = _run_cmd(["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"])
    info["commit"] = _run_cmd(["git", "-C", repo_root, "log", "-1", "--format=%h %s"])
    return info


def _get_ffmpeg_version() -> str:
    raw = _run_cmd(["ffmpeg", "-version"])
    if raw:
        # First line: "ffmpeg version N.N.N ..."
        return raw.splitlines()[0]
    return "not found"


def _get_ffprobe_version() -> str:
    raw = _run_cmd(["ffprobe", "-version"])
    if raw:
        return raw.splitlines()[0]
    return "not found"


def _get_python_packages() -> List[Tuple[str, str]]:
    """Return versions of key packages."""
    packages = [
        "torch", "torchvision", "ultralytics", "onnxruntime", "tensorrt",
        "cv2", "numpy", "scipy", "imgui", "glfw", "pillow", "PIL",
        "pynvml", "psutil", "pyperclip",
    ]
    results = []
    for name in packages:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
            results.append((name, str(ver)))
        except ImportError:
            pass
    return results


def _get_gpu_info() -> List[str]:
    """Detect GPU(s) — NVIDIA, Apple Silicon, or AMD."""
    gpus = []

    # NVIDIA via torch
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpus.append(torch.cuda.get_device_name(i))
    except Exception:
        pass

    # NVIDIA via nvidia-smi fallback
    if not gpus:
        raw = _run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"])
        if raw:
            gpus.extend(line.strip() for line in raw.splitlines() if line.strip())

    # Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        raw = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        if raw:
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("Chipset Model:"):
                    gpus.append(line.split(":", 1)[1].strip())

    # AMD ROCm
    if not gpus:
        raw = _run_cmd(["rocm-smi", "--showproductname"])
        if raw:
            for line in raw.splitlines():
                if "GPU" in line or "card" in line.lower():
                    gpus.append(line.strip())

    return gpus if gpus else ["No GPU detected"]


def _get_cuda_info() -> Dict[str, str]:
    info = {}
    try:
        import torch
        info["available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["version"] = str(torch.version.cuda)
            info["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except ImportError:
        info["available"] = "torch not installed"
    return info


def _get_memory_info() -> str:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"{mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available"
    except ImportError:
        return "psutil not installed"


def _get_addon_versions() -> List[Tuple[str, str]]:
    """Get versions of installed addons."""
    addons = [
        ("device_control", "Device Control"),
        ("streamer", "Video Streamer"),
        ("patreon_features", "Patreon Features"),
    ]
    results = []
    for module_name, display_name in addons:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "unknown")
            results.append((display_name, ver))
        except ImportError:
            pass
    return results


def _get_recent_log_issues(max_lines: int = 30) -> List[str]:
    """Return the last N WARNING/ERROR/CRITICAL lines from the FunGen log file."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(repo_root, "logs", "fungen.log")
    if not os.path.isfile(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read from end efficiently — read last 512KB max
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 512 * 1024)
            f.seek(max(0, size - read_size))
            tail = f.read()

        matches = []
        for line in tail.splitlines():
            if " - WARNING " in line or " - ERROR " in line or " - CRITICAL " in line:
                matches.append(line)

        return matches[-max_lines:]
    except Exception:
        return []


def generate_report() -> str:
    """Generate a full system report as formatted text."""
    from config.constants import APP_NAME, APP_VERSION, DEVICE

    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  {APP_NAME} System Report")
    lines.append(f"{'=' * 60}")

    # App
    lines.append(f"\n--- Application ---")
    lines.append(f"Version:    {APP_VERSION}")
    lines.append(f"Device:     {DEVICE}")
    git = _get_git_info()
    if git.get("branch"):
        lines.append(f"Branch:     {git['branch']}")
    if git.get("commit"):
        lines.append(f"Commit:     {git['commit']}")

    # Addons
    addons = _get_addon_versions()
    if addons:
        lines.append(f"\n--- Addons ---")
        for name, ver in addons:
            lines.append(f"{name}: {ver}")

    # System
    uname = platform.uname()
    lines.append(f"\n--- System ---")
    lines.append(f"OS:         {uname.system} {uname.release} ({uname.version})")
    lines.append(f"Machine:    {uname.machine}")
    lines.append(f"CPU:        {uname.processor or platform.processor() or 'unknown'}")
    try:
        lines.append(f"CPU Cores:  {os.cpu_count()}")
    except Exception:
        pass
    lines.append(f"Memory:     {_get_memory_info()}")

    # GPU
    gpus = _get_gpu_info()
    lines.append(f"\n--- GPU ---")
    for gpu in gpus:
        lines.append(f"  {gpu}")

    # CUDA
    cuda = _get_cuda_info()
    if cuda:
        lines.append(f"\n--- CUDA ---")
        lines.append(f"Available:  {cuda.get('available', 'N/A')}")
        if cuda.get("version"):
            lines.append(f"Version:    {cuda['version']}")
        if cuda.get("cudnn"):
            lines.append(f"cuDNN:      {cuda['cudnn']}")

    # Python
    lines.append(f"\n--- Python ---")
    lines.append(f"Version:    {sys.version}")
    lines.append(f"Executable: {_anonymize_path(sys.executable)}")

    # FFmpeg
    lines.append(f"\n--- FFmpeg ---")
    lines.append(f"ffmpeg:     {_get_ffmpeg_version()}")
    lines.append(f"ffprobe:    {_get_ffprobe_version()}")

    # Key packages
    pkgs = _get_python_packages()
    if pkgs:
        lines.append(f"\n--- Python Packages ---")
        for name, ver in pkgs:
            lines.append(f"{name}: {ver}")

    # Recent warnings/errors from log
    log_lines = _get_recent_log_issues()
    if log_lines:
        lines.append(f"\n--- Recent Warnings / Errors (last {len(log_lines)}) ---")
        lines.extend(log_lines)
    else:
        lines.append(f"\n--- Recent Warnings / Errors ---")
        lines.append("(none found)")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)
