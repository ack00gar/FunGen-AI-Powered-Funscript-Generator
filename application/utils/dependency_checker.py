import subprocess
import sys
import os
import shutil
import platform
import logging
from importlib.metadata import version, PackageNotFoundError
from packaging import version as pkg_version
from packaging.specifiers import SpecifierSet

logger = logging.getLogger(__name__)

def _parse_package_spec(package_spec):
    """
    Parses a package specification and returns (name, version_spec).
    Examples: 'torch~=2.5.1' -> ('torch', '~=2.5.1')
              'numpy' -> ('numpy', None)
    """
    # Split on version operators
    for op in ['~=', '>=', '<=', '==', '!=', '>', '<']:
        if op in package_spec:
            name, spec = package_spec.split(op, 1)
            return name.strip(), f"{op}{spec.strip()}"
    return package_spec.strip(), None

def _check_version_compatibility(installed_version, required_spec):
    """
    Checks if installed version satisfies the required specification.
    Returns: (is_compatible, needs_upgrade)
    """
    if not required_spec:
        return True, False
    
    try:
        spec_set = SpecifierSet(required_spec)
        installed = pkg_version.parse(installed_version)
        is_compatible = installed in spec_set
        
        # Check if we need to upgrade (installed version is too old)
        needs_upgrade = not is_compatible
        return is_compatible, needs_upgrade
    except Exception:
        # If we can't parse versions, assume compatible
        return True, False

def _ensure_packages(packages, pip_args=None, *, non_interactive: bool = True, auto_install: bool = True):
    """
    Ensures required packages are installed. Supports optional pip arguments (e.g., custom index URLs).
    Returns: True if any packages were installed (requiring restart)
    """
    missing = []
    for package_spec in packages:
        package_name, _ = _parse_package_spec(package_spec)
        try:
            version(package_name)
        except PackageNotFoundError:
            missing.append(package_spec)

    if not missing:
        return False

    logger.warning(f"The following required packages are missing: {', '.join(missing)}")
    install_cmd = [sys.executable, "-m", "pip", "install"] + (pip_args or []) + missing
    try:
        if non_interactive and auto_install:
            if pip_args:
                logger.info(f"Auto-installing with custom args ({' '.join(pip_args)}): {', '.join(missing)}")
            else:
                logger.info(f"Auto-installing missing packages: {', '.join(missing)}")
            subprocess.check_call(install_cmd)
            return True
        elif non_interactive and not auto_install:
            logger.warning("Non-interactive mode: skipping auto-install. Application may not function correctly.")
            return False
        else:
            prompt = "Would you like to install them now" + (" using custom arguments" if pip_args else "") + "? (y/n): "
            response = input(prompt).lower()
            if response == 'y':
                if pip_args:
                    logger.info(f"Installing missing packages with custom args ({' '.join(pip_args)}): {', '.join(missing)}")
                else:
                    logger.info(f"Installing missing packages: {', '.join(missing)}")
                subprocess.check_call(install_cmd)
                return True
            else:
                logger.warning("Installation skipped. The application may not function correctly.")
                return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to install required packages: {e}")
        logger.error("Please install them manually and restart.")
        sys.exit(1)

# Note: _ensure_packages_with_args was merged into _ensure_packages via the optional pip_args parameter

def get_bin_dir():
    """Gets the directory where binaries like ffmpeg should be stored."""
    # Place bin folder in the project root
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'bin')

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

def detect_gpu_environment():
    """
    Detects the GPU environment and returns the appropriate requirements file.
    Returns: (requirements_file, environment_description)
    """
    system = platform.system()
    
    # macOS: Use core requirements (MPS/CPU PyTorch)
    if system == "Darwin":
        return "core.requirements.txt", "macOS (Metal/CPU)"
    
    # Windows/Linux: Detect GPU type
    cuda_available = False
    rocm_available = False
    rtx_50_series = False
    
    # Check for NVIDIA CUDA
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cuda_available = True
            gpu_names = result.stdout.strip().split('\n')
            for gpu_name in gpu_names:
                # Check for RTX 50-series (5070, 5080, 5090)
                if any(model in gpu_name.upper() for model in ['RTX 507', 'RTX 508', 'RTX 509']):
                    rtx_50_series = True
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Check for AMD ROCm (Linux and Windows)
    if not cuda_available:
        try:
            result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                rocm_available = True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
    
    # Return appropriate requirements file
    if rtx_50_series:
        return "cuda.50series.requirements.txt", "NVIDIA RTX 50-series (CUDA)"
    elif cuda_available:
        return "cuda.requirements.txt", "NVIDIA CUDA"
    elif rocm_available:
        return "rocm.requirements.txt", "AMD ROCm"
    else:
        return "core.requirements.txt", "CPU-only"

def check_and_install_dependencies(*, non_interactive: bool = True, auto_install: bool = True):
    """
    Checks for and installs missing dependencies.
    This function is designed to be run before the main application starts.
    """
    # 1. Self-bootstrap: Ensure the checker has its own dependencies
    bootstrap_changed = _ensure_packages(['requests', 'tqdm', 'packaging'], pip_args=None, non_interactive=non_interactive, auto_install=auto_install)

    logger.info("=== Checking Application Dependencies ===")

    # 2. Detect GPU environment and select appropriate requirements
    requirements_file, env_description = detect_gpu_environment()
    logger.debug(f"Detected environment: {env_description}")
    logger.debug(f"Using requirements file: {requirements_file}")

    # 3. Load and install core requirements first
    try:
        with open('core.requirements.txt', 'r') as f:
            core_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        logger.error("core.requirements.txt not found.")
        sys.exit(1)

    core_changed = False
    if core_packages:
        logger.info("Checking core packages...")
        core_changed = _ensure_packages(core_packages, pip_args=None, non_interactive=non_interactive, auto_install=auto_install)

    # 4. Load and install GPU-specific requirements if needed
    gpu_changed = False
    if requirements_file != "core.requirements.txt":
        try:
            with open(requirements_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                # Handle pip index URLs (like -i https://download.pytorch.org/whl/cu128)
                pip_extra_args = []
                gpu_packages = []
                
                for line in lines:
                    if line.startswith('-i ') or line.startswith('--index-url '):
                        pip_extra_args.extend(line.split())
                    else:
                        gpu_packages.append(line)

            if gpu_packages:
                logger.info("Checking GPU-specific packages...")
                if pip_extra_args:
                    logger.info(f"Using custom index: {' '.join(pip_extra_args)}")
                    gpu_changed = _ensure_packages(gpu_packages, pip_args=pip_extra_args, non_interactive=non_interactive, auto_install=auto_install)
                else:
                    gpu_changed = _ensure_packages(gpu_packages, pip_args=None, non_interactive=non_interactive, auto_install=auto_install)
                    
        except FileNotFoundError:
            logger.warning(f"{requirements_file} not found. Continuing with core packages only.")

    # Check if we need to restart due to major package changes
    major_changes = bootstrap_changed or core_changed or gpu_changed
    
    if major_changes:
        logger.warning("\n=== Package Installation Complete ===")
        logger.warning("IMPORTANT: Major packages were installed/upgraded.")
        logger.warning("Please restart the application to ensure all changes take effect.")
        logger.warning("=== Exiting for Restart ===")
        sys.exit(0)  # Clean exit to allow restart
    
    logger.info("All required packages are installed and up to date.")

    # 5. Verify PyTorch installation
    try:
        version('torch')
        version('torchvision')
        logger.info("PyTorch (torch and torchvision) is installed.")
    except PackageNotFoundError:
        logger.error("\n=== PyTorch Installation Failed ===")
        logger.error("PyTorch installation may have failed. Please check the installation.")
        logger.error("Installation guide: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # 6. Check for ffmpeg and ffprobe
    check_ffmpeg_ffprobe()

    logger.info("=== Dependency Check Finished ===\n")


def check_ffmpeg_ffprobe(*, non_interactive: bool = True, auto_install: bool = False):
    """Checks for ffmpeg and ffprobe and offers to install them if missing."""
    ffmpeg_missing = not is_tool('ffmpeg')
    ffprobe_missing = not is_tool('ffprobe')

    if ffmpeg_missing or ffprobe_missing:
        missing_tools = []
        if ffmpeg_missing:
            missing_tools.append('ffmpeg')
        if ffprobe_missing:
            missing_tools.append('ffprobe')
        
        logger.warning(f"The following required tools are not found in your system's PATH: {', '.join(missing_tools)}.")
        
        system = platform.system()
        install_cmd = ""
        if system == "Darwin":
            install_cmd = "brew install ffmpeg"
        elif system == "Linux":
            install_cmd = "sudo apt-get update && sudo apt-get install ffmpeg"
        elif system == "Windows":
            # Safer: only suggest Chocolatey if available; otherwise guide manual install
            if shutil.which('choco'):
                install_cmd = "choco install ffmpeg"
            else:
                install_cmd = ""

        if install_cmd:
            try:
                if non_interactive:
                    if auto_install:
                        logger.info(f"Attempting non-interactive install: {install_cmd}")
                        subprocess.check_call(install_cmd, shell=True)
                        if not is_tool('ffmpeg') or not is_tool('ffprobe'):
                            logger.error("Installation may have failed. Please install ffmpeg manually.")
                            sys.exit(1)
                        else:
                            logger.info("ffmpeg installed successfully.")
                    else:
                        logger.warning("Non-interactive mode: skipping ffmpeg auto-install. Please install manually.")
                        sys.exit(1)
                else:
                    response = input(f"Would you like to attempt to install it now using '{install_cmd}'? (y/n): ").lower()
                    if response == 'y':
                        logger.info(f"Running installation command: {install_cmd}")
                        subprocess.check_call(install_cmd, shell=True)
                        # Re-check after installation
                        if not is_tool('ffmpeg') or not is_tool('ffprobe'):
                            logger.error("Installation may have failed. Please install ffmpeg manually.")
                            sys.exit(1)
                        else:
                            logger.info("ffmpeg installed successfully.")
                    else:
                        logger.warning("Installation skipped. Please install ffmpeg manually to proceed.")
                        sys.exit(1)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"Error during installation: {e}")
                logger.error("Please install ffmpeg manually.")
                sys.exit(1)
        else:
            # Provide safer guidance for manual installation on Windows without Chocolatey
            if system == "Windows":
                logger.error("ffmpeg/ffprobe not found. Install manually or install Chocolatey (https://chocolatey.org/install) and run 'choco install ffmpeg'.")
            else:
                logger.error("Could not determine the installation command for your OS. Please install ffmpeg manually.")
            sys.exit(1)
    else:
        logger.info("ffmpeg and ffprobe are available.")

if __name__ == '__main__':
    check_and_install_dependencies()
