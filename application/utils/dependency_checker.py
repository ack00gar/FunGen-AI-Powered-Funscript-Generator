import subprocess
import sys
import os
import shutil
import platform
from importlib.metadata import version, PackageNotFoundError
from packaging import version as pkg_version
from packaging.specifiers import SpecifierSet

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

def _ensure_packages(packages):
    """
    Checks if essential packages are installed and offers to install them if missing.
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

    print(f"The following required packages are missing: {', '.join(missing)}")
    try:
        response = input(f"Would you like to install them now? (y/n): ").lower()
        if response == 'y':
            print(f"Installing missing packages: {', '.join(missing)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            return True
        else:
            print("Installation skipped. The application may not function correctly.", file=sys.stderr)
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: Failed to install required packages: {e}", file=sys.stderr)
        print("Please install them manually and restart.", file=sys.stderr)
        sys.exit(1)

def _ensure_packages_with_args(packages, pip_args):
    """
    Checks if essential packages are installed and offers to install them with custom pip arguments.
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

    print(f"The following required packages are missing: {', '.join(missing)}")
    try:
        response = input(f"Would you like to install them now using custom arguments? (y/n): ").lower()
        if response == 'y':
            print(f"Installing missing packages with custom index: {', '.join(missing)}")
            cmd = [sys.executable, "-m", "pip", "install"] + pip_args + missing
            subprocess.check_call(cmd)
            return True
        else:
            print("Installation skipped. The application may not function correctly.", file=sys.stderr)
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: Failed to install required packages: {e}", file=sys.stderr)
        print("Please install them manually and restart.", file=sys.stderr)
        sys.exit(1)

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
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
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
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  capture_output=True, text=True, timeout=5)
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

def check_and_install_dependencies():
    """
    Checks for and installs missing dependencies.
    This function is designed to be run before the main application starts.
    """
    # 1. Self-bootstrap: Ensure the checker has its own dependencies
    bootstrap_changed = _ensure_packages(['requests', 'tqdm', 'packaging'])

    print("--- Checking Application Dependencies ---")

    # 2. Detect GPU environment and select appropriate requirements
    requirements_file, env_description = detect_gpu_environment()
    print(f"Detected environment: {env_description}")
    print(f"Using requirements file: {requirements_file}")

    # 3. Load and install core requirements first
    try:
        with open('core.requirements.txt', 'r') as f:
            core_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("ERROR: core.requirements.txt not found.", file=sys.stderr)
        sys.exit(1)

    core_changed = False
    if core_packages:
        print("Checking core packages...")
        core_changed = _ensure_packages(core_packages)

    # 4. Load and install GPU-specific requirements if needed
    gpu_changed = False
    if requirements_file != "core.requirements.txt":
        try:
            with open(requirements_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                # Handle pip index URLs (like -i https://download.pytorch.org/whl/cu129)
                pip_extra_args = []
                gpu_packages = []
                
                for line in lines:
                    if line.startswith('-i ') or line.startswith('--index-url '):
                        pip_extra_args.extend(line.split())
                    else:
                        gpu_packages.append(line)

            if gpu_packages:
                print(f"Checking GPU-specific packages...")
                if pip_extra_args:
                    print(f"Using custom index: {' '.join(pip_extra_args)}")
                    gpu_changed = _ensure_packages_with_args(gpu_packages, pip_extra_args)
                else:
                    gpu_changed = _ensure_packages(gpu_packages)
                    
        except FileNotFoundError:
            print(f"WARNING: {requirements_file} not found. Continuing with core packages only.", file=sys.stderr)

    # Check if we need to restart due to major package changes
    major_changes = bootstrap_changed or core_changed or gpu_changed
    
    if major_changes:
        print("\n--- Package Installation Complete ---")
        print("IMPORTANT: Major packages were installed/upgraded.")
        print("Please restart the application to ensure all changes take effect.")
        print("--- Exiting for Restart ---")
        sys.exit(0)  # Clean exit to allow restart
    
    print("All required packages are installed and up to date.")

    # 5. Verify PyTorch installation
    try:
        version('torch')
        version('torchvision')
        print("PyTorch (torch and torchvision) is installed.")
    except PackageNotFoundError:
        print("\n--- PyTorch Installation Failed ---", file=sys.stderr)
        print("PyTorch installation may have failed. Please check the installation.", file=sys.stderr)
        print("Installation guide: https://pytorch.org/get-started/locally/", file=sys.stderr)
        sys.exit(1)

    # 6. Check for ffmpeg and ffprobe
    check_ffmpeg_ffprobe()

    print("--- Dependency Check Finished ---\n")


def check_ffmpeg_ffprobe():
    """Checks for ffmpeg and ffprobe and offers to install them if missing."""
    ffmpeg_missing = not is_tool('ffmpeg')
    ffprobe_missing = not is_tool('ffprobe')

    if ffmpeg_missing or ffprobe_missing:
        missing_tools = []
        if ffmpeg_missing:
            missing_tools.append('ffmpeg')
        if ffprobe_missing:
            missing_tools.append('ffprobe')
        
        print(f"WARNING: The following required tools are not found in your system's PATH: {', '.join(missing_tools)}.")
        
        system = platform.system()
        install_cmd = ""
        if system == "Darwin":
            install_cmd = "brew install ffmpeg"
        elif system == "Linux":
            install_cmd = "sudo apt-get update && sudo apt-get install ffmpeg"
        elif system == "Windows":
            install_cmd = "choco install ffmpeg"

        if install_cmd:
            try:
                response = input(f"Would you like to attempt to install it now using '{install_cmd}'? (y/n): ").lower()
                if response == 'y':
                    print(f"Running installation command: {install_cmd}")
                    subprocess.check_call(install_cmd, shell=True)
                    # Re-check after installation
                    if not is_tool('ffmpeg') or not is_tool('ffprobe'):
                        print("Installation may have failed. Please install ffmpeg manually.", file=sys.stderr)
                        sys.exit(1)
                    else:
                        print("ffmpeg installed successfully.")
                else:
                    print("Installation skipped. Please install ffmpeg manually to proceed.", file=sys.stderr)
                    sys.exit(1)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Error during installation: {e}", file=sys.stderr)
                print(f"Please install ffmpeg manually.", file=sys.stderr)
                sys.exit(1)
        else:
            print("Could not determine the installation command for your OS. Please install ffmpeg manually.", file=sys.stderr)
            sys.exit(1)
    else:
        print("ffmpeg and ffprobe are available.")


if __name__ == '__main__':
    check_and_install_dependencies()
