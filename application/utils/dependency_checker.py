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
    Checks if essential packages are installed and upgrades them if needed.
    Returns: True if any packages were installed/upgraded (requiring restart)
    """
    missing = []
    to_upgrade = []
    
    for package_spec in packages:
        package_name, version_spec = _parse_package_spec(package_spec)
        
        try:
            installed_version = version(package_name)
            if version_spec:
                is_compatible, needs_upgrade = _check_version_compatibility(installed_version, version_spec)
                if not is_compatible:
                    print(f"Package {package_name} version {installed_version} doesn't satisfy {version_spec}")
                    #to_upgrade.append(package_spec)
            # else: package exists and no version requirement, keep it
        except PackageNotFoundError:
            missing.append(package_spec)

    packages_changed = False
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            packages_changed = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install required packages. Please install them manually and restart.", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)
    
    if to_upgrade:
        print(f"Upgrading packages: {', '.join(to_upgrade)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *to_upgrade])
            packages_changed = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to upgrade packages. Please upgrade them manually and restart.", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)
    
    return packages_changed

def _ensure_packages_with_args(packages, pip_args):
    """
    Checks if essential packages are installed and upgrades them with custom pip arguments.
    Returns: True if any packages were installed/upgraded (requiring restart)
    """
    missing = []
    to_upgrade = []
    
    for package_spec in packages:
        package_name, version_spec = _parse_package_spec(package_spec)
        
        try:
            installed_version = version(package_name)
            if version_spec:
                is_compatible, needs_upgrade = _check_version_compatibility(installed_version, version_spec)
                if not is_compatible:
                    print(f"Package {package_name} version {installed_version} doesn't satisfy {version_spec}")
                    to_upgrade.append(package_spec)
        except PackageNotFoundError:
            missing.append(package_spec)

    packages_changed = False
    
    if missing:
        print(f"Installing missing packages with custom index: {', '.join(missing)}")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + pip_args + missing
            subprocess.check_call(cmd)
            packages_changed = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install required packages. Please install them manually and restart.", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)
    
    if to_upgrade:
        print(f"Upgrading packages with custom index: {', '.join(to_upgrade)}")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + pip_args + to_upgrade
            subprocess.check_call(cmd)
            packages_changed = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to upgrade packages. Please upgrade them manually and restart.", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)
    
    return packages_changed

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
    
    # Now that we are sure they exist, we can import them.
    import requests
    from tqdm import tqdm

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
        print("Installing core packages...")
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
                print(f"Installing GPU-specific packages...")
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
    bin_dir = get_bin_dir()
    if bin_dir not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] = os.pathsep.join([bin_dir, os.environ['PATH']])

    if is_tool('ffmpeg') and is_tool('ffprobe'):
        print("ffmpeg and ffprobe are available.")
    else:
        print("ffmpeg and/or ffprobe not found. Attempting to download...")
        download_ffmpeg(bin_dir, requests, tqdm)

    print("--- Dependency Check Finished ---\n")


def download_ffmpeg(bin_dir, requests, tqdm):
    """Downloads and extracts ffmpeg and ffprobe."""
    os.makedirs(bin_dir, exist_ok=True)

    system = platform.system()
    arch = platform.machine()

    if system == "Windows":
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        archive_path = os.path.join(bin_dir, "ffmpeg.zip")
        extract_dir = os.path.join(bin_dir, "ffmpeg-temp-extract")
        ffmpeg_exe = "ffmpeg.exe"
        ffprobe_exe = "ffprobe.exe"
        bin_in_archive = "bin"
    elif system == "Darwin": # macOS
        # URL for Apple Silicon (arm64) vs Intel (x86_64)
        url = "https://evermeet.cx/ffmpeg/ffmpeg-113893-g37b9f52534.zip" # This is an older build that might be intel
        print("Warning: Using a potentially outdated ffmpeg build for macOS. If you encounter issues, please install ffmpeg manually via Homebrew ('brew install ffmpeg').")
        archive_path = os.path.join(bin_dir, "ffmpeg.zip")
        extract_dir = bin_dir # This specific zip extracts directly
        ffmpeg_exe = "ffmpeg"
        ffprobe_exe = "ffprobe"
        bin_in_archive = None # Not needed
    else: # Linux (assuming x86_64)
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        archive_path = os.path.join(bin_dir, "ffmpeg.tar.xz")
        extract_dir = os.path.join(bin_dir, "ffmpeg-temp-extract")
        ffmpeg_exe = "ffmpeg"
        ffprobe_exe = "ffprobe"
        bin_in_archive = "ffmpeg-*-amd64-static" # Using a glob pattern

    # Download the archive
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(archive_path, 'wb') as f, tqdm(
            desc="Downloading ffmpeg", total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download ffmpeg: {e}", file=sys.stderr)
        return

    # Extract the archive
    print("Extracting ffmpeg...")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        if archive_path.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.xz'):
            import tarfile
            with tarfile.open(archive_path, 'r:xz') as tar_ref:
                tar_ref.extractall(extract_dir)
    except Exception as e:
        print(f"ERROR: Failed to extract ffmpeg: {e}", file=sys.stderr)
        return
    finally:
        os.remove(archive_path)

    # Find and move binaries
    try:
        source_bin_dir = extract_dir
        if bin_in_archive:
            import glob
            # Find the directory that matches the pattern
            found_dirs = glob.glob(os.path.join(extract_dir, bin_in_archive))
            if not found_dirs:
                raise FileNotFoundError("Could not find ffmpeg binaries in extracted archive.")
            source_bin_dir = found_dirs[0]


        shutil.move(os.path.join(source_bin_dir, ffmpeg_exe), os.path.join(bin_dir, ffmpeg_exe))
        shutil.move(os.path.join(source_bin_dir, ffprobe_exe), os.path.join(bin_dir, ffprobe_exe))

        # Clean up the extraction directory
        shutil.rmtree(extract_dir, ignore_errors=True)
        # Also remove the parent dir if it was a zip extraction
        if "ffmpeg-master-latest-win64-gpl" in source_bin_dir:
             shutil.rmtree(os.path.dirname(source_bin_dir), ignore_errors=True)


    except Exception as e:
        print(f"ERROR: Could not locate and move ffmpeg binaries: {e}", file=sys.stderr)
        return

    # Make executable on non-windows
    if system != "Windows":
        os.chmod(os.path.join(bin_dir, ffmpeg_exe), 0o755)
        os.chmod(os.path.join(bin_dir, ffprobe_exe), 0o755)

    print("ffmpeg and ffprobe downloaded and configured successfully.")

if __name__ == '__main__':
    check_and_install_dependencies()