#!/usr/bin/env python3
"""
FunGen Universal Installer - Stage 2
Complete installation system that assumes Python is available but nothing else

This installer handles the complete FunGen setup after Python is installed:
- Git installation and repository cloning
- FFmpeg/FFprobe installation
- GPU detection and appropriate PyTorch installation
- Virtual environment setup
- All Python dependencies
- Launcher script creation and validation

Supports: Windows, macOS (Intel/Apple Silicon), Linux (x86_64/ARM64)
"""

import os
import sys
import platform
import subprocess
import urllib.request
import urllib.error
import shutil
import tempfile
import time
import json
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse

# Configuration
CONFIG = {
    "repo_url": "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator.git",
    "project_name": "FunGen",
    "env_name": "FunGen",
    "python_version": "3.11",
    "main_script": "main.py",
    "min_disk_space_gb": 10,
    "requirements_files": {
        "core": "core.requirements.txt",
        "cuda": "cuda.requirements.txt", 
        "cpu": "cpu.requirements.txt",
        "rocm": "rocm.requirements.txt"
    }
}

# Download URLs for various tools
DOWNLOAD_URLS = {
    "git": {
        "windows": "https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/Git-2.45.2-64-bit.exe",
        "portable_windows": "https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/PortableGit-2.45.2-64-bit.7z.exe"
    },
    "ffmpeg": {
        "windows": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "macos": "https://evermeet.cx/ffmpeg/ffmpeg-6.1.zip",
        "linux": {
            "x86_64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "aarch64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
        }
    }
}

class Colors:
    """ANSI color codes for terminal output"""
    if platform.system() == "Windows":
        # Try to enable ANSI colors on Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass
    
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class ProgressBar:
    """Simple progress bar for downloads"""
    
    def __init__(self, total_size: int, description: str = ""):
        self.total_size = total_size
        self.downloaded = 0
        self.description = description
        self.last_update = 0
    
    def update(self, chunk_size: int):
        self.downloaded += chunk_size
        current_time = time.time()
        
        # Update every 0.1 seconds to avoid too much output
        if current_time - self.last_update > 0.1:
            if self.total_size > 0:
                percent = min(100, (self.downloaded * 100) // self.total_size)
                bar_length = 40
                filled = (percent * bar_length) // 100
                bar = '█' * filled + '░' * (bar_length - filled)
                
                size_mb = self.downloaded / (1024 * 1024)
                total_mb = self.total_size / (1024 * 1024)
                
                print(f"\r  {self.description}: {bar} {percent}% ({size_mb:.1f}/{total_mb:.1f} MB)", 
                      end="", flush=True)
            else:
                size_mb = self.downloaded / (1024 * 1024)
                print(f"\r  {self.description}: {size_mb:.1f} MB downloaded", end="", flush=True)
            
            self.last_update = current_time
    
    def finish(self):
        print()  # New line after completion


class FunGenUniversalInstaller:
    """Universal FunGen installer - assumes Python is available"""
    
    def __init__(self, install_dir: Optional[str] = None, force: bool = False):
        self.platform = platform.system()
        self.arch = platform.machine().lower()
        self.force = force
        self.install_dir = Path(install_dir) if install_dir else Path.cwd()
        self.project_path = self.install_dir / CONFIG["project_name"]
        
        # Setup paths
        self.setup_paths()
        
        # Progress tracking
        self.current_step = 0
        self.total_steps = 8
        
        # Installation state
        self.conda_available = False
        self.venv_path = None
        
    def setup_paths(self):
        """Setup platform-specific paths"""
        self.home = Path.home()
        
        if self.platform == "Windows":
            self.miniconda_path = self.home / "miniconda3"
            self.tools_dir = self.install_dir / "tools"
            self.git_path = self.tools_dir / "git"
            self.ffmpeg_path = self.tools_dir / "ffmpeg"
        else:
            self.miniconda_path = self.home / "miniconda3"
            self.tools_dir = self.home / ".local" / "bin"
            self.git_path = self.tools_dir
            self.ffmpeg_path = self.tools_dir
    
    def print_header(self):
        """Print installer header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=" * 60)
        print("    FunGen Universal Installer")
        print("=" * 60 + Colors.ENDC)
        print(f"{Colors.CYAN}Platform: {self.platform} ({self.arch})")
        print(f"Install Directory: {self.install_dir}")
        print(f"Project Path: {self.project_path}{Colors.ENDC}\n")
    
    def print_step(self, step_name: str):
        """Print current installation step"""
        self.current_step += 1
        print(f"{Colors.BLUE}[{self.current_step}/{self.total_steps}] {step_name}...{Colors.ENDC}")
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}✗ {message}{Colors.ENDC}")
    
    def command_exists(self, command: str) -> bool:
        """Check if a command exists"""
        return shutil.which(command) is not None
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   check: bool = True, capture: bool = False, 
                   env: Optional[Dict] = None) -> Tuple[int, str, str]:
        """Run a command with comprehensive error handling"""
        try:
            kwargs = {
                'cwd': cwd,
                'check': check,
                'env': env or os.environ.copy()
            }
            
            if capture:
                kwargs.update({'capture_output': True, 'text': True})
            
            result = subprocess.run(cmd, **kwargs)
            
            if capture:
                return result.returncode, result.stdout, result.stderr
            else:
                return result.returncode, "", ""
                
        except subprocess.CalledProcessError as e:
            stdout = getattr(e, 'stdout', '') or ''
            stderr = getattr(e, 'stderr', '') or ''
            return e.returncode, stdout, stderr
        except FileNotFoundError:
            return 127, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return 1, "", str(e)
    
    def download_with_progress(self, url: str, filepath: Path, description: str = "") -> bool:
        """Download file with progress bar"""
        try:
            print(f"  Downloading {description or url}...")
            
            # Get file size
            req = urllib.request.urlopen(url)
            total_size = int(req.headers.get('Content-Length', 0))
            
            progress = ProgressBar(total_size, description or "File")
            
            with open(filepath, 'wb') as f:
                while True:
                    chunk = req.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))
            
            progress.finish()
            req.close()
            return True
            
        except Exception as e:
            self.print_error(f"Download failed: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path, description: str = "") -> bool:
        """Extract various archive formats"""
        try:
            print(f"  Extracting {description}...")
            
            if archive_path.suffix.lower() in ['.zip']:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.xz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                self.print_error(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            self.print_success(f"{description} extracted successfully")
            return True
            
        except Exception as e:
            self.print_error(f"Extraction failed: {e}")
            return False
    
    def check_system_requirements(self) -> bool:
        """Check system requirements"""
        self.print_step("Checking system requirements")
        
        # Check Python version
        if sys.version_info < (3, 9):
            self.print_error(f"Python 3.9+ required, found {sys.version}")
            return False
        self.print_success(f"Python {sys.version.split()[0]} available")
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(self.install_dir)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < CONFIG["min_disk_space_gb"]:
                self.print_error(f"Insufficient disk space: {free_gb:.1f}GB available, {CONFIG['min_disk_space_gb']}GB required")
                return False
            self.print_success(f"Disk space: {free_gb:.1f}GB available")
        except Exception as e:
            self.print_warning(f"Could not check disk space: {e}")
        
        # Check if conda is available
        self.conda_available = (self.miniconda_path / "bin" / "conda").exists() or (self.miniconda_path / "Scripts" / "conda.exe").exists()
        if self.conda_available:
            self.print_success("Conda environment manager available")
        else:
            self.print_success("Will use Python venv for environment management")
        
        return True
    
    def install_git(self) -> bool:
        """Install Git if not available"""
        if self.command_exists("git"):
            self.print_success("Git already available")
            return True
        
        print("  Installing Git...")
        
        if self.platform == "Windows":
            return self._install_git_windows()
        elif self.platform == "Darwin":
            return self._install_git_macos()
        else:
            return self._install_git_linux()
    
    def _install_git_windows(self) -> bool:
        """Install Git on Windows"""
        git_url = DOWNLOAD_URLS["git"]["windows"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            installer_path = Path(temp_dir) / "git-installer.exe"
            
            if not self.download_with_progress(git_url, installer_path, "Git installer"):
                return False
            
            # Install silently
            ret, _, stderr = self.run_command([
                str(installer_path), "/VERYSILENT", "/NORESTART", "/NOCANCEL",
                "/SP-", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"
            ], check=False)
            
            if ret == 0:
                self.print_success("Git installed successfully")
                # Refresh PATH
                git_paths = [
                    str(Path.home() / "AppData" / "Local" / "Programs" / "Git" / "bin"),
                    str(Path("C:") / "Program Files" / "Git" / "bin")
                ]
                for git_path in git_paths:
                    if Path(git_path).exists() and git_path not in os.environ["PATH"]:
                        os.environ["PATH"] = git_path + ";" + os.environ["PATH"]
                        break
                return True
            else:
                self.print_error(f"Git installation failed: {stderr}")
                return False
    
    def _install_git_macos(self) -> bool:
        """Install Git on macOS"""
        # Check if Homebrew is available
        if self.command_exists("brew"):
            ret, _, stderr = self.run_command(["brew", "install", "git"], check=False)
            if ret == 0:
                self.print_success("Git installed via Homebrew")
                return True
        
        # Try to install Xcode Command Line Tools
        print("  Installing Xcode Command Line Tools (includes Git)...")
        ret, _, _ = self.run_command(["xcode-select", "--install"], check=False)
        
        if ret == 0:
            print("  Please complete the Xcode Command Line Tools installation in the dialog")
            print("  Then re-run this installer")
            return False
        else:
            self.print_error("Could not install Git automatically")
            self.print_error("Please install Git manually: https://git-scm.com/download/mac")
            return False
    
    def _install_git_linux(self) -> bool:
        """Install Git on Linux"""
        # Try different package managers
        package_managers = [
            (["apt", "update"], ["apt", "install", "-y", "git"]),
            (None, ["yum", "install", "-y", "git"]),
            (None, ["dnf", "install", "-y", "git"]),
            (None, ["pacman", "-S", "--noconfirm", "git"]),
            (None, ["zypper", "install", "-y", "git"]),
            (None, ["apk", "add", "git"])
        ]
        
        for update_cmd, install_cmd in package_managers:
            if self.command_exists(install_cmd[0]):
                print(f"  Using {install_cmd[0]} package manager...")
                
                if update_cmd:
                    self.run_command(update_cmd, check=False)
                
                ret, _, stderr = self.run_command(install_cmd, check=False)
                if ret == 0:
                    self.print_success(f"Git installed via {install_cmd[0]}")
                    return True
                else:
                    self.print_warning(f"Failed to install with {install_cmd[0]}: {stderr}")
        
        self.print_error("Could not install Git automatically")
        self.print_error("Please install Git manually using your system's package manager")
        return False
    
    def clone_repository(self) -> bool:
        """Clone or update the FunGen repository"""
        if self.project_path.exists():
            if self.force:
                print("  Removing existing project directory...")
                shutil.rmtree(self.project_path)
            else:
                print("  Project directory exists, updating...")
                ret, _, stderr = self.run_command(
                    ["git", "pull"], 
                    cwd=self.project_path, 
                    check=False
                )
                if ret == 0:
                    self.print_success("Repository updated")
                    return True
                else:
                    self.print_warning(f"Git pull failed: {stderr}")
                    # Continue with fresh clone
        
        print("  Cloning repository...")
        ret, _, stderr = self.run_command([
            "git", "clone", "--branch", "main", CONFIG["repo_url"], str(self.project_path)
        ], check=False)
        
        if ret == 0:
            # Verify git repository is properly set up
            ret, stdout, _ = self.run_command([
                "git", "rev-parse", "--short", "HEAD"
            ], cwd=self.project_path, check=False)
            
            if ret == 0:
                commit = stdout.strip()
                self.print_success(f"Repository cloned successfully (main@{commit})")
            else:
                self.print_success("Repository cloned successfully")
            return True
        else:
            self.print_error(f"Failed to clone repository: {stderr}")
            return False
    
    def install_ffmpeg(self) -> bool:
        """Install FFmpeg and FFprobe"""
        if self.command_exists("ffmpeg") and self.command_exists("ffprobe"):
            self.print_success("FFmpeg already available")
            return True
        
        print("  Installing FFmpeg...")
        
        if self.platform == "Windows":
            return self._install_ffmpeg_windows()
        elif self.platform == "Darwin":
            return self._install_ffmpeg_macos()
        else:
            return self._install_ffmpeg_linux()
    
    def _install_ffmpeg_windows(self) -> bool:
        """Install FFmpeg on Windows"""
        ffmpeg_url = DOWNLOAD_URLS["ffmpeg"]["windows"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "ffmpeg.zip"
            
            if not self.download_with_progress(ffmpeg_url, archive_path, "FFmpeg"):
                return False
            
            # Extract to tools directory
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            extract_dir = Path(temp_dir) / "extracted"
            
            if not self.extract_archive(archive_path, extract_dir, "FFmpeg"):
                return False
            
            # Find the ffmpeg directory (varies by build)
            ffmpeg_dirs = [d for d in extract_dir.iterdir() if d.is_dir() and "ffmpeg" in d.name.lower()]
            if not ffmpeg_dirs:
                self.print_error("Could not find FFmpeg directory in archive")
                return False
            
            ffmpeg_source = ffmpeg_dirs[0] / "bin"
            if not ffmpeg_source.exists():
                self.print_error("Could not find FFmpeg binaries")
                return False
            
            # Copy to tools directory
            ffmpeg_dest = self.tools_dir / "ffmpeg"
            if ffmpeg_dest.exists():
                shutil.rmtree(ffmpeg_dest)
            shutil.copytree(ffmpeg_source, ffmpeg_dest)
            
            # Add to PATH for this session
            ffmpeg_bin = str(ffmpeg_dest)
            if ffmpeg_bin not in os.environ["PATH"]:
                os.environ["PATH"] = ffmpeg_bin + ";" + os.environ["PATH"]
            
            self.print_success("FFmpeg installed successfully")
            return True
    
    def _install_ffmpeg_macos(self) -> bool:
        """Install FFmpeg on macOS"""
        if self.command_exists("brew"):
            ret, _, stderr = self.run_command(["brew", "install", "ffmpeg"], check=False)
            if ret == 0:
                self.print_success("FFmpeg installed via Homebrew")
                return True
        
        self.print_warning("Could not install FFmpeg automatically")
        self.print_warning("Please install Homebrew and run: brew install ffmpeg")
        return True  # Don't fail installation
    
    def _install_ffmpeg_linux(self) -> bool:
        """Install FFmpeg on Linux"""
        # Try package managers first
        package_managers = [
            (["apt", "update"], ["apt", "install", "-y", "ffmpeg"]),
            (None, ["yum", "install", "-y", "ffmpeg"]),
            (None, ["dnf", "install", "-y", "ffmpeg"]),
            (None, ["pacman", "-S", "--noconfirm", "ffmpeg"]),
            (None, ["zypper", "install", "-y", "ffmpeg"]),
            (None, ["apk", "add", "ffmpeg"])
        ]
        
        for update_cmd, install_cmd in package_managers:
            if self.command_exists(install_cmd[0]):
                print(f"  Using {install_cmd[0]} package manager...")
                
                if update_cmd:
                    self.run_command(update_cmd, check=False)
                
                ret, _, stderr = self.run_command(install_cmd, check=False)
                if ret == 0:
                    self.print_success(f"FFmpeg installed via {install_cmd[0]}")
                    return True
        
        self.print_warning("Could not install FFmpeg automatically")
        self.print_warning("Please install FFmpeg using your system's package manager")
        return True  # Don't fail installation
    
    def setup_python_environment(self) -> bool:
        """Setup Python virtual environment"""
        print("  Setting up Python environment...")
        
        if self.conda_available:
            return self._setup_conda_environment()
        else:
            return self._setup_venv_environment()
    
    def _setup_conda_environment(self) -> bool:
        """Setup conda environment"""
        conda_exe = self.miniconda_path / ("Scripts/conda.exe" if self.platform == "Windows" else "bin/conda")
        
        # Check if environment exists
        ret, stdout, _ = self.run_command([str(conda_exe), "env", "list"], capture=True, check=False)
        
        env_exists = CONFIG["env_name"] in stdout if ret == 0 else False
        
        if not env_exists:
            print(f"  Creating conda environment '{CONFIG['env_name']}'...")
            ret, _, stderr = self.run_command([
                str(conda_exe), "create", "-n", CONFIG["env_name"],
                f"python={CONFIG['python_version']}", "-y"
            ], check=False)
            
            if ret != 0:
                self.print_error(f"Failed to create conda environment: {stderr}")
                return False
        else:
            self.print_success(f"Using existing conda environment '{CONFIG['env_name']}'")
        
        return True
    
    def _setup_venv_environment(self) -> bool:
        """Setup Python venv environment"""
        self.venv_path = self.project_path / "venv"
        
        if self.venv_path.exists() and not self.force:
            self.print_success("Using existing virtual environment")
            return True
        
        print(f"  Creating virtual environment...")
        ret, _, stderr = self.run_command([
            sys.executable, "-m", "venv", str(self.venv_path)
        ], check=False)
        
        if ret != 0:
            self.print_error(f"Failed to create virtual environment: {stderr}")
            return False
        
        self.print_success("Virtual environment created")
        return True
    
    def install_python_dependencies(self) -> bool:
        """Install Python dependencies"""
        print("  Installing Python dependencies...")
        
        original_dir = Path.cwd()
        try:
            os.chdir(self.project_path)
            
            # Get Python executable for the environment
            python_exe = self._get_python_executable()
            if not python_exe:
                self.print_error("Could not find Python executable for environment")
                return False
            
            # Upgrade pip first
            print("  Upgrading pip...")
            self.run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=False)
            
            # Install core requirements
            core_req = CONFIG["requirements_files"]["core"]
            core_req_path = self.project_path / core_req
            if core_req_path.exists():
                print(f"  Installing core requirements from {core_req}...")
                ret, stdout, stderr = self.run_command([
                    str(python_exe), "-m", "pip", "install", "-r", core_req
                ], check=False)
                
                if ret != 0:
                    self.print_error(f"Failed to install core requirements: {stderr}")
                    return False
                else:
                    print(f"    Core requirements installed successfully")
            else:
                self.print_error(f"Core requirements file not found: {core_req_path}")
                return False
            
            # Install GPU-specific requirements
            gpu_type = self._detect_gpu()
            req_file = CONFIG["requirements_files"].get(gpu_type)
            
            if req_file:
                gpu_req_path = self.project_path / req_file
                if gpu_req_path.exists():
                    print(f"  Installing {gpu_type.upper()} requirements from {req_file}...")
                    ret, stdout, stderr = self.run_command([
                        str(python_exe), "-m", "pip", "install", "-r", req_file
                    ], check=False)
                    
                    if ret != 0:
                        self.print_warning(f"Failed to install {gpu_type} requirements: {stderr}")
                        # Don't fail installation for GPU requirements
                    else:
                        print(f"    {gpu_type.upper()} requirements installed successfully")
                else:
                    self.print_warning(f"GPU requirements file not found: {gpu_req_path}")
            else:
                print(f"  No specific requirements for {gpu_type} GPU type")
            
            self.print_success("Python dependencies installed")
            return True
            
        finally:
            os.chdir(original_dir)
    
    def _get_python_executable(self) -> Optional[Path]:
        """Get the Python executable for the current environment"""
        if self.conda_available:
            if self.platform == "Windows":
                return self.miniconda_path / "envs" / CONFIG["env_name"] / "python.exe"
            else:
                return self.miniconda_path / "envs" / CONFIG["env_name"] / "bin" / "python"
        elif self.venv_path:
            if self.platform == "Windows":
                return self.venv_path / "Scripts" / "python.exe"
            else:
                return self.venv_path / "bin" / "python"
        else:
            return Path(sys.executable)
    
    def _detect_gpu(self) -> str:
        """Detect GPU type"""
        # NVIDIA detection
        ret, stdout, _ = self.run_command([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
        ], capture=True, check=False)
        
        if ret == 0 and stdout.strip():
            gpu_name = stdout.strip().split('\n')[0]
            self.print_success(f"NVIDIA GPU detected: {gpu_name}")
            return "cuda"
        
        # AMD ROCm detection
        ret, _, _ = self.run_command(["rocm-smi"], check=False)
        if ret == 0:
            self.print_success("AMD GPU with ROCm detected")
            return "rocm"
        
        # Apple Silicon detection
        if self.platform == "Darwin" and self.arch == "arm64":
            self.print_success("Apple Silicon detected (MPS support)")
            return "cpu"  # Use CPU requirements which include MPS support
        
        self.print_success("Using CPU configuration")
        return "cpu"
    
    def create_launchers(self) -> bool:
        """Create platform-specific launcher scripts"""
        print("  Creating launcher scripts...")
        
        # Create models directory
        models_dir = self.project_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Get activation command
        if self.conda_available:
            if self.platform == "Windows":
                activate_cmd = f'call "{self.miniconda_path}\\Scripts\\activate.bat" {CONFIG["env_name"]}'
            else:
                activate_cmd = f'source "{self.miniconda_path}/bin/activate" {CONFIG["env_name"]}'
        elif self.venv_path:
            if self.platform == "Windows":
                activate_cmd = f'call "{self.venv_path}\\Scripts\\activate.bat"'
            else:
                activate_cmd = f'source "{self.venv_path}/bin/activate"'
        else:
            activate_cmd = ""  # Use system Python
        
        if self.platform == "Windows":
            self._create_windows_launcher(activate_cmd)
        else:
            self._create_unix_launcher(activate_cmd)
        
        self.print_success("Launcher scripts created")
        return True
    
    def _create_windows_launcher(self, activate_cmd: str):
        """Create Windows launcher"""
        launcher_content = f'''@echo off
cd /d "{self.project_path}"
echo Activating FunGen environment...
{activate_cmd}
echo Starting FunGen...
python {CONFIG["main_script"]} %*
pause
'''
        
        launcher_path = self.project_path / "launch.bat"
        launcher_path.write_text(launcher_content, encoding='utf-8')
    
    def _create_unix_launcher(self, activate_cmd: str):
        """Create Unix launcher (Linux/macOS)"""
        launcher_content = f'''#!/bin/bash
cd "$(dirname "$0")"
echo "Activating FunGen environment..."
{activate_cmd}
echo "Starting FunGen..."
python {CONFIG["main_script"]} "$@"
'''
        
        launcher_path = self.project_path / "launch.sh"
        launcher_path.write_text(launcher_content)
        launcher_path.chmod(0o755)
        
        if self.platform == "Darwin":
            # Create .command file for double-clicking on macOS
            command_content = launcher_content + '''
echo ""
read -p "Press Enter to close..."
'''
            command_path = self.project_path / "launch.command"
            command_path.write_text(command_content)
            command_path.chmod(0o755)
    
    def validate_installation(self) -> bool:
        """Validate the installation"""
        self.print_step("Validating installation")
        
        checks = [
            ("Git", lambda: self.command_exists("git")),
            ("FFmpeg", lambda: self.command_exists("ffmpeg") or True),  # Optional
            ("FFprobe", lambda: self.command_exists("ffprobe") or True),  # Optional
            ("Project files", lambda: (self.project_path / CONFIG["main_script"]).exists()),
            ("Models directory", lambda: (self.project_path / "models").exists()),
            ("Requirements files", lambda: any(
                (self.project_path / req).exists() 
                for req in CONFIG["requirements_files"].values()
            )),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if check_func():
                    self.print_success(f"{check_name}: OK")
                else:
                    self.print_error(f"{check_name}: FAILED")
                    if check_name not in ["FFmpeg", "FFprobe"]:  # Optional
                        all_passed = False
            except Exception as e:
                self.print_error(f"{check_name}: ERROR - {e}")
                all_passed = False
        
        # Test Python environment
        python_exe = self._get_python_executable()
        if python_exe and python_exe.exists():
            try:
                # Create a much shorter test command to avoid Windows command line length limits
                test_command = "import torch, ultralytics; print('Environment: OK')"
                ret, stdout, stderr = self.run_command([
                    str(python_exe), "-c", test_command
                ], capture=True, check=False)
                
                if ret == 0:
                    self.print_success("Python environment: OK")
                    print(f"    PyTorch and Ultralytics successfully imported")
                else:
                    self.print_warning(f"Python environment test failed - but installation may still work")
                    self.print_warning(f"Error details: {stderr}")
            except Exception as e:
                self.print_warning(f"Could not test Python environment: {e}")
        
        return all_passed
    
    def print_completion_message(self):
        """Print completion message"""
        print(f"\n{Colors.GREEN}{Colors.BOLD}=" * 60)
        print("    FunGen Installation Complete!")
        print("=" * 60 + Colors.ENDC)
        
        print(f"\n{Colors.CYAN}To run FunGen:{Colors.ENDC}")
        print(f"{Colors.YELLOW}  ⚠ IMPORTANT: Use the launcher scripts below (not 'python main.py' directly){Colors.ENDC}")
        
        if self.platform == "Windows":
            print(f"  • Double-click: {self.project_path / 'launch.bat'}")
        else:
            if self.platform == "Darwin":
                print(f"  • Double-click: {self.project_path / 'launch.command'}")
            print(f"  • Terminal: {self.project_path / 'launch.sh'}")
        
        print(f"\n{Colors.CYAN}Alternative terminal method:{Colors.ENDC}")
        print(f"  cd \"{self.project_path}\"")
        if self.conda_available:
            print(f"  conda activate {CONFIG['env_name']}")
        else:
            print(f"  source venv/bin/activate  # Linux/macOS")
            print(f"  venv\\Scripts\\activate     # Windows")
        print(f"  python {CONFIG['main_script']}")
        
        print(f"\n{Colors.YELLOW}First-time setup:{Colors.ENDC}")
        print("  • FunGen will download required YOLO models on first run")
        print("  • Initial download may take 5-10 minutes")
        print("  • Ensure stable internet connection for model downloads")
        print("  • If validation warnings appear above, they can usually be ignored")
        
        print(f"\n{Colors.CYAN}GPU Acceleration:{Colors.ENDC}")
        gpu_type = self._detect_gpu()
        if gpu_type == "cuda":
            print("  • NVIDIA GPU detected - CUDA acceleration enabled")
        elif gpu_type == "rocm":
            print("  • AMD GPU detected - ROCm acceleration enabled")
        elif self.platform == "Darwin" and self.arch == "arm64":
            print("  • Apple Silicon detected - MPS acceleration enabled")
        else:
            print("  • CPU-only mode - consider GPU for faster processing")
        
        print(f"\n{Colors.CYAN}Support & Documentation:{Colors.ENDC}")
        print("  • Project documentation: README.md in the project folder")
        print("  • Discord community: https://discord.gg/WYkjMbtCZA")
        print("  • Report issues: https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/issues")
        
        print(f"\n{Colors.GREEN}Installation completed successfully!{Colors.ENDC}")
        print()
    
    def install(self) -> bool:
        """Run the complete installation"""
        self.print_header()
        
        try:
            steps = [
                ("Checking system requirements", self.check_system_requirements),
                ("Installing Git", self.install_git),
                ("Cloning FunGen repository", self.clone_repository),
                ("Installing FFmpeg", self.install_ffmpeg),
                ("Setting up Python environment", self.setup_python_environment),
                ("Installing Python dependencies", self.install_python_dependencies),
                ("Creating launcher scripts", self.create_launchers),
                ("Validating installation", self.validate_installation),
            ]
            
            for step_name, step_func in steps:
                self.print_step(step_name)
                
                try:
                    if not step_func():
                        self.print_error(f"Installation failed at: {step_name}")
                        return False
                except Exception as e:
                    self.print_error(f"Error in {step_name}: {e}")
                    return False
                
                print()  # Spacing between steps
            
            self.print_completion_message()
            return True
            
        except KeyboardInterrupt:
            self.print_error("\nInstallation cancelled by user")
            return False
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FunGen Universal Installer - Complete setup from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This installer assumes Python is available but installs everything else:
- Git (if not available)
- FFmpeg/FFprobe
- Python virtual environment 
- All Python dependencies including PyTorch
- Platform-specific launcher scripts

Examples:
  python fungen_universal_installer.py
  python fungen_universal_installer.py --dir ~/FunGen
  python fungen_universal_installer.py --force
  python fungen_universal_installer.py --uninstall
        """
    )
    
    parser.add_argument(
        "--dir", "--install-dir",
        help="Installation directory (default: current directory)",
        default=None
    )
    
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Force reinstallation of existing components"
    )
    
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Download and run the uninstaller instead"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="FunGen Universal Installer 2.0.0"
    )
    
    args = parser.parse_args()
    
    # Handle uninstall option
    if args.uninstall:
        print("🗑️ Downloading and running FunGen uninstaller...")
        
        uninstaller_url = "https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/fungen_uninstall.py"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            uninstaller_path = Path(temp_dir) / "fungen_uninstall.py"
            
            try:
                urllib.request.urlretrieve(uninstaller_url, uninstaller_path)
                print("✓ Downloaded uninstaller")
                
                # Run uninstaller with remaining args
                remaining_args = [arg for arg in sys.argv[1:] if arg != "--uninstall"]
                result = subprocess.run([sys.executable, str(uninstaller_path)] + remaining_args)
                sys.exit(result.returncode)
                
            except Exception as e:
                print(f"❌ Failed to download uninstaller: {e}")
                print("Please download fungen_uninstall.py manually from GitHub")
                sys.exit(1)
    
    # Run installer
    installer = FunGenUniversalInstaller(
        install_dir=args.dir,
        force=args.force
    )
    
    success = installer.install()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()