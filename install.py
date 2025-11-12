#!/usr/bin/env python3
"""
FunGen Universal Installer - Stage 2
Version: 1.2.0
Complete installation system that assumes Python is available but nothing else

This installer handles the complete FunGen setup after Python is installed:
- Git installation and repository cloning
- FFmpeg suite installation (ffmpeg, ffprobe, ffplay)
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

# Version information
INSTALLER_VERSION = "1.3.5"

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
    
    def __init__(self, install_dir: Optional[str] = None, force: bool = False, bootstrap_version: Optional[str] = None):
        self.platform = platform.system()
        self.arch = platform.machine().lower()
        self.force = force
        self.bootstrap_version = bootstrap_version
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
        print(f"              v{INSTALLER_VERSION}")
        if self.bootstrap_version:
            print(f"         (Bootstrap v{self.bootstrap_version})")
        print("=" * 60 + Colors.ENDC)
        print(f"{Colors.CYAN}Platform: {self.platform} ({self.arch})")
        print(f"Install Directory: {self.install_dir}")
        print(f"Project Path: {self.project_path}{Colors.ENDC}")
        
        # Add interactive warning for macOS/Linux
        if self.platform in ["Darwin", "Linux"]:
            print(f"\n{Colors.YELLOW}⚠️  INTERACTIVE INSTALLATION NOTICE:")
            print("   Some system installations may require your interaction:")
            print("   • Password prompts for system package installation")
            print("   • License agreement acceptance (Xcode Command Line Tools)")
            print("   • Package manager confirmations")
            print(f"   Please stay near your computer during installation.{Colors.ENDC}")
        
        print()
    
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

            # Check if conda Python is wrong architecture on macOS
            if self.platform == "Darwin" and self.arch == "arm64":
                conda_python = self.miniconda_path / "bin" / "python"
                if conda_python.exists():
                    try:
                        result = subprocess.run(['file', str(conda_python)],
                                              capture_output=True, text=True, timeout=5)
                        if 'x86_64' in result.stdout:
                            self.print_warning("╔" + "="*70 + "╗")
                            self.print_warning("║  WARNING: x86_64 (Intel) Miniconda detected on Apple Silicon!      ║")
                            self.print_warning("║  This will cause:                                                   ║")
                            self.print_warning("║    • Slower performance (running under Rosetta 2)                   ║")
                            self.print_warning("║    • CoreML model conversion will NOT work                          ║")
                            self.print_warning("║    • No GPU acceleration via Metal Performance Shaders (MPS)        ║")
                            self.print_warning("║                                                                     ║")
                            self.print_warning("║  Recommended: Delete ~/miniconda3 and rerun installer               ║")
                            self.print_warning("║    rm -rf ~/miniconda3                                              ║")
                            self.print_warning("║    curl -fsSL https://raw.githubusercontent.com/.../install.sh | bash ║")
                            self.print_warning("╚" + "="*70 + "╝")

                            # Give user option to abort
                            if not self.force:
                                response = input("\n  Continue anyway? [y/N]: ").strip().lower()
                                if response != 'y':
                                    print("\n  Installation aborted. Please reinstall with ARM64 Miniconda.")
                                    return False
                    except Exception:
                        pass
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
            
            # Install silently with user-only installation (no admin required)
            ret, _, stderr = self.run_command([
                str(installer_path), "/VERYSILENT", "/NORESTART", "/NOCANCEL",
                "/SP-", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", 
                "/CURRENTUSER"  # Install for current user only
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
            print("  Installing Git via Homebrew...")
            print("  Note: This may prompt for your password or Xcode license acceptance")
            ret, _, stderr = self.run_command(["brew", "install", "git"], check=False)
            if ret == 0:
                self.print_success("Git installed via Homebrew")
                return True
            else:
                print(f"  Homebrew install failed: {stderr}")
        
        # Try to install Xcode Command Line Tools
        print("  Installing Xcode Command Line Tools (includes Git)...")
        print("  This will open a dialog - please accept the license agreement")
        ret, _, _ = self.run_command(["xcode-select", "--install"], check=False)
        
        if ret == 0:
            print("  ⚠️  INTERACTIVE STEP REQUIRED:")
            print("     A dialog has opened for Xcode Command Line Tools installation")
            print("     Please accept the license agreement and complete the installation")
            print("     This may take several minutes to download and install")
            print("  ")
            input("  Press Enter when the installation is complete...")
            
            # Verify installation
            if self.command_exists("git"):
                self.print_success("Git installed via Xcode Command Line Tools")
                return True
            else:
                self.print_error("Git installation verification failed")
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
                print("  Note: You may be prompted for your password or to accept terms")
                
                if update_cmd:
                    print("  Updating package lists...")
                    self.run_command(update_cmd, check=False)
                
                print(f"  Installing Git with {install_cmd[0]}...")
                ret, _, stderr = self.run_command(install_cmd, check=False)
                if ret == 0:
                    self.print_success(f"Git installed via {install_cmd[0]}")
                    return True
                else:
                    self.print_warning(f"Failed to install with {install_cmd[0]}: {stderr}")
                    # If interactive prompts were missed, suggest manual installation
                    if "interactive" in stderr.lower() or "prompt" in stderr.lower():
                        print("  ⚠️  This package manager may require interactive input")
                        print(f"     Try running manually: sudo {' '.join(install_cmd)}")
                        return False
        
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
            # Configure git safe.directory to prevent permission issues
            self.run_command([
                "git", "config", "--add", "safe.directory", str(self.project_path)
            ], cwd=self.project_path, check=False)
            
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
        """Install FFmpeg, FFprobe, and FFplay"""
        if self.command_exists("ffmpeg") and self.command_exists("ffprobe") and self.command_exists("ffplay"):
            self.print_success("FFmpeg suite already available")
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
            
            # Verify all FFmpeg tools are available
            if self.command_exists("ffmpeg") and self.command_exists("ffprobe") and self.command_exists("ffplay"):
                self.print_success("FFmpeg suite installed successfully")
                return True
            else:
                self.print_error("FFmpeg installation incomplete - some tools missing")
                return False
    
    def _install_ffmpeg_macos(self) -> bool:
        """Install FFmpeg on macOS"""
        if self.command_exists("brew"):
            ret, _, stderr = self.run_command(["brew", "install", "ffmpeg"], check=False)
            if ret == 0:
                # Verify all FFmpeg tools are available
                if self.command_exists("ffmpeg") and self.command_exists("ffprobe") and self.command_exists("ffplay"):
                    self.print_success("FFmpeg suite installed via Homebrew")
                    return True
                else:
                    self.print_error("FFmpeg installation incomplete - some tools missing")
                    return False
        
        self.print_warning("Could not install FFmpeg automatically")
        self.print_warning("Please install Homebrew and run: brew install ffmpeg")
        self.print_warning("FFmpeg suite (including ffplay) is required for fullscreen functionality")
        return False  # Fail installation if FFmpeg not available
    
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
                    # Verify all FFmpeg tools are available
                    if self.command_exists("ffmpeg") and self.command_exists("ffprobe") and self.command_exists("ffplay"):
                        self.print_success(f"FFmpeg suite installed via {install_cmd[0]}")
                        return True
                    else:
                        self.print_error("FFmpeg installation incomplete - some tools missing")
                        continue  # Try next package manager
        
        self.print_warning("Could not install FFmpeg automatically")
        self.print_warning("Please install FFmpeg using your system's package manager")
        self.print_warning("FFmpeg suite (including ffplay) is required for fullscreen functionality")
        return False  # Fail installation if FFmpeg not available
    
    def _check_arm64_windows_compatibility(self):
        """Check for ARM64 Windows and provide guidance."""
        if platform.system() == "Windows" and platform.machine().lower() in ['arm64', 'aarch64']:
            python_arch = platform.architecture()[0]
            
            self.print_warning("ARM64 Windows detected!")
            self.print_warning("")
            self.print_warning("ARM64 Windows has limited Python package support.")
            self.print_warning("Many packages (including imgui) cannot compile on ARM64.")
            self.print_warning("")
            
            if "arm" in platform.platform().lower() or "arm64" in python_arch.lower():
                self.print_warning("RECOMMENDED SOLUTION:")
                self.print_warning("1. Uninstall current Python (if ARM64)")
                self.print_warning("2. Download Python 3.11 x64 from python.org")
                self.print_warning("3. Install x64 Python (will run via emulation)")
                self.print_warning("4. Rerun this installer")
                self.print_warning("")
                self.print_warning("ALTERNATIVE: Use Windows Subsystem for Linux (WSL)")
                self.print_warning("- Run: wsl --install")
                self.print_warning("- Install Ubuntu and run FunGen in Linux")
                self.print_warning("")
                
                response = input("Continue anyway? (not recommended) [y/N]: ").lower()
                if response != 'y':
                    self.print_error("Installation cancelled. Please install x64 Python first.")
                    return False
                    
        return True

    def _handle_imgui_installation_failure(self):
        """Handle imgui installation failure with appropriate guidance."""
        self.print_error("All imgui installation strategies failed.")
        self.print_error("This means no precompiled wheels are available and compilation failed.")
        self.print_error("")
        
        # ARM64-specific guidance
        if platform.machine().lower() in ['arm64', 'aarch64']:
            self.print_error("ARM64 WINDOWS DETECTED:")
            self.print_error("imgui does not compile on ARM64 Windows.")
            self.print_error("")
            self.print_error("RECOMMENDED SOLUTION:")
            self.print_error("1. Uninstall current ARM64 Python")
            self.print_error("2. Download Python 3.11 x64 from python.org")
            self.print_error("3. Install x64 Python (runs via emulation)")
            self.print_error("4. Rerun this installer")
            self.print_error("")
            self.print_error("ALTERNATIVE: Use WSL2 Ubuntu")
        else:
            self.print_error("SOLUTION OPTIONS:")
            self.print_error("1. EASIEST: Install Microsoft Visual C++ Build Tools:")
            self.print_error("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            self.print_error("   - Download 'Build Tools for Visual Studio 2022'")
            self.print_error("   - Select 'C++ build tools' workload")
            self.print_error("   - Restart computer after installation")
            self.print_error("")
            self.print_error("2. Install Visual Studio Community (includes build tools)")
            self.print_error("3. Use Windows Subsystem for Linux (WSL2)")
        
        self.print_error("")
        self.print_error("⚠️  WITHOUT IMGUI, FUNGEN CANNOT DISPLAY ITS GUI!")
        self.print_error("The installation will continue, but FunGen won't work until this is fixed.")
        print("  Core requirements installed (GUI unavailable)")

    def setup_python_environment(self) -> bool:
        """Setup Python virtual environment"""
        print("  Setting up Python environment...")
        
        # Check ARM64 compatibility before proceeding
        if not self._check_arm64_windows_compatibility():
            return False
        
        if self.conda_available:
            if self._setup_conda_environment():
                return True
            else:
                print("  Conda environment setup failed, trying Python venv as fallback...")
                self.conda_available = False  # Switch to venv mode
                return self._setup_venv_environment()
        else:
            return self._setup_venv_environment()
    
    def _setup_conda_environment(self) -> bool:
        """Setup conda environment"""
        conda_exe = self.miniconda_path / ("condabin/conda.bat" if self.platform == "Windows" else "bin/conda")

        # Accept conda Terms of Service if not already accepted
        print("  Accepting conda Terms of Service...")
        channels = [
            "https://repo.anaconda.com/pkgs/main",
            "https://repo.anaconda.com/pkgs/r"
        ]
        for channel in channels:
            ret, stdout, stderr = self.run_command([
                str(conda_exe), "tos", "accept", "--override-channels", "--channel", channel
            ], capture=True, check=False)
            # Continue even if this fails - may already be accepted

        # Check if environment exists
        ret, stdout, _ = self.run_command([str(conda_exe), "env", "list"], capture=True, check=False)

        env_exists = CONFIG["env_name"] in stdout if ret == 0 else False

        if not env_exists:
            print(f"  Creating conda environment '{CONFIG['env_name']}'...")
            print(f"  Using conda at: {conda_exe}")
            print(f"  Command: conda create -n {CONFIG['env_name']} python={CONFIG['python_version']} -y")

            ret, stdout, stderr = self.run_command([
                str(conda_exe), "create", "-n", CONFIG["env_name"],
                f"python={CONFIG['python_version']}", "-y"
            ], check=False)

            if ret != 0:
                self.print_error(f"Failed to create conda environment")
                self.print_error(f"Error details: {stderr}")
                if stdout:
                    self.print_error(f"Output: {stdout}")

                # Common solutions
                print("\nPossible solutions:")
                print("1. Try running manually:")
                print(f"   conda create -n {CONFIG['env_name']} python={CONFIG['python_version']} -y")
                print("2. Check if conda is properly initialized:")
                print("   conda init")
                print("3. Try using system Python instead (rerun installer)")
                print("4. Check available conda channels:")
                print("   conda info")

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
        print(f"  Using Python: {sys.executable}")
        print(f"  Target path: {self.venv_path}")
        
        ret, _, stderr = self.run_command([
            sys.executable, "-m", "venv", str(self.venv_path)
        ], check=False)
        
        if ret != 0:
            self.print_error(f"Failed to create virtual environment")
            self.print_error(f"Error: {stderr}")
            print("\nPossible solutions:")
            print("1. Check if Python venv module is available:")
            print(f"   {sys.executable} -m venv --help")
            print("2. Try installing python3-venv (Linux):")
            print("   sudo apt install python3-venv")
            print("3. Use system Python directly (not recommended)")
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
                # On Windows, try to avoid imgui compilation issues by installing packages individually first
                if platform.system() == "Windows":
                    print(f"  Installing core requirements (Windows optimized approach)...")
                    print("  Installing packages individually to avoid compilation issues...")
                    
                    # Install all packages except imgui first
                    non_imgui_packages = [
                        "numpy", "ultralytics==8.3.78", "glfw~=2.8.0", "pyopengl~=3.1.7",
                        "imageio~=2.36.1", "tqdm~=4.67.1", "colorama~=0.4.6", 
                        "opencv-python~=4.10.0.84", "scipy~=1.15.1", "simplification~=0.7.13",
                        "msgpack~=1.1.0", "pillow~=11.1.0", "orjson~=3.10.15", 
                        "send2trash~=1.8.3", "aiosqlite"
                    ]
                    
                    ret, stdout, stderr = self.run_command([
                        str(python_exe), "-m", "pip", "install"
                    ] + non_imgui_packages, check=False)
                    
                    if ret != 0:
                        self.print_error(f"Failed to install core packages: {stderr}")
                        return False
                    
                    # Now try to install imgui separately with multiple fallback strategies
                    print("  Attempting to install imgui...")
                    
                    # Strategy 1: Force precompiled wheels first (avoid compilation)
                    print("  Trying imgui with precompiled wheels only...")
                    ret_imgui, stdout_imgui, stderr_imgui = self.run_command([
                        str(python_exe), "-m", "pip", "install", "imgui", 
                        "--only-binary=all", "--prefer-binary"
                    ], check=False)
                    
                    if ret_imgui != 0:
                        self.print_warning("Precompiled wheels not available. Trying alternative strategies...")
                        
                        # Strategy 2: Try with fresh cache (sometimes wheels are corrupted)
                        print("  Trying imgui with fresh cache...")
                        ret_imgui2, stdout_imgui2, stderr_imgui2 = self.run_command([
                            str(python_exe), "-m", "pip", "install", "imgui", 
                            "--no-cache-dir", "--only-binary=all"
                        ], check=False)
                        
                        if ret_imgui2 != 0:
                            # Strategy 3: Try imgui 2.0.0 specifically (most likely to have wheels)
                            print("  Trying imgui 2.0.0 (most likely to have precompiled wheels)...")
                            ret_imgui3, stdout_imgui3, stderr_imgui3 = self.run_command([
                                str(python_exe), "-m", "pip", "install", "imgui==2.0.0",
                                "--only-binary=all", "--prefer-binary"
                            ], check=False)
                            
                            if ret_imgui3 == 0:
                                print("  Core requirements installed successfully (imgui 2.0.0)")
                                imgui_installed = True
                            else:
                                imgui_installed = False
                            
                            if not imgui_installed:
                                # Strategy 4: Last resort - allow compilation but with better error handling
                                print("  Last resort: attempting compilation...")
                                ret_imgui4, stdout_imgui4, stderr_imgui4 = self.run_command([
                                    str(python_exe), "-m", "pip", "install", "imgui", "--no-cache-dir"
                                ], check=False)
                                
                                if ret_imgui4 == 0:
                                    print("  Core requirements installed successfully (imgui compiled)")
                                else:
                                    # All strategies failed - provide guidance
                                    self._handle_imgui_installation_failure()
                        else:
                            print("  Core requirements installed successfully (imgui retry)")
                    else:
                        print("  Core requirements installed successfully (including imgui)")
                        
                else:
                    # Non-Windows: use regular requirements file installation
                    print(f"  Installing core requirements from {core_req}...")
                    ret, stdout, stderr = self.run_command([
                        str(python_exe), "-m", "pip", "install", "-r", core_req
                    ], check=False)
                    
                    if platform.system() != "Windows" and ret != 0:
                        self.print_error(f"Failed to install core requirements: {stderr}")
                        return False
                    elif platform.system() != "Windows":
                        print(f"    Core requirements installed successfully")
            else:
                self.print_error(f"Core requirements file not found: {core_req_path}")
                return False

            # macOS-specific: Use conda for PyTorch to get newer versions (2.5+) with NumPy 2.x support
            # pip only has PyTorch 2.2.2 for macOS x86_64, which requires NumPy 1.x
            pytorch_via_conda = False  # Track if we installed PyTorch via conda
            if self.platform == "Darwin" and self.conda_available:
                print("  Checking PyTorch version for macOS compatibility...")

                # Check installed PyTorch version
                ret, stdout, stderr = self.run_command([
                    str(python_exe), "-c", "import torch; print(torch.__version__)"
                ], capture=True, check=False)

                if ret == 0:
                    torch_version = stdout.strip().split('+')[0]  # Remove +cpu suffix if present
                    major, minor = map(int, torch_version.split('.')[:2])

                    # If PyTorch < 2.4, it was compiled with NumPy 1.x and won't work with NumPy 2.x
                    if major == 2 and minor < 4:
                        print(f"  Detected PyTorch {torch_version} (requires NumPy 1.x)")
                        print("  Upgrading to conda PyTorch for NumPy 2.x support...")

                        # Get conda executable
                        conda_exe = self.miniconda_path / ("Scripts/conda.exe" if self.platform == "Windows" else "bin/conda")

                        # Uninstall pip-installed torch, torchvision, numpy, and scipy
                        # These will be reinstalled via conda for binary compatibility
                        print("  Uninstalling pip PyTorch, numpy, and scipy packages...")
                        self.run_command([
                            str(python_exe), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "numpy", "scipy"
                        ], check=False)

                        # Install via conda from pkgs/main channel with numpy 2.1 for compatibility
                        # Installing pytorch, torchvision, and numpy together ensures compatible versions
                        # pkgs/main has PyTorch 2.5.1 + torchvision 0.20.1 which are compatible
                        print("  Installing PyTorch, torchvision, and numpy 2.1 via conda...")
                        ret, stdout, stderr = self.run_command([
                            str(conda_exe), "install", "-n", CONFIG["env_name"],
                            "pytorch", "torchvision", "numpy=2.1", "-y"
                        ], check=False)

                        if ret == 0:
                            # Verify new version
                            ret2, stdout2, _ = self.run_command([
                                str(python_exe), "-c", "import torch; print(torch.__version__)"
                            ], capture=True, check=False)

                            if ret2 == 0:
                                new_version = stdout2.strip()
                                print(f"  PyTorch upgraded to {new_version} via conda")

                                # Verify torchvision version
                                ret3, stdout3, _ = self.run_command([
                                    str(python_exe), "-c", "import torchvision; print(torchvision.__version__)"
                                ], capture=True, check=False)

                                if ret3 == 0:
                                    tv_version = stdout3.strip()
                                    print(f"  torchvision {tv_version} installed via conda")

                                # Install scipy via conda for binary compatibility with conda numpy
                                print("  Installing scipy via conda for NumPy compatibility...")
                                self.run_command([
                                    str(conda_exe), "install", "-n", CONFIG["env_name"],
                                    "scipy", "-y"
                                ], check=False)

                                self.print_success(f"PyTorch {new_version} + torchvision {tv_version} installed via conda")
                                pytorch_via_conda = True  # Mark that we installed via conda
                            else:
                                self.print_success("PyTorch installed via conda")
                                pytorch_via_conda = True  # Mark that we installed via conda
                        else:
                            self.print_warning(f"Failed to install PyTorch via conda: {stderr}")
                            self.print_warning("Continuing with pip PyTorch (may have NumPy compatibility issues)")
                    else:
                        print(f"  PyTorch {torch_version} is compatible with NumPy 2.x")

            # Install GPU-specific requirements
            # Skip on macOS if we already installed PyTorch via conda
            gpu_type = self._detect_gpu()
            req_file = CONFIG["requirements_files"].get(gpu_type)

            if req_file and not pytorch_via_conda:
                gpu_req_path = self.project_path / req_file
                if gpu_req_path.exists():
                    print(f"  Installing {gpu_type.upper()} requirements from {req_file}...")
                    ret, stdout, stderr = self.run_command([
                        str(python_exe), "-m", "pip", "install", "-r", req_file
                    ], check=False)
                    
                    if ret != 0:
                        # Special handling for ROCm on Windows
                        if gpu_type == "rocm" and self.platform == "Windows":
                            if not self._handle_rocm_fallback(python_exe, stderr):
                                return False
                            # Successfully fell back to CPU
                            gpu_type = "cpu"  # Update gpu_type for later steps
                        else:
                            # Try with precompiled wheels for GPU requirements too
                            error_text = stderr + stdout  # Check both stderr and stdout
                            compilation_errors = [
                                "Microsoft Visual C++",
                                "failed-wheel-build", 
                                "error: Microsoft Visual C++",
                                "Building wheel",
                                "Failed building wheel",
                                "build dependencies",
                                "error: subprocess-exited-with-error"
                            ]
                            
                            if any(error in error_text for error in compilation_errors):
                                self.print_warning("Compilation error in GPU requirements. Trying precompiled wheels...")
                                ret2, stdout2, stderr2 = self.run_command([
                                    str(python_exe), "-m", "pip", "install", "-r", req_file,
                                    "--only-binary=all", "--prefer-binary"
                                ], check=False)
                                
                                if ret2 != 0:
                                    self.print_warning(f"Failed to install {gpu_type} requirements even with precompiled wheels")
                                    self.print_warning("GPU acceleration may not work properly. Consider installing Visual C++ Build Tools.")
                                else:
                                    print(f"    {gpu_type.upper()} requirements installed successfully (precompiled)")
                            else:
                                self.print_warning(f"Failed to install {gpu_type} requirements: {stderr}")
                                # Don't fail installation for GPU requirements
                    else:
                        print(f"    {gpu_type.upper()} requirements installed successfully")
                else:
                    self.print_warning(f"GPU requirements file not found: {gpu_req_path}")
            elif pytorch_via_conda:
                print(f"  Skipping {gpu_type.upper()} requirements (PyTorch already installed via conda)")
            else:
                print(f"  No specific requirements for {gpu_type} GPU type")

            # Install torch-tensorrt and tensorrt with version constraints to prevent PyTorch upgrade
            if gpu_type == "cuda":
                cuda_version = None
                torch_version = "2.8.0"  # Constrain to installed version

                if req_file == "cuda.requirements.txt":
                    cuda_version = "cu128"
                elif req_file == "cuda.50series.requirements.txt":
                    cuda_version = "cu129"

                if cuda_version:
                    nightly_index_url = f"https://download.pytorch.org/whl/nightly/{cuda_version}"
                    print(f"  Installing torch-tensorrt and tensorrt for {cuda_version} from nightly index...")
                    print(f"  Constraining torch to version {torch_version} to prevent upgrade...")

                    # Install with version constraint to prevent torch from upgrading to 2.9.0
                    ret, stdout, stderr = self.run_command([
                        str(python_exe), "-m", "pip", "install",
                        "torch-tensorrt", "tensorrt",
                        f"torch=={torch_version}+{cuda_version}",  # Constrain torch version
                        "--extra-index-url", nightly_index_url
                    ], check=False)

                    if ret != 0:
                        self.print_warning(f"Failed to install torch-tensorrt and tensorrt for {cuda_version}: {stderr}")
                        self.print_warning("TensorRT acceleration may not work properly.")
                        self.print_warning("FunGen will fall back to standard PyTorch inference.")
                    else:
                        print(f"    torch-tensorrt and tensorrt for {cuda_version} installed successfully")
                        print(f"    torch constrained to {torch_version}+{cuda_version}")
            
            # Install device_control requirements if available (supporter feature)
            device_control_req_path = self.project_path / "device_control" / "requirements.txt"
            if device_control_req_path.exists():
                print("  Installing device control requirements (supporter feature)...")
                ret, stdout, stderr = self.run_command([
                    str(python_exe), "-m", "pip", "install", "-r", str(device_control_req_path)
                ], check=False)
                
                if ret != 0:
                    # Try with precompiled wheels for device control too
                    error_text = stderr + stdout
                    compilation_errors = [
                        "Microsoft Visual C++",
                        "failed-wheel-build", 
                        "error: Microsoft Visual C++",
                        "Building wheel",
                        "Failed building wheel",
                        "build dependencies",
                        "error: subprocess-exited-with-error"
                    ]
                    
                    if any(error in error_text for error in compilation_errors):
                        self.print_warning("Compilation error in device control requirements. Trying precompiled wheels...")
                        ret2, stdout2, stderr2 = self.run_command([
                            str(python_exe), "-m", "pip", "install", "-r", str(device_control_req_path),
                            "--only-binary=all", "--prefer-binary"
                        ], check=False)
                        
                        if ret2 != 0:
                            self.print_warning("Failed to install device control requirements even with precompiled wheels")
                            self.print_warning("Device control features may not work properly.")
                        else:
                            print("    Device control requirements installed successfully (precompiled)")
                    else:
                        self.print_warning(f"Failed to install device control requirements: {stderr}")
                        self.print_warning("Device control features may not work properly.")
                else:
                    print("    Device control requirements installed successfully")
            else:
                print("  Device control not found - core features only")
            
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
        # Windows uses hipinfo, Linux uses rocm-smi
        if self.platform == "Windows":
            ret, stdout, _ = self.run_command(["hipinfo"], capture=True, check=False)
                # Check if we actually have AMD GPU info in output
            if "amd" in stdout.lower() or "hip" in stdout.lower():
                # Try to extract GPU name for better user feedback
                gpu_name = "AMD GPU"
                for line in stdout.split('\n'):
                    if 'name' in line.lower() or 'device' in line.lower():
                        gpu_name = line.strip()
                        break
                self.print_success(f"AMD GPU with ROCm detected: {gpu_name}")
                return "rocm"
        else:
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
    
    def _handle_rocm_fallback(self, python_exe: Path, error_msg: str) -> bool:
        """Handle ROCm installation failure with user acknowledgement"""
        print()
        self.print_warning("=" * 70)
        self.print_warning("ROCm PyTorch installation failed on Windows")
        self.print_warning("=" * 70)
        self.print_warning("Error details:")
        print(f"  {error_msg}")
        print()
        self.print_warning("This can happen because:")
        self.print_warning("  • ROCm 6 Windows support is still experimental")
        self.print_warning("  • PyTorch ROCm wheels may not be available for your Python version")
        self.print_warning("  • ROCm drivers may need updating")
        print()
        self.print_warning("Falling back to CPU-only installation will:")
        self.print_warning("  ✓ Allow FunGen to run (slower performance)")
        self.print_warning("  ✗ No GPU acceleration")
        print()
        
        # Wait for user acknowledgement
        try:
            input(f"{Colors.YELLOW}Press ENTER to continue with CPU-only installation...{Colors.ENDC}")
        except KeyboardInterrupt:
            print()
            self.print_error("Installation cancelled by user")
            return False
        
        print()
        print("  Falling back to CPU-only installation...")
        
        # Install CPU requirements instead
        cpu_req_path = self.project_path / CONFIG["requirements_files"]["cpu"]
        if cpu_req_path.exists():
            ret, stdout, stderr = self.run_command([
                str(python_exe), "-m", "pip", "install", "-r", str(cpu_req_path)
            ], check=False)
            
            if ret != 0:
                self.print_error(f"CPU fallback installation also failed: {stderr}")
                return False
            else:
                self.print_success("CPU-only requirements installed successfully")
                return True
        else:
            self.print_error(f"CPU requirements file not found: {cpu_req_path}")
            return False
    
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
        # Add tool paths to PATH
        path_additions = []

        # Add FFmpeg path if it exists
        ffmpeg_path = self.tools_dir / "ffmpeg"
        if ffmpeg_path.exists():
            path_additions.append(str(ffmpeg_path))

        # Add common Git paths if they exist
        git_paths = [
            Path.home() / "AppData" / "Local" / "Programs" / "Git" / "bin",
            Path("C:\\Program Files\\Git\\bin"),
            Path("C:\\Program Files (x86)\\Git\\bin")
        ]
        for git_path in git_paths:
            if git_path.exists():
                path_additions.append(str(git_path))
                break  # Only add the first one found

        path_setup = ""
        if path_additions:
            path_setup = f'set "PATH={";".join(path_additions)};%PATH%"\n'

        # Check if path is UNC (network share)
        project_path_str = str(self.project_path)
        is_unc = project_path_str.startswith('\\\\') or project_path_str.startswith('//')

        if is_unc:
            # For UNC paths, use pushd which maps to a temporary drive letter
            cd_command = f'pushd "{self.project_path}"'
            end_command = 'popd\n'
        else:
            # For regular paths, use cd /d
            cd_command = f'cd /d "{self.project_path}"'
            end_command = ''

        launcher_content = f'''@echo off
{cd_command}
{path_setup}echo Activating FunGen environment...
{activate_cmd}
echo Starting FunGen...
python {CONFIG["main_script"]} %*
{end_command}pause
'''

        launcher_path = self.project_path / "launch.bat"
        launcher_path.write_text(launcher_content, encoding='utf-8')
    
    def _create_unix_launcher(self, activate_cmd: str):
        """Create Unix launcher (Linux/macOS)"""
        # Add tool paths to PATH  
        path_additions = []
        
        # Add FFmpeg path if it exists
        ffmpeg_path = self.tools_dir / "ffmpeg"
        if ffmpeg_path.exists():
            path_additions.append(str(ffmpeg_path))
        
        # Add Git paths that might not be in standard PATH (mainly for Homebrew)
        git_paths = [
            Path("/usr/local/bin"),  # Homebrew on Intel macOS
            Path("/opt/homebrew/bin"),  # Homebrew on Apple Silicon macOS
        ]
        for git_path in git_paths:
            if git_path.exists() and (git_path / "git").exists():
                path_additions.append(str(git_path))
                break  # Only add the first one found
        
        path_setup = ""
        if path_additions:
            path_setup = f'export PATH="{":".join(path_additions)}:$PATH"\n'
        
        launcher_content = f'''#!/bin/bash
cd "$(dirname "$0")"
{path_setup}echo "Activating FunGen environment..."
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
            ("FFplay", lambda: self.command_exists("ffplay") or True),  # Optional
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
                    # Git, FFmpeg, FFprobe, FFplay are optional - may be in conda env only
                    if check_name in ["Git", "FFmpeg", "FFprobe", "FFplay"]:
                        self.print_warning(f"{check_name}: Not in PATH (may be in conda environment)")
                    else:
                        self.print_error(f"{check_name}: FAILED")
                        all_passed = False
            except Exception as e:
                self.print_error(f"{check_name}: ERROR - {e}")
                if check_name not in ["Git", "FFmpeg", "FFprobe", "FFplay"]:
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
        
        # Early ARM64 Windows detection
        if platform.system() == "Windows" and platform.machine().lower() in ['arm64', 'aarch64']:
            self.print_warning("⚠️  ARM64 Windows system detected!")
            self.print_warning("📦 Python packages may have limited compatibility on ARM64.")
            self.print_warning("🔧 If installation fails, consider using x64 Python instead.")
            print()
        
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
  python install.py
  python install.py --dir ~/FunGen
  python install.py --force
  python install.py --uninstall
        """
    )
    
    parser.add_argument(
        "--dir", "--install-dir",
        help="Installation directory (default: current directory)",
        default=None
    )
    
    parser.add_argument(
        "--bootstrap-version",
        help="Version of the bootstrap script (for troubleshooting)",
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
        version=f"FunGen Universal Installer {INSTALLER_VERSION}"
    )
    
    args = parser.parse_args()
    
    # Handle uninstall option
    if args.uninstall:
        print("🗑️ Downloading and running FunGen uninstaller...")
        
        uninstaller_url = "https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/uninstall.py"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            uninstaller_path = Path(temp_dir) / "uninstall.py"
            
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
        force=args.force,
        bootstrap_version=args.bootstrap_version
    )
    
    success = installer.install()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()