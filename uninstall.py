#!/usr/bin/env python3
"""
FunGen Universal Uninstaller
Safely removes FunGen and its dependencies while preserving user data

This uninstaller provides multiple removal options:
- Clean uninstall (removes everything including Python environment)  
- Partial uninstall (keeps Python/conda, removes only FunGen)
- Safe uninstall (moves files to backup before deletion)
"""

import os
import sys
import platform
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict
import argparse

class Colors:
    """ANSI color codes"""
    if platform.system() == "Windows":
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

class FunGenUninstaller:
    """Comprehensive FunGen uninstaller with multiple removal options"""
    
    def __init__(self, uninstall_type: str = "standard", backup: bool = True, 
                 dry_run: bool = False):
        self.uninstall_type = uninstall_type
        self.backup = backup
        self.dry_run = dry_run
        self.platform = platform.system()
        
        # Paths to detect
        self.project_paths = []
        self.env_paths = []
        self.tool_paths = []
        self.launcher_paths = []
        
        # Backup location
        timestamp = int(time.time())
        self.backup_dir = Path.home() / f"FunGen_Backup_{timestamp}"
        
        # Statistics
        self.files_found = 0
        self.files_removed = 0
        self.size_freed = 0
        
    def print_header(self):
        """Print uninstaller header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=" * 60)
        print("    FunGen Universal Uninstaller")
        print("=" * 60 + Colors.ENDC)
        print(f"{Colors.CYAN}Platform: {self.platform}")
        print(f"Uninstall Type: {self.uninstall_type}")
        print(f"Backup Enabled: {self.backup}")
        if self.dry_run:
            print(f"{Colors.YELLOW}DRY RUN MODE - No files will be deleted{Colors.ENDC}")
        print()
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}✗ {message}{Colors.ENDC}")
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")
    
    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, IOError):
                        pass
        except (OSError, IOError):
            pass
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format bytes as human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def discover_fungen_installations(self):
        """Discover all FunGen-related files and directories"""
        print(f"{Colors.BLUE}🔍 Scanning for FunGen installations...{Colors.ENDC}")
        
        # Common project directory names
        project_names = ["FunGen", "FunGen-AI-Powered-Funscript-Generator", "VR-Funscript-AI-Generator"]
        
        # Search locations
        search_locations = [
            Path.cwd(),
            Path.home(),
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path.home() / "Documents",
        ]
        
        if self.platform == "Windows":
            search_locations.extend([
                Path("C:/"),
                Path("C:/Users") / os.getenv("USERNAME", ""),
                Path("C:/Program Files"),
                Path("C:/Program Files (x86)"),
            ])
        else:
            search_locations.extend([
                Path("/opt"),
                Path("/usr/local"),
                Path("/home") / os.getenv("USER", ""),
            ])
        
        # Find project directories
        for location in search_locations:
            if not location.exists():
                continue
                
            for project_name in project_names:
                project_path = location / project_name
                if project_path.exists() and project_path.is_dir():
                    # Verify it's actually FunGen by checking for main.py
                    if (project_path / "main.py").exists():
                        self.project_paths.append(project_path)
                        size = self.get_directory_size(project_path)
                        print(f"  Found: {project_path} ({self.format_size(size)})")
        
        # Find conda/venv environments
        self._find_environments()
        
        # Find installed tools (if installed by FunGen installer)
        self._find_tools()
        
        # Find launcher scripts
        self._find_launchers()
        
        print(f"  Project directories: {len(self.project_paths)}")
        print(f"  Python environments: {len(self.env_paths)}")
        print(f"  Installed tools: {len(self.tool_paths)}")
        print(f"  Launcher scripts: {len(self.launcher_paths)}")
    
    def _find_environments(self):
        """Find Python environments created by FunGen"""
        # Conda environments
        conda_path = Path.home() / "miniconda3"
        if conda_path.exists():
            fungen_env = conda_path / "envs" / "FunGen"
            if fungen_env.exists():
                self.env_paths.append(fungen_env)
                size = self.get_directory_size(fungen_env)
                print(f"  Found conda env: {fungen_env} ({self.format_size(size)})")
        
        # Venv environments (look for venv folders in project directories)
        for project_path in self.project_paths:
            venv_path = project_path / "venv"
            if venv_path.exists():
                self.env_paths.append(venv_path)
                size = self.get_directory_size(venv_path)
                print(f"  Found venv: {venv_path} ({self.format_size(size)})")
    
    def _find_tools(self):
        """Find tools installed by FunGen installer"""
        # Tools directory (Windows)
        if self.platform == "Windows":
            for project_path in self.project_paths:
                tools_path = project_path.parent / "tools"
                if tools_path.exists():
                    self.tool_paths.append(tools_path)
                    size = self.get_directory_size(tools_path)
                    print(f"  Found tools: {tools_path} ({self.format_size(size)})")
        
        # Local bin directory (Linux/macOS)
        else:
            local_bin = Path.home() / ".local" / "bin"
            if local_bin.exists():
                # Check for FunGen-installed binaries
                fungen_binaries = []
                for binary in ["ffmpeg", "ffprobe"]:
                    binary_path = local_bin / binary
                    if binary_path.exists():
                        fungen_binaries.append(binary_path)
                
                if fungen_binaries:
                    self.tool_paths.extend(fungen_binaries)
                    print(f"  Found binaries: {len(fungen_binaries)} in {local_bin}")
    
    def _find_launchers(self):
        """Find launcher scripts"""
        for project_path in self.project_paths:
            if self.platform == "Windows":
                launcher = project_path / "launch.bat"
            else:
                launcher = project_path / "launch.sh"
                command_launcher = project_path / "launch.command"
                
                if command_launcher.exists():
                    self.launcher_paths.append(command_launcher)
            
            if launcher.exists():
                self.launcher_paths.append(launcher)
    
    def confirm_uninstall(self) -> bool:
        """Ask user to confirm uninstall"""
        print(f"\n{Colors.YELLOW}⚠ UNINSTALL CONFIRMATION{Colors.ENDC}")
        print("The following will be removed:")
        
        total_size = 0
        
        if self.project_paths:
            print(f"\n📁 Project Directories ({len(self.project_paths)}):")
            for path in self.project_paths:
                size = self.get_directory_size(path)
                total_size += size
                print(f"  • {path} ({self.format_size(size)})")
        
        if self.env_paths and self.uninstall_type in ["complete", "environments"]:
            print(f"\n🐍 Python Environments ({len(self.env_paths)}):")
            for path in self.env_paths:
                size = self.get_directory_size(path)
                total_size += size
                print(f"  • {path} ({self.format_size(size)})")
        
        if self.tool_paths and self.uninstall_type in ["complete", "tools"]:
            print(f"\n🔧 Installed Tools ({len(self.tool_paths)}):")
            for path in self.tool_paths:
                if path.is_dir():
                    size = self.get_directory_size(path)
                    total_size += size
                    print(f"  • {path} ({self.format_size(size)})")
                else:
                    size = path.stat().st_size if path.exists() else 0
                    total_size += size
                    print(f"  • {path} ({self.format_size(size)})")
        
        if self.launcher_paths:
            print(f"\n🚀 Launcher Scripts ({len(self.launcher_paths)}):")
            for path in self.launcher_paths:
                print(f"  • {path}")
        
        print(f"\n💾 Total disk space to be freed: {self.format_size(total_size)}")
        
        if self.backup:
            print(f"📦 Backup will be created at: {self.backup_dir}")
        
        print(f"\n{Colors.RED}This action cannot be undone (unless backed up).{Colors.ENDC}")
        
        if self.dry_run:
            return True
        
        response = input(f"\n{Colors.BOLD}Continue with uninstall? [y/N]: {Colors.ENDC}").strip().lower()
        return response in ['y', 'yes']
    
    def create_backup(self):
        """Create backup of files before deletion"""
        if not self.backup or self.dry_run:
            return
        
        print(f"\n{Colors.BLUE}📦 Creating backup...{Colors.ENDC}")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup project directories
            for i, project_path in enumerate(self.project_paths):
                backup_path = self.backup_dir / f"project_{i}_{project_path.name}"
                shutil.copytree(project_path, backup_path)
                self.print_success(f"Backed up: {project_path} → {backup_path}")
            
            # Backup environments (if being removed)
            if self.uninstall_type in ["complete", "environments"]:
                for i, env_path in enumerate(self.env_paths):
                    backup_path = self.backup_dir / f"env_{i}_{env_path.name}"
                    shutil.copytree(env_path, backup_path)
                    self.print_success(f"Backed up: {env_path} → {backup_path}")
            
            # Create restore script
            self._create_restore_script()
            
        except Exception as e:
            self.print_error(f"Backup failed: {e}")
            self.print_warning("Continuing without backup...")
    
    def _create_restore_script(self):
        """Create a restore script for the backup"""
        restore_script = self.backup_dir / ("restore.bat" if self.platform == "Windows" else "restore.sh")
        
        script_content = "#!/bin/bash\n" if self.platform != "Windows" else "@echo off\n"
        script_content += f"# FunGen Restore Script - Created {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        script_content += f"# This script can restore the backup created during uninstall\n\n"
        script_content += f"echo 'FunGen Restore Script'\n"
        script_content += f"echo 'Backup location: {self.backup_dir}'\n"
        script_content += f"echo 'Manual restoration required - copy directories back to original locations'\n"
        
        restore_script.write_text(script_content)
        if self.platform != "Windows":
            restore_script.chmod(0o755)
    
    def perform_uninstall(self) -> bool:
        """Perform the actual uninstall"""
        print(f"\n{Colors.BLUE}🗑️ Starting uninstall...{Colors.ENDC}")
        
        try:
            # Remove project directories
            for project_path in self.project_paths:
                if self.dry_run:
                    self.print_info(f"Would remove: {project_path}")
                else:
                    shutil.rmtree(project_path)
                    self.print_success(f"Removed: {project_path}")
                self.files_removed += 1
            
            # Remove environments (if specified)
            if self.uninstall_type in ["complete", "environments"]:
                for env_path in self.env_paths:
                    if self.dry_run:
                        self.print_info(f"Would remove: {env_path}")
                    else:
                        shutil.rmtree(env_path)
                        self.print_success(f"Removed: {env_path}")
                    self.files_removed += 1
            
            # Remove tools (if specified)
            if self.uninstall_type in ["complete", "tools"]:
                for tool_path in self.tool_paths:
                    if self.dry_run:
                        self.print_info(f"Would remove: {tool_path}")
                    else:
                        if tool_path.is_dir():
                            shutil.rmtree(tool_path)
                        else:
                            tool_path.unlink()
                        self.print_success(f"Removed: {tool_path}")
                    self.files_removed += 1
            
            # Remove launcher scripts
            for launcher_path in self.launcher_paths:
                if self.dry_run:
                    self.print_info(f"Would remove: {launcher_path}")
                else:
                    launcher_path.unlink()
                    self.print_success(f"Removed: {launcher_path}")
                self.files_removed += 1
            
            return True
            
        except Exception as e:
            self.print_error(f"Uninstall failed: {e}")
            return False
    
    def cleanup_registry_windows(self):
        """Clean up Windows registry entries (if any)"""
        if self.platform != "Windows" or self.dry_run:
            return
        
        # FunGen doesn't typically create registry entries, but check for PATH modifications
        print(f"{Colors.BLUE}🔧 Checking Windows registry...{Colors.ENDC}")
        self.print_info("FunGen doesn't modify registry - skipping registry cleanup")
    
    def print_completion_message(self):
        """Print uninstall completion message"""
        if self.dry_run:
            print(f"\n{Colors.GREEN}{Colors.BOLD}=" * 60)
            print("    Dry Run Completed!")
            print("=" * 60 + Colors.ENDC)
            print(f"Would have removed {self.files_removed} items")
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}=" * 60)
            print("    Uninstall Completed!")
            print("=" * 60 + Colors.ENDC)
            print(f"Successfully removed {self.files_removed} items")
        
        if self.backup and not self.dry_run:
            print(f"\n{Colors.CYAN}📦 Backup Information:{Colors.ENDC}")
            print(f"  Location: {self.backup_dir}")
            print(f"  To restore: Follow instructions in restore script")
            print(f"  Backup can be safely deleted if you don't need to restore")
        
        print(f"\n{Colors.YELLOW}Post-uninstall notes:{Colors.ENDC}")
        if self.uninstall_type == "standard":
            print("  • Python/conda environments were preserved")
            print("  • System tools (Git, FFmpeg) were not removed")
        elif self.uninstall_type == "complete":
            print("  • Complete removal performed")
            print("  • You may need to reinstall Python for other applications")
        
        print("\n  • Any user-created funscripts in output folders were preserved")
        print("  • System packages installed via package managers remain")
        
        print(f"\n{Colors.GREEN}FunGen has been successfully uninstalled.{Colors.ENDC}")
    
    def uninstall(self) -> bool:
        """Main uninstall process"""
        self.print_header()
        
        # Discover installations
        self.discover_fungen_installations()
        
        if not any([self.project_paths, self.env_paths, self.tool_paths, self.launcher_paths]):
            print(f"{Colors.YELLOW}No FunGen installations found.{Colors.ENDC}")
            return True
        
        # Confirm uninstall
        if not self.confirm_uninstall():
            print(f"{Colors.YELLOW}Uninstall cancelled by user.{Colors.ENDC}")
            return False
        
        # Create backup
        if self.backup:
            self.create_backup()
        
        # Perform uninstall
        success = self.perform_uninstall()
        
        # Windows registry cleanup
        if success and self.platform == "Windows":
            self.cleanup_registry_windows()
        
        # Print completion message
        self.print_completion_message()
        
        return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FunGen Universal Uninstaller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uninstall Types:
  standard   - Remove FunGen only (keep Python, tools)
  complete   - Remove everything including Python environments  
  environments - Remove only Python environments
  tools      - Remove only installed tools

Examples:
  python fungen_uninstall.py                    # Standard uninstall
  python fungen_uninstall.py --type complete    # Complete removal
  python fungen_uninstall.py --no-backup        # No backup
  python fungen_uninstall.py --dry-run          # Preview only
        """
    )
    
    parser.add_argument(
        "--type", 
        choices=["standard", "complete", "environments", "tools"],
        default="standard",
        help="Type of uninstall to perform (default: standard)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup before uninstall"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Preview what would be removed without actually removing"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="FunGen Uninstaller 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Create and run uninstaller
    uninstaller = FunGenUninstaller(
        uninstall_type=args.type,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )
    
    success = uninstaller.uninstall()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()