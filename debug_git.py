#!/usr/bin/env python3
"""
Git Detection Diagnostic Script for FunGen
Run this to troubleshoot unknown@unknown issues
"""

import os
import subprocess
import sys
from pathlib import Path

def check_git_availability():
    """Check if git is available and working"""
    print("=== Git Diagnostic Tool ===\n")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")
    print()
    
    # Check if git command exists
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Git is available: {result.stdout.strip()}")
        else:
            print(f"❌ Git command failed: return code {result.returncode}")
            print(f"   Stderr: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Git command not found in PATH")
        return False
    except Exception as e:
        print(f"❌ Error running git: {e}")
        return False
    
    # Check if we're in a git repository
    try:
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            git_dir = result.stdout.strip()
            print(f"✅ Git repository found: {git_dir}")
        else:
            print(f"❌ Not in a git repository")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking git repository: {e}")
        return False
    
    # Try to get branch name
    try:
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            branch = result.stdout.strip()
            print(f"✅ Current branch: {branch}")
        else:
            print(f"❌ Could not get branch name")
            print(f"   Return code: {result.returncode}")
            print(f"   Stdout: '{result.stdout}'")
            print(f"   Stderr: '{result.stderr}'")
    except Exception as e:
        print(f"❌ Error getting branch: {e}")
    
    # Try to get commit hash
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
            print(f"✅ Current commit: {commit}")
        else:
            print(f"❌ Could not get commit hash")
            print(f"   Return code: {result.returncode}")
            print(f"   Stdout: '{result.stdout}'")
            print(f"   Stderr: '{result.stderr}'")
    except Exception as e:
        print(f"❌ Error getting commit: {e}")
    
    print("\n=== Recommendations ===")
    print("If you see ❌ errors above:")
    print("1. Make sure you're running from the FunGen project directory")
    print("2. Use the launcher script (launch.bat/launch.sh) instead of 'python main.py'")
    print("3. Rerun the installer to get updated launcher scripts")
    print("4. Check if Git is properly installed and in PATH")
    
    return True

if __name__ == "__main__":
    check_git_availability()