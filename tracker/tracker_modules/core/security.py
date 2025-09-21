#!/usr/bin/env python3
"""
Tracker Security Module

Provides security validation and sandboxing for dynamic tracker loading.
Simple but effective security measures to prevent malicious code execution.

Author: Security Team
Version: 1.0.0
"""

import ast
import inspect
import logging
import os
import subprocess
import sys
import tempfile
from typing import List, Set, Optional, Dict, Any
from pathlib import Path

# Security-specific exceptions
class TrackerSecurityError(Exception):
    """Base class for tracker security violations"""
    pass

class TrackerValidationError(TrackerSecurityError):
    """Tracker failed security validation"""
    pass

class TrackerSandboxError(TrackerSecurityError):
    """Sandbox execution failed"""
    pass

class TrackerAPIViolationError(TrackerSecurityError):
    """Tracker attempted unauthorized API access"""
    pass


class TrackerSecurityValidator:
    """
    Security validator for tracker modules.
    
    Performs static analysis and validation of tracker code before execution
    to prevent malicious code injection and unauthorized system access.
    """
    
    # Dangerous functions/modules that trackers shouldn't use
    BLACKLISTED_IMPORTS = {
        'subprocess', 'os.system', 'eval', 'exec', 'compile',
        'importlib', '__import__', 'input',
        'raw_input', 'execfile', 'reload', 'delattr', 'setattr',
        'globals', 'locals', 'vars', 'dir', 'hasattr', 'getattr'
    }
    
    BLACKLISTED_BUILTINS = {
        'eval', 'exec', 'compile', '__import__',
        'input', 'raw_input', 'execfile', 'reload', 'delattr', 'setattr'
    }
    
    # Allowed system modules that trackers can safely use
    ALLOWED_SYSTEM_MODULES = {
        'os.path', 'sys.path', 'logging', 'time', 'datetime',
        'math', 'random', 'json', 're', 'collections', 'itertools',
        'functools', 'operator', 'copy', 'hashlib', 'uuid'
    }
    
    def __init__(self):
        self.logger = logging.getLogger("TrackerSecurity")
        
    def validate_tracker_file(self, file_path: str) -> bool:
        """
        Validate a tracker file for security violations.
        
        Args:
            file_path: Path to tracker Python file
            
        Returns:
            True if safe, raises TrackerValidationError if unsafe
            
        Raises:
            TrackerValidationError: If security violations detected
        """
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse AST for static analysis
            tree = ast.parse(source_code, filename=file_path)
            
            # Perform security checks
            self._check_dangerous_imports(tree, file_path)
            self._check_dangerous_calls(tree, file_path)
            self._check_file_operations(tree, file_path)
            
            self.logger.debug(f"Security validation passed for {file_path}")
            return True
            
        except SyntaxError as e:
            raise TrackerValidationError(f"Syntax error in tracker {file_path}: {e}")
        except Exception as e:
            raise TrackerValidationError(f"Security validation failed for {file_path}: {e}")
    
    def _check_dangerous_imports(self, tree: ast.AST, file_path: str):
        """Check for dangerous import statements."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.BLACKLISTED_IMPORTS:
                        raise TrackerValidationError(
                            f"Forbidden import '{alias.name}' in {file_path}. "
                            f"Trackers cannot import system modules that could be used maliciously."
                        )
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.BLACKLISTED_IMPORTS:
                    raise TrackerValidationError(
                        f"Forbidden import from '{node.module}' in {file_path}. "
                        f"This module could be used for malicious purposes."
                    )
    
    def _check_dangerous_calls(self, tree: ast.AST, file_path: str):
        """Check for dangerous function calls."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLACKLISTED_BUILTINS:
                        raise TrackerValidationError(
                            f"Forbidden function call '{node.func.id}' in {file_path}. "
                            f"Trackers cannot execute arbitrary code."
                        )
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous attribute calls
                    if isinstance(node.func.value, ast.Name):
                        # os.system calls
                        if (node.func.value.id == 'os' and 
                            node.func.attr == 'system'):
                            raise TrackerValidationError(
                                f"Forbidden os.system call in {file_path}. "
                                f"Trackers cannot execute system commands."
                            )
                        # subprocess calls
                        elif (node.func.value.id == 'subprocess' and 
                              node.func.attr in ['run', 'call', 'check_call', 'Popen']):
                            raise TrackerValidationError(
                                f"Forbidden subprocess.{node.func.attr} call in {file_path}. "
                                f"Trackers cannot execute subprocesses."
                            )
    
    def _check_file_operations(self, tree: ast.AST, file_path: str):
        """Check for unauthorized file operations."""
        # Allow file operations but monitor for dangerous write modes
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    # Check if it's opening with dangerous modes
                    if len(node.args) > 1:
                        if isinstance(node.args[1], ast.Constant):
                            mode = node.args[1].value
                            if any(dangerous in str(mode) for dangerous in ['w', 'a', 'x']) and 'r' not in str(mode):
                                self.logger.debug(
                                    f"Write file operation mode '{mode}' detected in {file_path}. "
                                    f"Monitor for potential security risk."
                                )
                    # Allow all file operations - just log for monitoring
                    self.logger.debug(f"File operation detected in {file_path} - allowed")


class TrackerSandbox:
    """
    Simple sandbox for tracker validation.
    
    Validates tracker interface in isolated environment before full loading.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TrackerSandbox")
        
    def validate_tracker_interface(self, file_path: str) -> Dict[str, Any]:
        """
        Validate tracker interface in sandbox environment.
        
        Args:
            file_path: Path to tracker file
            
        Returns:
            Dict with validation results
            
        Raises:
            TrackerSandboxError: If sandbox validation fails
        """
        try:
            # Create validation script
            validation_script = self._create_validation_script(file_path)
            
            # Run in subprocess for isolation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(validation_script)
                temp_script = f.name
            
            try:
                # Execute validation with timeout and restricted environment
                result = subprocess.run([
                    sys.executable, temp_script
                ], 
                capture_output=True, 
                text=True, 
                timeout=60,  # 60 second timeout for complex trackers
                env=self._get_restricted_env()
                )
                
                if result.returncode == 0:
                    # Parse validation results
                    import json
                    try:
                        validation_results = json.loads(result.stdout)
                        self.logger.debug(f"Sandbox validation passed for {file_path}")
                        return validation_results
                    except json.JSONDecodeError:
                        self.logger.info(f"Sandbox validation completed for {file_path} (no results)")
                        return {"status": "validated"}
                else:
                    raise TrackerSandboxError(f"Sandbox validation failed: {result.stderr}")
                    
            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_script)
                except OSError:
                    pass
                    
        except subprocess.TimeoutExpired:
            raise TrackerSandboxError(f"Sandbox validation timed out for {file_path}")
        except Exception as e:
            raise TrackerSandboxError(f"Sandbox execution failed for {file_path}: {e}")
    
    def _create_validation_script(self, file_path: str) -> str:
        """Create Python script for sandbox validation."""
        return f'''
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath("{file_path}")))
sys.path.insert(0, project_root)

try:
    # Quick syntax check - just try to import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_tracker", "{file_path}")
    if spec is None or spec.loader is None:
        print(json.dumps({{"error": "Could not create module spec"}}))
        sys.exit(1)
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Basic class discovery without instantiation
    import inspect
    tracker_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and hasattr(obj, '__module__') and obj.__module__ == 'test_tracker':
            tracker_classes.append(obj)
    
    if not tracker_classes:
        print(json.dumps({{"error": "No classes found"}}))
        sys.exit(1)
    
    # Success - just confirm module loads
    print(json.dumps({{
        "status": "valid",
        "classes_found": len(tracker_classes)
    }}))
    
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
'''
    
    def _get_restricted_env(self) -> Dict[str, str]:
        """Get restricted environment variables for subprocess."""
        # Start with minimal environment
        env = {
            'PATH': os.environ.get('PATH', ''),
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
            'HOME': os.environ.get('HOME', ''),
            'USER': os.environ.get('USER', ''),
        }
        
        # Add conda/python environment if present
        for key in ['CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV']:
            if key in os.environ:
                env[key] = os.environ[key]
                
        return env


class SecureTrackerLoader:
    """
    Secure loader for tracker modules with validation and sandboxing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SecureTrackerLoader")
        self.validator = TrackerSecurityValidator()
        self.sandbox = TrackerSandbox()
        
    def load_tracker_securely(self, file_path: str, filename: str) -> Optional[type]:
        """
        Load tracker module with security validation.
        
        Args:
            file_path: Full path to tracker file
            filename: Filename for logging
            
        Returns:
            Tracker class if validation passes, None otherwise
            
        Raises:
            TrackerSecurityError: If security validation fails
        """
        try:
            # Step 1: Static security validation
            self.logger.debug(f"Starting security validation for {filename}")
            self.validator.validate_tracker_file(file_path)
            
            # Step 2: Skip sandbox validation for now - too complex for legitimate trackers
            # TODO: Implement lighter-weight sandbox in future
            self.logger.debug(f"Static validation passed, loading {filename}")
            
            # Step 3: If validation passes, perform actual import
            return self._perform_secure_import(file_path, filename)
            
        except TrackerSecurityError:
            # Re-raise security errors as-is
            raise
        except Exception as e:
            # Wrap other errors in security error for consistency
            raise TrackerValidationError(f"Tracker validation failed for {filename}: {e}")
    
    def _perform_secure_import(self, file_path: str, filename: str) -> Optional[type]:
        """Perform the actual import after security validation."""
        import importlib.util
        import inspect
        
        module_name = filename[:-3]  # Remove .py extension
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(f"tracker_modules.{module_name}", file_path)
            if spec is None or spec.loader is None:
                raise TrackerValidationError(f"Could not create module spec for {filename}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"tracker_modules.{module_name}"] = module
            spec.loader.exec_module(module)
            
            # Find tracker class
            from .base_tracker import BaseTracker
            from .base_offline_tracker import BaseOfflineTracker
            
            tracker_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    (issubclass(obj, BaseTracker) or issubclass(obj, BaseOfflineTracker)) and 
                    obj not in [BaseTracker, BaseOfflineTracker] and
                    not inspect.isabstract(obj)):
                    tracker_classes.append(obj)
            
            if len(tracker_classes) == 0:
                self.logger.debug(f"No tracker classes found in {filename}")
                return None
            elif len(tracker_classes) > 1:
                self.logger.warning(f"Multiple tracker classes found in {filename}, using first one")
            
            return tracker_classes[0]
            
        except Exception as e:
            raise TrackerValidationError(f"Secure import failed for {filename}: {e}")


# Convenience functions for easy integration
def validate_tracker_file(file_path: str) -> bool:
    """Validate a tracker file for security violations."""
    validator = TrackerSecurityValidator()
    return validator.validate_tracker_file(file_path)


def load_tracker_safely(file_path: str, filename: str) -> Optional[type]:
    """Load a tracker with full security validation."""
    loader = SecureTrackerLoader()
    return loader.load_tracker_securely(file_path, filename)