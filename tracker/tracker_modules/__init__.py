"""
Modular tracker system with auto-discovery.

This module automatically discovers and registers all tracker implementations,
making them available for use in the application without requiring manual
configuration or hardcoded lists.
"""

import ast
import os
import sys
import importlib.util
import inspect
import logging
from typing import Dict, List, Type, Optional

try:
    from .core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, TrackerError
    from .core.base_offline_tracker import BaseOfflineTracker
    from .core.security import (
        TrackerSecurityError, TrackerValidationError, TrackerSandboxError, 
        TrackerAPIViolationError, load_tracker_safely
    )
except ImportError:
    # Fallback for direct execution
    from core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, TrackerError
    from core.base_offline_tracker import BaseOfflineTracker
    from core.security import (
        TrackerSecurityError, TrackerValidationError, TrackerSandboxError,
        TrackerAPIViolationError, load_tracker_safely
    )


class _NotLiteral(Exception):
    """Raised when an AST node can't be evaluated as a literal."""
    pass


class TrackerRegistry:
    """
    Registry that automatically discovers and manages tracker implementations.
    
    The registry scans the tracker_modules directory and community subdirectory
    for Python files containing BaseTracker subclasses, validates them, and
    makes them available for instantiation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TrackerRegistry")
        self._trackers: Dict[str, Type] = {}
        self._metadata_cache: Dict[str, TrackerMetadata] = {}
        self._folder_map: Dict[str, str] = {}  # tracker_name -> folder_name
        self._discovery_errors: List[str] = []
        # Lazy refs: name -> (file_path, filename, folder_name, is_community).
        # Populated when metadata was extracted statically without importing
        # the module. Class is loaded on first get_tracker()/create_tracker().
        self._lazy_refs: Dict[str, tuple] = {}

        # Background discovery. Even with static AST extraction we still want
        # to scan dozens of files off the main thread so GUI init runs in
        # parallel; public getters block on _ensure_discovered() only if a
        # query arrives before scanning finishes.
        import threading
        self._discovered = threading.Event()
        self._discovery_lock = threading.Lock()  # guards lazy materialization
        self._discovery_thread = threading.Thread(
            target=self._run_discovery, daemon=True, name="TrackerDiscovery")
        self._discovery_thread.start()

    def _run_discovery(self) -> None:
        """Thread body: run _discover_trackers, then signal completion."""
        try:
            self._discover_trackers()
            total = len(self._trackers) + len(self._lazy_refs)
            if total:
                self.logger.debug(
                    f"Discovered {total} trackers ({len(self._lazy_refs)} lazy, "
                    f"{len(self._trackers)} eager)")
            else:
                self.logger.warning("No trackers discovered!")
            if self._discovery_errors:
                self.logger.warning(f"Discovery errors: {len(self._discovery_errors)}")
        finally:
            self._discovered.set()

    def _ensure_discovered(self) -> None:
        """Block the current thread until the discovery thread finishes.

        Called at the top of every public getter. No-op if discovery is
        already complete (cheap Event.wait fast path).
        """
        if not self._discovered.is_set():
            self._discovered.wait()
    
    def _discover_trackers(self):
        """Auto-discover tracker modules in the tracker_modules subdirectories."""
        tracker_dir = os.path.dirname(__file__)
        
        # Scan live trackers subdirectory
        live_dir = os.path.join(tracker_dir, 'live')
        if os.path.exists(live_dir):
            self._scan_directory(live_dir, folder_name='live', is_community=False)
        
        # Scan offline trackers subdirectory
        offline_dir = os.path.join(tracker_dir, 'offline')
        if os.path.exists(offline_dir):
            self._scan_directory(offline_dir, folder_name='offline', is_community=False)

        # Scan tool trackers subdirectory (accessory utilities, opt-in via UI filter)
        tool_dir = os.path.join(tracker_dir, 'tool')
        if os.path.exists(tool_dir):
            self._scan_directory(tool_dir, folder_name='tool', is_community=False)
        
        # Scan legacy trackers
        legacy_dir = os.path.join(tracker_dir, 'legacy')
        if os.path.exists(legacy_dir):
            self._scan_directory(legacy_dir, folder_name='legacy', is_community=False)


        # Scan community subdirectory
        community_dir = os.path.join(tracker_dir, 'community')
        if os.path.exists(community_dir):
            self._scan_directory(community_dir, folder_name='community', is_community=True)
        
        self.logger.debug(
            f"Discovery complete. Found {len(self._trackers) + len(self._lazy_refs)} trackers.")
    
    def _scan_directory(self, directory: str, folder_name: str, is_community: bool = False):
        """Scan a directory for tracker modules.

        For community trackers, also recursively scans subdirectories so users
        can organize their trackers in folders (e.g. community/my_trackers/).
        """
        try:
            for filename in os.listdir(directory):
                full_path = os.path.join(directory, filename)
                if filename.endswith('.py') and filename not in ['__init__.py']:
                    self._load_tracker_module(full_path, filename, folder_name, is_community)
                elif is_community and os.path.isdir(full_path) and not filename.startswith(('__', '.')):
                    # Recursively scan user-created subdirectories within community/
                    self.logger.debug(f"Scanning community subfolder: {filename}/")
                    self._scan_directory(full_path, folder_name, is_community)
        except OSError as e:
            error_msg = f"Failed to scan directory {directory}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
    
    def _load_tracker_module(self, file_path: str, filename: str, folder_name: str, is_community: bool):
        """Register a tracker by metadata, deferring the actual class import.

        Static AST extraction avoids pulling torch/ultralytics/cv2 for trackers
        the user never selects. Files without a tracker subclass are skipped.
        Files with one but non-literal metadata fall back to eager load.
        """
        try:
            tree = self._parse_file(file_path)
            if tree is None or not self._has_tracker_subclass(tree):
                return  # standalone script, test file, helper — not a tracker
            meta = self._extract_metadata_from_tree(tree)
            if meta is not None:
                self._register_lazy(meta, file_path, filename, folder_name, is_community)
                return
            # Tracker subclass exists but metadata isn't literal — eager path.
            if is_community:
                tracker_class = load_tracker_safely(file_path, filename)
            else:
                tracker_class = self._direct_import(file_path, filename)
            if tracker_class:
                self._register_tracker(tracker_class, folder_name, is_community, file_path)
            else:
                self.logger.debug(f"No valid tracker classes found in {filename}")

        except TrackerSecurityError as e:
            error_msg = f"SECURITY VIOLATION in {filename}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to load tracker module {filename}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.warning(error_msg)

    @staticmethod
    def _parse_file(file_path: str):
        try:
            with open(file_path, "rb") as f:
                return ast.parse(f.read(), filename=file_path)
        except (OSError, SyntaxError):
            return None

    @staticmethod
    def _has_tracker_subclass(tree) -> bool:
        """True iff any top-level class inherits from BaseTracker/BaseOfflineTracker by name."""
        wanted = ("BaseTracker", "BaseOfflineTracker")
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in wanted:
                    return True
                if isinstance(base, ast.Attribute) and base.attr in wanted:
                    return True
        return False

    def _extract_metadata_from_tree(self, tree) -> Optional[TrackerMetadata]:
        """Find `@property def metadata(self)` in an AST and evaluate its return."""
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name != "metadata":
                    continue
                if not any(
                    (isinstance(d, ast.Name) and d.id == "property")
                    for d in item.decorator_list
                ):
                    continue
                return self._eval_metadata_body(item)
        return None

    def _eval_metadata_body(self, func_node) -> Optional[TrackerMetadata]:
        """Find `return TrackerMetadata(...)` inside a metadata property body."""
        for stmt in ast.walk(func_node):
            if not isinstance(stmt, ast.Return) or stmt.value is None:
                continue
            call = stmt.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "TrackerMetadata"):
                return None
            try:
                args = [self._literalize(a) for a in call.args]
                kwargs = {kw.arg: self._literalize(kw.value) for kw in call.keywords if kw.arg}
            except _NotLiteral:
                return None
            try:
                return TrackerMetadata(*args, **kwargs)
            except Exception:
                return None
        return None

    @staticmethod
    def _literalize(node):
        """Recursive evaluator: literals + nested StageDefinition(...) calls."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "True": return True
            if node.id == "False": return False
            if node.id == "None": return None
            raise _NotLiteral()
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [TrackerRegistry._literalize(e) for e in node.elts]
            if isinstance(node, ast.Tuple): return tuple(items)
            if isinstance(node, ast.Set): return set(items)
            return items
        if isinstance(node, ast.Dict):
            return {
                TrackerRegistry._literalize(k): TrackerRegistry._literalize(v)
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = TrackerRegistry._literalize(node.operand)
            return -val if isinstance(node.op, ast.USub) else +val
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "StageDefinition":
            args = [TrackerRegistry._literalize(a) for a in node.args]
            kwargs = {kw.arg: TrackerRegistry._literalize(kw.value) for kw in node.keywords if kw.arg}
            from .core.base_tracker import StageDefinition
            return StageDefinition(*args, **kwargs)
        raise _NotLiteral()

    def _register_lazy(self, metadata: TrackerMetadata, file_path: str,
                       filename: str, folder_name: str, is_community: bool) -> None:
        """Register metadata + a lazy ref; class loads on first get_tracker()."""
        if not isinstance(metadata, TrackerMetadata) or not metadata.name:
            return
        if metadata.name in self._trackers or metadata.name in self._lazy_refs:
            existing = self._metadata_cache.get(metadata.name)
            self.logger.warning(
                f"Tracker name conflict: '{metadata.name}' "
                f"(existing: {existing.display_name if existing else '?'}, "
                f"new: {metadata.display_name}). Skipping new tracker."
            )
            return
        self._metadata_cache[metadata.name] = metadata
        self._folder_map[metadata.name] = folder_name
        self._lazy_refs[metadata.name] = (file_path, filename, folder_name, is_community)

    def _materialize(self, name: str) -> Optional[Type]:
        """Load the class for a lazily-registered tracker. Thread-safe."""
        with self._discovery_lock:
            if name in self._trackers:
                return self._trackers[name]
            ref = self._lazy_refs.get(name)
            if ref is None:
                return None
            file_path, filename, folder_name, is_community = ref
            try:
                if is_community:
                    tracker_class = load_tracker_safely(file_path, filename)
                else:
                    tracker_class = self._direct_import(file_path, filename)
            except TrackerSecurityError as e:
                self.logger.error(f"SECURITY VIOLATION materializing {name}: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Failed to materialize tracker {name}: {e}")
                return None
            if tracker_class is None:
                return None
            self._trackers[name] = tracker_class
            # Refresh metadata from the live class in case the static parse
            # missed anything (e.g. auto-generated properties).
            try:
                live_meta = tracker_class().metadata
                if isinstance(live_meta, TrackerMetadata) and live_meta.name == name:
                    self._metadata_cache[name] = live_meta
            except Exception:
                pass
            return tracker_class

    def _direct_import(self, file_path: str, filename: str) -> Optional[Type]:
        """Import an official tracker module without security sandbox."""
        module_name = filename[:-3]
        spec = importlib.util.spec_from_file_location(f"tracker_modules.{module_name}", file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"tracker_modules.{module_name}"] = module
        spec.loader.exec_module(module)

        tracker_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                (issubclass(obj, BaseTracker) or issubclass(obj, BaseOfflineTracker)) and
                obj not in [BaseTracker, BaseOfflineTracker] and
                not inspect.isabstract(obj)):
                tracker_classes.append(obj)

        return tracker_classes[0] if tracker_classes else None
    
    
    def _register_tracker(self, tracker_class: Type, folder_name: str, is_community: bool, file_path: str):
        """Register a validated tracker class with resource management."""
        temp_instance = None
        try:
            # Validate by attempting to access metadata
            # Note: We create a temporary instance just to get metadata
            # This ensures the metadata property is properly implemented
            temp_instance = tracker_class()
            metadata = temp_instance.metadata
            
            if not isinstance(metadata, TrackerMetadata):
                raise TrackerValidationError(f"Tracker metadata must be TrackerMetadata instance")
            
            if not metadata.name:
                raise TrackerValidationError(f"Tracker name cannot be empty")
            
            # Check for name conflicts
            if metadata.name in self._trackers:
                existing_metadata = self._metadata_cache[metadata.name]
                self.logger.warning(
                    f"Tracker name conflict: '{metadata.name}' "
                    f"(existing: {existing_metadata.display_name}, "
                    f"new: {metadata.display_name}). Skipping new tracker."
                )
                return
            
            # Register the tracker
            self._trackers[metadata.name] = tracker_class
            self._metadata_cache[metadata.name] = metadata
            self._folder_map[metadata.name] = folder_name
            
            # Log at debug level to reduce verbosity
            category_prefix = "[Community] " if is_community else ""
            self.logger.debug(
                f"Registered: {category_prefix}{metadata.display_name} ({metadata.name})"
            )
            
        except TrackerSecurityError as e:
            error_msg = f"SECURITY VIOLATION during registration of {tracker_class.__name__}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to register tracker {tracker_class.__name__}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
        finally:
            # Resource cleanup - ensure temp instance is properly cleaned up
            if temp_instance:
                try:
                    if hasattr(temp_instance, 'cleanup'):
                        temp_instance.cleanup()
                    # Clear references to help garbage collection
                    temp_instance = None
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp instance: {cleanup_error}")
    
    def get_tracker(self, name: str) -> Optional[Type]:
        """
        Get tracker class by name.

        Args:
            name: Internal tracker name (from metadata.name)

        Returns:
            Type: Tracker class, or None if not found
        """
        self._ensure_discovered()
        cls = self._trackers.get(name)
        if cls is not None:
            return cls
        if name in self._lazy_refs:
            return self._materialize(name)
        return None
    
    def create_tracker(self, name: str) -> Optional:
        """
        Create a new instance of the named tracker.
        
        Args:
            name: Internal tracker name (from metadata.name)
        
        Returns:
            BaseTracker or BaseOfflineTracker: New tracker instance, or None if not found
        """
        tracker_class = self.get_tracker(name)
        if tracker_class:
            try:
                return tracker_class()
            except Exception as e:
                self.logger.error(f"Failed to create tracker instance {name}: {e}")
                return None
        return None
    
    def list_trackers(self, category: Optional[str] = None) -> List[TrackerMetadata]:
        """
        List all discovered tracker metadata.

        Args:
            category: Optional category filter ("live", "offline", etc.)

        Returns:
            List[TrackerMetadata]: List of tracker metadata
        """
        self._ensure_discovered()
        trackers = list(self._metadata_cache.values())
        
        if category:
            trackers = [t for t in trackers if t.category == category]
        
        # Custom category priority: live first, then offline, then community
        # Within each category, sort alphabetically by display name
        category_priority = {'live': 1, 'offline': 2, 'community': 3}
        return sorted(trackers, key=lambda t: (category_priority.get(t.category, 999), t.display_name))
    
    def get_metadata(self, name: str) -> Optional[TrackerMetadata]:
        """
        Get metadata for a specific tracker.

        Args:
            name: Internal tracker name

        Returns:
            TrackerMetadata: Tracker metadata, or None if not found
        """
        self._ensure_discovered()
        return self._metadata_cache.get(name)

    def get_available_names(self) -> List[str]:
        """Get list of all available tracker names."""
        self._ensure_discovered()
        names = set(self._trackers.keys())
        names.update(self._lazy_refs.keys())
        return list(names)

    def get_discovery_errors(self) -> List[str]:
        """Get list of errors encountered during discovery."""
        self._ensure_discovered()
        return self._discovery_errors.copy()

    def get_tracker_folder(self, name: str) -> Optional[str]:
        """Get folder name for a specific tracker."""
        self._ensure_discovered()
        return self._folder_map.get(name)
    
    def reload_trackers(self):
        """Reload all trackers (useful for development)."""
        self.logger.debug("Reloading trackers...")
        self._trackers.clear()
        self._metadata_cache.clear()
        self._folder_map.clear()
        self._lazy_refs.clear()
        self._discovery_errors.clear()
        self._discover_trackers()


# Global registry instance - automatically discovers trackers on import
tracker_registry = TrackerRegistry()


def get_tracker_registry() -> TrackerRegistry:
    """Get the global tracker registry instance."""
    return tracker_registry


def list_available_trackers(category: Optional[str] = None) -> List[TrackerMetadata]:
    """
    Convenience function to list available trackers.
    
    Args:
        category: Optional category filter
    
    Returns:
        List[TrackerMetadata]: Available trackers
    """
    return tracker_registry.list_trackers(category)


def create_tracker(name: str):
    """
    Convenience function to create a tracker instance.
    
    Args:
        name: Internal tracker name
    
    Returns:
        BaseTracker or BaseOfflineTracker: New tracker instance, or None if not found
    """
    return tracker_registry.create_tracker(name)


# Export commonly used classes for easy importing
__all__ = [
    'TrackerRegistry', 
    'tracker_registry',
    'BaseTracker',
    'BaseOfflineTracker',
    'TrackerMetadata', 
    'TrackerResult',
    'TrackerError',
    'TrackerSecurityError',
    'TrackerValidationError', 
    'TrackerSandboxError',
    'TrackerAPIViolationError',
    'get_tracker_registry',
    'list_available_trackers',
    'create_tracker'
]