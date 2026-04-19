"""
Modular tracker system with auto-discovery.

Built-in trackers (live / offline / tool / legacy) are discovered via
pkgutil.iter_modules on each folder's package, which enumerates submodules
regardless of whether they are Python source or compiled .so / .pyd. For
each module, a lightweight AST parse pulls the metadata when source is
available; otherwise the module is eager-imported and the live class is
queried. Loading uses importlib.import_module throughout, so the same
discovery code path works in source and compiled builds.

Community trackers stay on the file-path security sandbox
(load_tracker_safely) because they are user-supplied .py by contract
and need code validation before execution.
"""

import ast
import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
from typing import Dict, List, Optional, Type

try:
    from .core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, TrackerError
    from .core.base_offline_tracker import BaseOfflineTracker
    from .core.security import (
        TrackerSecurityError, TrackerValidationError, TrackerSandboxError,
        TrackerAPIViolationError, load_tracker_safely
    )
except ImportError:
    from core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, TrackerError
    from core.base_offline_tracker import BaseOfflineTracker
    from core.security import (
        TrackerSecurityError, TrackerValidationError, TrackerSandboxError,
        TrackerAPIViolationError, load_tracker_safely
    )


# Built-in tracker folder packages -- discovered by pkgutil.
BUILTIN_FOLDERS = ("live", "offline", "tool", "legacy")


class _NotLiteral(Exception):
    pass


class TrackerRegistry:
    """Discovers and manages tracker implementations."""

    def __init__(self):
        self.logger = logging.getLogger("TrackerRegistry")
        self._trackers: Dict[str, Type] = {}
        self._metadata_cache: Dict[str, TrackerMetadata] = {}
        self._folder_map: Dict[str, str] = {}  # tracker_name -> folder_name
        self._discovery_errors: List[str] = []

        # Lazy refs for built-in trackers: name -> (dotted_module_name, folder).
        # Resolved via importlib.import_module on first materialization, which
        # is indifferent to .py vs .so.
        self._builtin_refs: Dict[str, tuple] = {}

        # Lazy refs for community trackers: name -> (file_path, filename, folder).
        # Resolved via the security sandbox (load_tracker_safely). Always .py.
        self._community_refs: Dict[str, tuple] = {}

        import threading
        self._discovered = threading.Event()
        self._discovery_lock = threading.Lock()
        self._discovery_thread = threading.Thread(
            target=self._run_discovery, daemon=True, name="TrackerDiscovery")
        self._discovery_thread.start()

    def _run_discovery(self) -> None:
        try:
            self._discover_trackers()
            total = len(self._trackers) + len(self._builtin_refs) + len(self._community_refs)
            if total:
                self.logger.debug(
                    f"Discovered {total} trackers ({len(self._builtin_refs)} built-in lazy, "
                    f"{len(self._community_refs)} community lazy, "
                    f"{len(self._trackers)} eager)")
            else:
                self.logger.warning("No trackers discovered!")
            if self._discovery_errors:
                self.logger.warning(f"Discovery errors: {len(self._discovery_errors)}")
        finally:
            self._discovered.set()

    def _ensure_discovered(self) -> None:
        if not self._discovered.is_set():
            self._discovered.wait()

    def _discover_trackers(self):
        """Walk built-in folder packages via pkgutil, then scan community/."""
        for folder in BUILTIN_FOLDERS:
            pkg_name = f"tracker.tracker_modules.{folder}"
            try:
                pkg = importlib.import_module(pkg_name)
            except ImportError:
                continue  # folder not present on this install
            if not hasattr(pkg, "__path__"):
                continue
            for _finder, mod_name, _ispkg in pkgutil.iter_modules(pkg.__path__):
                if mod_name.startswith("_"):
                    continue
                dotted = f"{pkg_name}.{mod_name}"
                try:
                    self._discover_builtin(dotted, folder)
                except Exception as e:
                    err = f"Discovery error in {dotted}: {e}"
                    self._discovery_errors.append(err)
                    self.logger.warning(err)

        tracker_dir = os.path.dirname(__file__)
        community_dir = os.path.join(tracker_dir, "community")
        if os.path.isdir(community_dir):
            self._scan_community(community_dir)

    def _discover_builtin(self, dotted: str, folder: str) -> None:
        """Register a built-in tracker by dotted module name.

        Prefers AST parse of the source .py file when available, so
        metadata loads without importing torch / ultralytics. Falls back
        to eager import if the module is compiled (.so) or if its metadata
        property is not a literal call.
        """
        try:
            spec = importlib.util.find_spec(dotted)
        except (ImportError, ValueError):
            return
        if spec is None:
            return

        origin = spec.origin
        if origin and origin.endswith(".py"):
            tree = self._parse_file(origin)
            if tree is None or not self._has_tracker_subclass(tree):
                return
            meta = self._extract_metadata_from_tree(tree)
            if meta is not None:
                self._register_builtin_lazy(meta, dotted, folder)
                return
            # Metadata not literal -- fall through to eager load.

        # Compiled module or non-literal metadata: import now.
        try:
            tracker_class = self._import_and_find_class(dotted)
        except Exception as e:
            err = f"Failed to import built-in tracker {dotted}: {e}"
            self._discovery_errors.append(err)
            self.logger.warning(err)
            return
        if tracker_class is not None:
            self._register_tracker(tracker_class, folder, is_community=False, source=dotted)

    def _import_and_find_class(self, dotted: str) -> Optional[Type]:
        """Import dotted and return the first concrete BaseTracker subclass."""
        module = importlib.import_module(dotted)
        for _name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj)
                    and (issubclass(obj, BaseTracker) or issubclass(obj, BaseOfflineTracker))
                    and obj not in (BaseTracker, BaseOfflineTracker)
                    and not inspect.isabstract(obj)):
                return obj
        return None

    def _scan_community(self, directory: str) -> None:
        """Directory scan for community trackers -- always .py, sandboxed."""
        try:
            entries = os.listdir(directory)
        except OSError as e:
            err = f"Failed to scan community directory {directory}: {e}"
            self._discovery_errors.append(err)
            self.logger.error(err)
            return

        for filename in entries:
            full_path = os.path.join(directory, filename)
            if filename.endswith(".py") and filename != "__init__.py":
                self._load_community(full_path, filename)
            elif os.path.isdir(full_path) and not filename.startswith(("__", ".")):
                # Recurse into user-organised subfolders.
                self._scan_community(full_path)

    def _load_community(self, file_path: str, filename: str) -> None:
        try:
            tree = self._parse_file(file_path)
            if tree is None or not self._has_tracker_subclass(tree):
                return
            meta = self._extract_metadata_from_tree(tree)
            if meta is not None:
                self._register_community_lazy(meta, file_path, filename)
                return
            # Non-literal metadata: eager load through the sandbox.
            tracker_class = load_tracker_safely(file_path, filename)
            if tracker_class:
                self._register_tracker(tracker_class, "community", True, file_path)
        except TrackerSecurityError as e:
            err = f"SECURITY VIOLATION in {filename}: {e}"
            self._discovery_errors.append(err)
            self.logger.error(err)
        except Exception as e:
            err = f"Failed to load community tracker {filename}: {e}"
            self._discovery_errors.append(err)
            self.logger.warning(err)

    # ----- AST helpers (unchanged from prior revisions) -----

    @staticmethod
    def _parse_file(file_path: str):
        try:
            with open(file_path, "rb") as f:
                return ast.parse(f.read(), filename=file_path)
        except (OSError, SyntaxError):
            return None

    @staticmethod
    def _has_tracker_subclass(tree) -> bool:
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

    # ----- Lazy registration -----

    def _register_builtin_lazy(self, meta: TrackerMetadata, dotted: str, folder: str) -> None:
        if not isinstance(meta, TrackerMetadata) or not meta.name:
            return
        if self._name_conflict(meta):
            return
        self._metadata_cache[meta.name] = meta
        self._folder_map[meta.name] = folder
        self._builtin_refs[meta.name] = (dotted, folder)

    def _register_community_lazy(self, meta: TrackerMetadata, file_path: str, filename: str) -> None:
        if not isinstance(meta, TrackerMetadata) or not meta.name:
            return
        if self._name_conflict(meta):
            return
        self._metadata_cache[meta.name] = meta
        self._folder_map[meta.name] = "community"
        self._community_refs[meta.name] = (file_path, filename, "community")

    def _name_conflict(self, meta: TrackerMetadata) -> bool:
        if (meta.name in self._trackers or meta.name in self._builtin_refs
                or meta.name in self._community_refs):
            existing = self._metadata_cache.get(meta.name)
            self.logger.warning(
                f"Tracker name conflict: '{meta.name}' "
                f"(existing: {existing.display_name if existing else '?'}, "
                f"new: {meta.display_name}). Skipping new tracker."
            )
            return True
        return False

    # ----- Materialisation -----

    def _materialize(self, name: str) -> Optional[Type]:
        with self._discovery_lock:
            if name in self._trackers:
                return self._trackers[name]

            if name in self._builtin_refs:
                dotted, _folder = self._builtin_refs[name]
                try:
                    tracker_class = self._import_and_find_class(dotted)
                except Exception as e:
                    self.logger.error(f"Failed to materialize built-in {name}: {e}")
                    return None

            elif name in self._community_refs:
                file_path, filename, _folder = self._community_refs[name]
                try:
                    tracker_class = load_tracker_safely(file_path, filename)
                except TrackerSecurityError as e:
                    self.logger.error(f"SECURITY VIOLATION materializing {name}: {e}")
                    return None
                except Exception as e:
                    self.logger.error(f"Failed to materialize community {name}: {e}")
                    return None
            else:
                return None

            if tracker_class is None:
                return None
            self._trackers[name] = tracker_class
            # Refresh metadata from the live instance -- picks up anything the
            # static parse couldn't see (e.g. computed properties).
            try:
                live_meta = tracker_class().metadata
                if isinstance(live_meta, TrackerMetadata) and live_meta.name == name:
                    self._metadata_cache[name] = live_meta
            except Exception:
                pass
            return tracker_class

    def _register_tracker(self, tracker_class: Type, folder_name: str,
                          is_community: bool, source: str) -> None:
        """Eagerly register a tracker class that was imported during discovery."""
        temp_instance = None
        try:
            temp_instance = tracker_class()
            metadata = temp_instance.metadata

            if not isinstance(metadata, TrackerMetadata):
                raise TrackerValidationError("Tracker metadata must be TrackerMetadata instance")
            if not metadata.name:
                raise TrackerValidationError("Tracker name cannot be empty")

            if metadata.name in self._trackers:
                existing = self._metadata_cache.get(metadata.name)
                self.logger.warning(
                    f"Tracker name conflict: '{metadata.name}' "
                    f"(existing: {existing.display_name if existing else '?'}, "
                    f"new: {metadata.display_name}). Skipping new tracker."
                )
                return

            self._trackers[metadata.name] = tracker_class
            self._metadata_cache[metadata.name] = metadata
            self._folder_map[metadata.name] = folder_name

            category_prefix = "[Community] " if is_community else ""
            self.logger.debug(
                f"Registered: {category_prefix}{metadata.display_name} ({metadata.name})")

        except TrackerSecurityError as e:
            err = f"SECURITY VIOLATION registering {tracker_class.__name__}: {e}"
            self._discovery_errors.append(err)
            self.logger.error(err)
        except Exception as e:
            err = f"Failed to register tracker {tracker_class.__name__}: {e}"
            self._discovery_errors.append(err)
            self.logger.error(err)
        finally:
            if temp_instance:
                try:
                    if hasattr(temp_instance, "cleanup"):
                        temp_instance.cleanup()
                    temp_instance = None
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp instance: {cleanup_error}")

    # ----- Public API -----

    def get_tracker(self, name: str) -> Optional[Type]:
        self._ensure_discovered()
        cls = self._trackers.get(name)
        if cls is not None:
            return cls
        if name in self._builtin_refs or name in self._community_refs:
            return self._materialize(name)
        return None

    def create_tracker(self, name: str):
        tracker_class = self.get_tracker(name)
        if tracker_class:
            try:
                return tracker_class()
            except Exception as e:
                self.logger.error(f"Failed to create tracker instance {name}: {e}")
                return None
        return None

    def list_trackers(self, category: Optional[str] = None) -> List[TrackerMetadata]:
        self._ensure_discovered()
        trackers = list(self._metadata_cache.values())
        if category:
            trackers = [t for t in trackers if t.category == category]
        category_priority = {"live": 1, "offline": 2, "community": 3}
        return sorted(trackers, key=lambda t: (category_priority.get(t.category, 999), t.display_name))

    def get_metadata(self, name: str) -> Optional[TrackerMetadata]:
        self._ensure_discovered()
        return self._metadata_cache.get(name)

    def get_available_names(self) -> List[str]:
        self._ensure_discovered()
        names = set(self._trackers.keys())
        names.update(self._builtin_refs.keys())
        names.update(self._community_refs.keys())
        return list(names)

    def get_discovery_errors(self) -> List[str]:
        self._ensure_discovered()
        return self._discovery_errors.copy()

    def get_tracker_folder(self, name: str) -> Optional[str]:
        self._ensure_discovered()
        return self._folder_map.get(name)

    def reload_trackers(self):
        self.logger.debug("Reloading trackers...")
        self._trackers.clear()
        self._metadata_cache.clear()
        self._folder_map.clear()
        self._builtin_refs.clear()
        self._community_refs.clear()
        self._discovery_errors.clear()
        self._discover_trackers()


# Global registry instance -- automatically discovers trackers on import
tracker_registry = TrackerRegistry()


def get_tracker_registry() -> TrackerRegistry:
    return tracker_registry


def list_available_trackers(category: Optional[str] = None) -> List[TrackerMetadata]:
    return tracker_registry.list_trackers(category)


def create_tracker(name: str):
    return tracker_registry.create_tracker(name)


__all__ = [
    "TrackerRegistry",
    "tracker_registry",
    "BaseTracker",
    "BaseOfflineTracker",
    "TrackerMetadata",
    "TrackerResult",
    "TrackerError",
    "TrackerSecurityError",
    "TrackerValidationError",
    "TrackerSandboxError",
    "TrackerAPIViolationError",
    "get_tracker_registry",
    "list_available_trackers",
    "create_tracker",
]
