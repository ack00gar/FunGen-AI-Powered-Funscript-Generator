"""
Unified addon status registry.

Each addon publishes a single AddonStatus snapshot to the global registry.
The UI reads `get_all_statuses()` once per frame and renders them in a single
panel so users do not have to hunt through three different tabs to see whether
the streamer is up, the device is connected, or a subtitle job is running.

Designed to be cheap to call -- snapshots are plain dicts, no locks held
across the read.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AddonStatus:
    """Single-line view of an addon's runtime state."""
    addon: str                       # e.g. "device_control", "streamer", "subtitle"
    state: str                       # ServiceState.value (stopped/starting/running/stopping/error)
    summary: str = ""                # One-line status, shown next to the badge
    detail: str = ""                 # Multi-line tooltip / expander content
    progress: Optional[float] = None # 0..1 if this addon is doing a long task
    error: Optional[str] = None      # Last error message, if any
    timestamp: float = field(default_factory=time.time)
    extras: Dict[str, Any] = field(default_factory=dict)


_LOCK = threading.Lock()
_PROVIDERS: Dict[str, Callable[[], Optional[AddonStatus]]] = {}


def register_status_provider(addon: str, provider: Callable[[], Optional[AddonStatus]]) -> None:
    """Register or replace the status provider for an addon."""
    with _LOCK:
        _PROVIDERS[addon] = provider


def unregister_status_provider(addon: str) -> None:
    with _LOCK:
        _PROVIDERS.pop(addon, None)


def get_all_statuses() -> List[AddonStatus]:
    """Snapshot every registered addon's current status. Stable order."""
    with _LOCK:
        items = list(_PROVIDERS.items())
    out: List[AddonStatus] = []
    for addon, provider in sorted(items, key=lambda kv: kv[0]):
        try:
            st = provider()
        except Exception as e:
            st = AddonStatus(addon=addon, state="error", summary="provider raised", error=str(e))
        if st is not None:
            out.append(st)
    return out


def get_status(addon: str) -> Optional[AddonStatus]:
    """Snapshot a single addon. Returns None if no provider is registered."""
    with _LOCK:
        provider = _PROVIDERS.get(addon)
    if provider is None:
        return None
    try:
        return provider()
    except Exception as e:
        return AddonStatus(addon=addon, state="error", summary="provider raised", error=str(e))
