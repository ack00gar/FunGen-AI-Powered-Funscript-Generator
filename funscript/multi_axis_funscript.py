import numpy as np
from typing import Optional, Callable, List, Tuple, Dict, Any
import logging
import bisect
import copy

from common.frame_utils import ms_to_frame, frame_to_ms

# Attempt to import optional libraries for processing
try:
    from scipy.signal import savgol_filter, find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from rdp import rdp

    RDP_AVAILABLE = True
except ImportError:
    RDP_AVAILABLE = False


class MultiAxisFunscript:
    def __init__(self, logger: Optional[logging.Logger] = None, fps: Optional[float] = None):
        self.primary_actions: List[Dict] = []
        self.secondary_actions: List[Dict] = []
        self.chapters: List[Dict] = []  # Funscript chapters/segments
        self.min_interval_ms: int = 10
        self._fps: Optional[float] = fps if fps and fps > 0 else None
        self.last_timestamp_primary: int = 0
        self.last_timestamp_secondary: int = 0

        # Timestamp caching mechanism
        self._primary_timestamps_cache: List[int] = []
        self._secondary_timestamps_cache: List[int] = []
        self._cache_dirty_primary: bool = True
        self._cache_dirty_secondary: bool = True
        # Numpy array caches for timeline drawing (invalidated with timestamps)
        self._primary_np_cache = None    # (ats_f32, poss_f32) or None
        self._secondary_np_cache = None

        # Parallel int64/uint8 array caches kept in lockstep with the action
        # lists. Built lazily on first access, invalidated whenever
        # _invalidate_cache is called. Hot paths can avoid Python-level dict
        # iteration by using get_arrays / bisect_at / range_indices /
        # get_values_at_times.
        # _pa_times[axis] / _pa_values[axis] are exposed to readers as
        # numpy VIEWS of these oversized buffers (cap ≥ length). Append
        # writes into the unused tail and slices a new view, so live tracking
        # doesn't rebuild the whole array on every append/read cycle.
        self._pa_times: Dict[str, Optional[np.ndarray]] = {}
        self._pa_values: Dict[str, Optional[np.ndarray]] = {}
        self._pa_buf_t: Dict[str, np.ndarray] = {}
        self._pa_buf_v: Dict[str, np.ndarray] = {}

        # Additional axes for multi-timeline (supporter feature)
        self.additional_axes: Dict[str, List[Dict]] = {}
        self._additional_timestamps_cache: Dict[str, List[int]] = {}
        self._additional_cache_dirty: Dict[str, bool] = {}
        self._additional_np_cache: Dict[str, tuple] = {}
        self._additional_last_timestamps: Dict[str, int] = {}

        # Timeline-to-axis assignment mapping (timeline_num -> axis_name string)
        # Tells the file manager which suffix to use, and the device controller which TCode channel.
        self._axis_assignments: Dict[int, str] = {1: "stroke", 2: "roll"}

        # get_value bracket cache: during playback time_ms advances roughly
        # monotonically, so the bisect we'd normally do for every call is
        # usually bracketed by the same action pair as the previous call. Cache
        # the last (axis, idx) pair; on cache hit we skip bisect entirely.
        self._gv_cache_axis: Optional[str] = None
        self._gv_cache_idx: int = 0
        self._gv_cache_n: int = 0  # len snapshot; invalidate on mismatch

        # Point simplification settings.
        # tolerance 2 keeps error ≤ 1 pos unit (under device resolution) while
        # dropping ~1.7x more points than tolerance 1 on typical tracker output.
        self.enable_point_simplification: bool = True
        self.simplification_tolerance: int = 2

        # Point simplification statistics
        self._simplification_stats_primary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
        self._simplification_stats_secondary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
        self._last_simplification_log_time = 0
        self._simplification_log_interval_sec = 10  # Log every 10 seconds

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('MultiAxisFunscript_fallback')
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())

        # Composed helpers. Flat methods below are thin delegators for
        # backwards compatibility; new code should prefer fs.signal.<op>()
        # directly to skip one attribute lookup in hot paths.
        from funscript.signal_processor import SignalProcessor
        from funscript.plugin_controller import PluginController
        from funscript.action_editor import ActionEditor
        self.signal = SignalProcessor(self)
        self.plugins = PluginController(self)
        self.editor = ActionEditor(self)

    @property
    def fps(self) -> Optional[float]:
        return self._fps

    @fps.setter
    def fps(self, value: Optional[float]):
        self._fps = value if value and value > 0 else None

    def snap_to_frame(self, timestamp_ms: int) -> int:
        """Snap a timestamp to the nearest frame boundary.
        Ensures max 1 action per frame. When fps is not set, returns as-is."""
        if self._fps is None:
            return timestamp_ms
        return frame_to_ms(ms_to_frame(timestamp_ms, self._fps), self._fps)

    def _invalidate_cache(self, axis: str = 'both'):
        """Marks the timestamp cache(s) as dirty."""
        if axis == 'primary' or axis == 'both':
            self._cache_dirty_primary = True
            self._drop_pa('primary')
        if axis == 'secondary' or axis == 'both':
            self._cache_dirty_secondary = True
            self._drop_pa('secondary')
        if axis in self._additional_cache_dirty:
            self._additional_cache_dirty[axis] = True
            self._drop_pa(axis)
        if axis == 'both':
            for ax_name in self._additional_cache_dirty:
                self._additional_cache_dirty[ax_name] = True
                self._drop_pa(ax_name)
        # Clear get_value bracket cache too — stale idx after mutation is wrong.
        self._gv_cache_n = 0

    def _drop_pa(self, axis: str) -> None:
        """Drop the parallel-array view AND underlying buffer for `axis`."""
        self._pa_times.pop(axis, None)
        self._pa_values.pop(axis, None)
        self._pa_buf_t.pop(axis, None)
        self._pa_buf_v.pop(axis, None)

    def _pa_append(self, axis: str, t_val: int, v_val: int) -> None:
        """O(1)-amortized append to the parallel arrays for `axis`.
        Extends the cached view in place when capacity permits; doubles the
        backing buffer otherwise. No-op when the cache isn't populated
        (get_arrays rebuilds lazily on next read)."""
        view_t = self._pa_times.get(axis)
        if view_t is None:
            return
        n = view_t.shape[0]
        buf_t = self._pa_buf_t.get(axis)
        buf_v = self._pa_buf_v.get(axis)
        if buf_t is None or buf_v is None or buf_t.shape[0] < n + 1:
            new_cap = max(16, 1 << n.bit_length()) if n > 0 else 16
            nt = np.empty(new_cap, dtype=np.int64)
            nv = np.empty(new_cap, dtype=np.uint8)
            if n > 0:
                nt[:n] = view_t
                nv[:n] = self._pa_values[axis]
            buf_t = nt
            buf_v = nv
            self._pa_buf_t[axis] = buf_t
            self._pa_buf_v[axis] = buf_v
        buf_t[n] = t_val
        buf_v[n] = v_val
        self._pa_times[axis] = buf_t[:n + 1]
        self._pa_values[axis] = buf_v[:n + 1]

    def _patch_cache_entry(self, axis: str, idx: int, at: int, pos: int) -> bool:
        """O(1) in-place cache update. Returns False if caches aren't ready."""
        if axis == 'primary':
            if not self._cache_dirty_primary and self._primary_timestamps_cache is not None:
                if 0 <= idx < len(self._primary_timestamps_cache):
                    self._primary_timestamps_cache[idx] = at
                else:
                    return False
            else:
                return False
        elif axis == 'secondary':
            if not self._cache_dirty_secondary and self._secondary_timestamps_cache is not None:
                if 0 <= idx < len(self._secondary_timestamps_cache):
                    self._secondary_timestamps_cache[idx] = at
                else:
                    return False
            else:
                return False
        else:
            return False

        # float32 numpy cache used by the timeline
        np_cache = self._primary_np_cache if axis == 'primary' else self._secondary_np_cache
        if np_cache is not None:
            ats_np, poss_np = np_cache
            if 0 <= idx < ats_np.shape[0]:
                ats_np[idx] = at
                poss_np[idx] = pos

        pa_t = self._pa_times.get(axis)
        pa_v = self._pa_values.get(axis)
        if pa_t is not None and pa_v is not None:
            if 0 <= idx < pa_t.shape[0]:
                pa_t[idx] = at
                pa_v[idx] = pos

        self._gv_cache_n = 0
        return True

    # ----- Parallel array API -----
    # Fast-path access for hot loops. Hand back numpy arrays in lockstep with
    # primary_actions / secondary_actions / additional_axes; mutate via the
    # action dicts as before, the arrays rebuild lazily on next read.

    def _actions_for_axis(self, axis: str) -> List[Dict]:
        if axis == 'primary':
            return self.primary_actions
        if axis == 'secondary':
            return self.secondary_actions
        return self.additional_axes.get(axis, [])

    def get_arrays(self, axis: str = 'primary'):
        """Return (times_ms_int64, values_uint8) numpy arrays for `axis`.

        Cached and rebuilt only when _invalidate_cache is called for `axis`.
        Returned arrays are views — do NOT mutate them; mutate primary_actions
        instead and the arrays will rebuild on next access.
        """
        cached_t = self._pa_times.get(axis)
        cached_v = self._pa_values.get(axis)
        src = self._actions_for_axis(axis)
        n = len(src)
        # Guard: in-place dict mutation that changes the action count without
        # going through the helper APIs would still be detected here.
        if cached_t is not None and cached_v is not None and cached_t.shape[0] == n:
            return cached_t, cached_v
        if n == 0:
            self._pa_buf_t.pop(axis, None)
            self._pa_buf_v.pop(axis, None)
            t = np.empty(0, dtype=np.int64)
            v = np.empty(0, dtype=np.uint8)
        else:
            # Allocate with headroom so subsequent _pa_append calls can fill
            # the tail without a realloc until the next power-of-2 is hit.
            cap = max(16, 1 << (n - 1).bit_length())
            buf_t = np.empty(cap, dtype=np.int64)
            buf_v = np.empty(cap, dtype=np.uint8)
            buf_t[:n] = np.fromiter((a['at'] for a in src), dtype=np.int64, count=n)
            buf_v[:n] = np.fromiter((a['pos'] for a in src), dtype=np.uint8, count=n)
            self._pa_buf_t[axis] = buf_t
            self._pa_buf_v[axis] = buf_v
            t = buf_t[:n]
            v = buf_v[:n]
        self._pa_times[axis] = t
        self._pa_values[axis] = v
        return t, v

    def bisect_at(self, axis: str, time_ms, side: str = 'left') -> int:
        """np.searchsorted wrapper. side='left' or 'right'."""
        t, _ = self.get_arrays(axis)
        return int(np.searchsorted(t, time_ms, side=side))

    def range_indices(self, axis: str, t0_ms, t1_ms):
        """(lo, hi) such that times[lo:hi] is the actions in [t0_ms, t1_ms]."""
        t, _ = self.get_arrays(axis)
        return (int(np.searchsorted(t, t0_ms, side='left')),
                int(np.searchsorted(t, t1_ms, side='right')))

    def get_values_at_times(self, times_ms, axis: str = 'primary') -> np.ndarray:
        """Vectorized linear interpolation. Outside-range clamps to first/last."""
        t, v = self.get_arrays(axis)
        times_ms = np.asarray(times_ms, dtype=np.int64)
        if t.size == 0:
            return np.full(times_ms.shape, 50.0, dtype=np.float32)
        if t.size == 1:
            return np.full(times_ms.shape, float(v[0]), dtype=np.float32)
        return np.interp(times_ms, t, v).astype(np.float32)

    def mark_actions_dirty(self, axis: str = 'both'):
        """Public hook for callers that mutated dicts in place (e.g. plugin
        modifying actions[i]['pos'] without changing 'at'). Invalidates all
        caches so next read sees fresh data."""
        self._invalidate_cache(axis)

    def _append_to_cache(self, axis_name: str, timestamp_ms: int, pos: Optional[int] = None):
        """Append a single timestamp to the caches without rebuilding.
        `pos` enables the parallel-array fast path; omit only from code
        paths that can't supply it (the PA cache then rebuilds on next read)."""
        if axis_name == 'primary':
            if not self._cache_dirty_primary:
                self._primary_timestamps_cache.append(timestamp_ms)
                self._primary_np_cache = None  # Invalidate numpy cache so heatmap rebuilds
        elif axis_name == 'secondary':
            if not self._cache_dirty_secondary:
                self._secondary_timestamps_cache.append(timestamp_ms)
                self._secondary_np_cache = None
        elif axis_name in self._additional_timestamps_cache:
            if not self._additional_cache_dirty.get(axis_name, True):
                self._additional_timestamps_cache[axis_name].append(timestamp_ms)
                self._additional_np_cache.pop(axis_name, None)
        if pos is None:
            self._drop_pa(axis_name)
        else:
            self._pa_append(axis_name, timestamp_ms, pos)

    def _pop_from_cache(self, axis_name: str, index: int):
        """Remove an entry from the timestamp cache at the given index."""
        if axis_name == 'primary' and not self._cache_dirty_primary:
            if self._primary_timestamps_cache:
                self._primary_timestamps_cache.pop(index)
                self._primary_np_cache = None
        elif axis_name == 'secondary' and not self._cache_dirty_secondary:
            if self._secondary_timestamps_cache:
                self._secondary_timestamps_cache.pop(index)
                self._secondary_np_cache = None
        elif axis_name in self._additional_timestamps_cache:
            if not self._additional_cache_dirty.get(axis_name, True):
                cache = self._additional_timestamps_cache[axis_name]
                if cache:
                    cache.pop(index)
                    self._additional_np_cache.pop(axis_name, None)
        # Parallel arrays: try to splice out `index` in place so live-tracker
        # simplification (which pops -2 on nearly every append when collinear)
        # doesn't force an O(N) rebuild on the next read.
        self._pa_pop(axis_name, index)

    def _pa_pop(self, axis: str, index: int) -> None:
        """Remove one element from the parallel-array views at `index`.
        Keeps the backing buffer, just shifts [index+1:] down by one and
        shortens the view. Falls back to a full drop for weird indices."""
        view_t = self._pa_times.get(axis)
        if view_t is None:
            return
        n = view_t.shape[0]
        if n == 0:
            return
        if index < 0:
            index += n
        if index < 0 or index >= n:
            self._drop_pa(axis)
            return
        buf_t = self._pa_buf_t.get(axis)
        buf_v = self._pa_buf_v.get(axis)
        if buf_t is None or buf_v is None:
            self._drop_pa(axis)
            return
        if index < n - 1:
            buf_t[index:n - 1] = buf_t[index + 1:n]
            buf_v[index:n - 1] = buf_v[index + 1:n]
        self._pa_times[axis] = buf_t[:n - 1]
        self._pa_values[axis] = buf_v[:n - 1]

    def _maybe_log_simplification_stats(self):
        """No-op: per-tick simplification logs were noisy. Final summary only."""
        return

    def _log_simplification_stats_internal(self):
        """Internal helper to log stats (called by periodic logger and final summary)."""
        # Log stats for primary axis if active
        stats_p = self._simplification_stats_primary
        if stats_p['total_considered'] > 0:
            reduction_pct = (stats_p['total_removed'] / stats_p['total_considered']) * 100
            current_points = len(self.primary_actions)
            would_have_been = current_points + stats_p['total_removed']

            # Calculate time window
            if current_points > 0 and stats_p['start_time_ms'] > 0:
                time_window_ms = self.primary_actions[-1]['at'] - stats_p['start_time_ms']
                time_window_sec = time_window_ms / 1000.0

                self.logger.info(
                    f"Point Simplification (Primary): {time_window_sec:.1f}s window, "
                    f"{stats_p['total_considered']:,} frames -> {stats_p['total_removed']:,} points removed ({reduction_pct:.1f}% reduction), "
                    f"{would_have_been:,} -> {current_points:,} points"
                )

        # Log stats for secondary axis if active
        stats_s = self._simplification_stats_secondary
        if stats_s['total_considered'] > 0:
            reduction_pct = (stats_s['total_removed'] / stats_s['total_considered']) * 100
            current_points = len(self.secondary_actions)
            would_have_been = current_points + stats_s['total_removed']

            # Calculate time window
            if current_points > 0 and stats_s['start_time_ms'] > 0:
                time_window_ms = self.secondary_actions[-1]['at'] - stats_s['start_time_ms']
                time_window_sec = time_window_ms / 1000.0

                self.logger.info(
                    f"Point Simplification (Secondary): {time_window_sec:.1f}s window, "
                    f"{stats_s['total_considered']:,} frames -> {stats_s['total_removed']:,} points removed ({reduction_pct:.1f}% reduction), "
                    f"{would_have_been:,} -> {current_points:,} points"
                )

    def log_final_simplification_summary(self):
        """
        Log final point simplification summary (called when tracking stops).
        Forces a log regardless of time interval.
        """
        # Force log if any simplification happened
        if (self._simplification_stats_primary['total_considered'] > 0 or
            self._simplification_stats_secondary['total_considered'] > 0):
            self.logger.info("Final Point Simplification Summary:")
            self._log_simplification_stats_internal()
            # Reset stats for next session
            self._simplification_stats_primary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
            self._simplification_stats_secondary = {'total_removed': 0, 'total_considered': 0, 'start_time_ms': 0}
            self._last_simplification_log_time = 0

    def _simplify_last_points(self, actions_list: List[Dict], axis: str = 'primary') -> None:
        """
        Per-frame simplification. Removes the middle of the last 3 points iff
        it is NOT a local extremum AND it is collinear within tolerance with
        its neighbors.

        Peak/valley timing is sacred — the whole meaning of a funscript is in
        the extrema, so we never drop them regardless of tolerance. Only
        monotonic intermediate points (same direction in, same direction out)
        are candidates for removal.
        """
        stats = self._simplification_stats_primary if axis == 'primary' else self._simplification_stats_secondary

        if len(actions_list) < 3:
            return

        if stats['start_time_ms'] == 0 and len(actions_list) >= 3:
            stats['start_time_ms'] = actions_list[-3]['at']
        stats['total_considered'] += 1

        p1 = actions_list[-3]
        p2 = actions_list[-2]
        p3 = actions_list[-1]
        pos1, pos2, pos3 = p1['pos'], p2['pos'], p3['pos']

        # Preserve extrema: if p2 is a strict local peak or valley, keep it.
        # Even weak extrema (e.g. 48-50-48) are preserved — the device timing
        # follows extrema, not amplitude.
        if (pos1 < pos2 and pos2 > pos3) or (pos1 > pos2 and pos2 < pos3):
            return

        # Flat triple — middle is redundant.
        if pos1 == pos2 == pos3:
            actions_list.pop(-2)
            self._pop_from_cache(axis, -2)
            stats['total_removed'] += 1
            self._maybe_log_simplification_stats()
            return

        # Monotonic intermediate: drop if the perpendicular distance to the
        # 1-3 line is within tolerance. |cross|/time_range == distance.
        t1, t2, t3 = p1['at'], p2['at'], p3['at']
        time_range = t3 - t1
        if time_range == 0:
            return

        cross = (t2 - t1) * (pos3 - pos1) - (t3 - t1) * (pos2 - pos1)
        if abs(cross) <= self.simplification_tolerance * time_range:
            actions_list.pop(-2)
            self._pop_from_cache(axis, -2)
            stats['total_removed'] += 1
            self._maybe_log_simplification_stats()

    def _get_timestamps_for_axis(self, axis: str) -> List[int]:
        """Cached timestamp list. Rebuilt via PA int64 array's tolist()
        which is one C-level conversion, ~10x faster than the per-dict
        list comp on big scripts."""
        if axis == 'primary':
            if self._cache_dirty_primary:
                t, _ = self.get_arrays('primary')
                self._primary_timestamps_cache = t.tolist()
                self._primary_np_cache = None
                self._cache_dirty_primary = False
            return self._primary_timestamps_cache
        elif axis == 'secondary':
            if self._cache_dirty_secondary:
                t, _ = self.get_arrays('secondary')
                self._secondary_timestamps_cache = t.tolist()
                self._secondary_np_cache = None
                self._cache_dirty_secondary = False
            return self._secondary_timestamps_cache
        elif axis in self.additional_axes:
            if self._additional_cache_dirty.get(axis, True):
                t, _ = self.get_arrays(axis)
                self._additional_timestamps_cache[axis] = t.tolist()
                self._additional_np_cache.pop(axis, None)
                self._additional_cache_dirty[axis] = False
            return self._additional_timestamps_cache.get(axis, [])
        return []

    def _get_numpy_arrays_for_axis(self, axis: str):
        """Return cached (ats_np, poss_np) float32 arrays via astype on PA cache."""
        actions = self.get_axis_actions(axis)
        if not actions:
            _empty = np.empty(0, dtype=np.float32)
            return _empty, _empty

        # Determine cache slot
        if axis == 'primary':
            self._get_timestamps_for_axis(axis)
            if self._primary_np_cache is not None:
                return self._primary_np_cache
            t, v = self.get_arrays(axis)
            ats = t.astype(np.float32)
            poss = v.astype(np.float32)
            self._primary_np_cache = (ats, poss)
            return ats, poss
        elif axis == 'secondary':
            self._get_timestamps_for_axis(axis)
            if self._secondary_np_cache is not None:
                return self._secondary_np_cache
            t, v = self.get_arrays(axis)
            ats = t.astype(np.float32)
            poss = v.astype(np.float32)
            self._secondary_np_cache = (ats, poss)
            return ats, poss
        else:
            self._get_timestamps_for_axis(axis)
            cached = self._additional_np_cache.get(axis)
            if cached is not None:
                return cached
            t, v = self.get_arrays(axis)
            ats = t.astype(np.float32)
            poss = v.astype(np.float32)
            self._additional_np_cache[axis] = (ats, poss)
            return ats, poss

    def _process_action_for_axis(self,
                                 actions_target_list: List[Dict],
                                 timestamp_ms: int,
                                 pos: int,
                                 min_interval_ms: int,
                                 axis_name: str # 'primary' or 'secondary'
                                 ) -> int:
        """
        Processes and adds/updates a single action in the target list (in-place).
        Uses a fast O(1) path for chronological appends (the 99%+ tracker case)
        and falls back to bisect for out-of-order insertions (manual editing).
        Returns the timestamp of the last action in the list.
        """
        timestamp_ms = self.snap_to_frame(timestamp_ms)
        clamped_pos = max(0, min(100, pos))
        new_action = {"at": timestamp_ms, "pos": clamped_pos}

        # === FAST PATH: chronological append (covers 99%+ of tracker calls) ===
        if actions_target_list:
            last = actions_target_list[-1]
            if timestamp_ms > last["at"]:
                # New point is strictly after all existing points
                if timestamp_ms - last["at"] >= min_interval_ms:
                    actions_target_list.append(new_action)  # O(1)
                    self._append_to_cache(axis_name, timestamp_ms, clamped_pos)
                    if self.enable_point_simplification:
                        self._simplify_last_points(actions_target_list, axis=axis_name)
                    return timestamp_ms
                else:
                    # Too close to last point — skip
                    return last["at"]
            elif timestamp_ms == last["at"]:
                # Same timestamp as last — update in place
                if last["pos"] != clamped_pos:
                    last["pos"] = clamped_pos
                    # Timestamp cache stays valid; patch the last pa value.
                    pa_v = self._pa_values.get(axis_name)
                    if pa_v is not None and pa_v.shape[0] > 0:
                        pa_v[pa_v.shape[0] - 1] = clamped_pos
                return timestamp_ms
        else:
            # Empty list — just append
            actions_target_list.append(new_action)
            self._append_to_cache(axis_name, timestamp_ms, clamped_pos)
            return timestamp_ms

        # === SLOW PATH: out-of-order insertion (manual editing) ===
        # Use the cached timestamps for bisect lookup
        action_timestamps = self._get_timestamps_for_axis(axis_name)
        idx = bisect.bisect_left(action_timestamps, timestamp_ms)

        # Guard: clamp idx to actions list bounds (cache may be stale)
        idx = min(idx, len(actions_target_list))

        action_inserted_or_updated = False
        if idx < len(actions_target_list) and actions_target_list[idx]["at"] == timestamp_ms:
            if actions_target_list[idx]["pos"] != clamped_pos:
                actions_target_list[idx]["pos"] = clamped_pos
                # No timestamp change, so cache is still valid
        else:
            can_insert = True
            if idx > 0 and len(actions_target_list) > 0:
                prev_action = actions_target_list[min(idx - 1, len(actions_target_list) - 1)]
                if timestamp_ms - prev_action["at"] < min_interval_ms:
                    can_insert = False

            if can_insert:
                actions_target_list.insert(idx, new_action)
                action_inserted_or_updated = True
                self._invalidate_cache(axis_name) # Cache is now dirty

                # Apply lightweight point simplification after insertion
                if self.enable_point_simplification:
                    self._simplify_last_points(actions_target_list, axis=axis_name)

        if action_inserted_or_updated and min_interval_ms > 0:
            if not actions_target_list:
                return 0

            original_len = len(actions_target_list)
            current_valid_idx = 0
            if len(actions_target_list) > 1:
                for i in range(1, len(actions_target_list)):
                    if actions_target_list[i]["at"] - actions_target_list[current_valid_idx]["at"] >= min_interval_ms:
                        current_valid_idx += 1
                        if i != current_valid_idx:
                            actions_target_list[current_valid_idx] = actions_target_list[i]

            if current_valid_idx + 1 < len(actions_target_list):
                del actions_target_list[current_valid_idx + 1:]

            # If filtering removed points, invalidate the cache again
            if len(actions_target_list) != original_len:
                self._invalidate_cache(axis_name)

        return actions_target_list[-1]["at"] if actions_target_list else 0

    def add_action(self, timestamp_ms: int, primary_pos: Optional[int], secondary_pos: Optional[int] = None,
                   is_from_live_tracker: bool = True):
        """
        Adds an action for primary axis and optionally for secondary axis.
        :param timestamp_ms: The timestamp of the action in milliseconds.
        :param primary_pos: The position for the primary axis (0-100). Can be None.
        :param secondary_pos: Optional. The position for the secondary axis (0-100). Can be None.
        :param is_from_live_tracker: True if this action originates from live tracking,
                                     influencing max_history application. False for programmatic
                                     additions (e.g. file load, undo/redo) where max_history
                                     might not be desired for the loaded portion.
        """
        new_last_ts_primary = self.last_timestamp_primary
        if primary_pos is not None:
            new_last_ts_primary = self._process_action_for_axis(
                actions_target_list=self.primary_actions,
                timestamp_ms=timestamp_ms,
                pos=primary_pos,
                min_interval_ms=self.min_interval_ms,
                axis_name='primary' # Pass axis name
            )
        # Update last_timestamp_primary only if actions were actually processed or if list became empty
        self.last_timestamp_primary = new_last_ts_primary if self.primary_actions else 0


        new_last_ts_secondary = self.last_timestamp_secondary
        if secondary_pos is not None:
            new_last_ts_secondary = self._process_action_for_axis(
                actions_target_list=self.secondary_actions,
                timestamp_ms=timestamp_ms,
                pos=secondary_pos,
                min_interval_ms=self.min_interval_ms,
                axis_name='secondary' # Pass axis name
            )
            self.last_timestamp_secondary = new_last_ts_secondary if self.secondary_actions else 0

    def reset_to_neutral(self, timestamp_ms: int):
        self.add_action(timestamp_ms, 100, 50, is_from_live_tracker=True)

    @staticmethod
    def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        """Catmull-Rom spline interpolation between p1 and p2 with t in [0,1]."""
        return 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
        )

    def get_value(self, time_ms: int, axis: str = 'primary',
                  interpolation: str = 'linear') -> int:
        """
        Returns the interpolated position value at a given timestamp.
        Uses the cached timestamp list for O(1) amortised bisect lookups.

        Args:
            interpolation: 'linear' (default) or 'spline' (catmull-rom, smoother at peaks)
        """
        if axis == 'primary':
            actions_list = self.primary_actions
        elif axis == 'secondary':
            actions_list = self.secondary_actions
        else:
            actions_list = self.additional_axes.get(axis, [])

        if not actions_list:
            return 50  # Default neutral position

        # Use the cached timestamp list, rebuilt only when dirty.
        action_timestamps = self._get_timestamps_for_axis(axis)

        # Bracket cache: during playback time_ms advances monotonically so the
        # same bracket pair often holds between calls. A 3-point probe around
        # the cached idx beats a full bisect for the common case. Falls back
        # to bisect on miss (identical behavior, no correctness risk).
        n = len(action_timestamps)
        if (self._gv_cache_axis == axis and self._gv_cache_n == n
                and 0 < self._gv_cache_idx < n):
            ci = self._gv_cache_idx
            if action_timestamps[ci - 1] <= time_ms < action_timestamps[ci]:
                idx = ci
            elif ci + 1 < n and action_timestamps[ci] <= time_ms < action_timestamps[ci + 1]:
                idx = ci + 1
            else:
                idx = bisect.bisect_left(action_timestamps, time_ms)
        else:
            idx = bisect.bisect_left(action_timestamps, time_ms)
        self._gv_cache_axis = axis
        self._gv_cache_idx = idx
        self._gv_cache_n = n

        if idx == 0:
            return actions_list[0]["pos"]
        if idx >= len(actions_list):
            return actions_list[-1]["pos"]

        p1 = actions_list[idx - 1]
        p2 = actions_list[idx]

        if time_ms == p1["at"]:
            return p1["pos"]

        time_diff = float(p2["at"] - p1["at"])
        if time_diff == 0:
            return p1["pos"]

        t = (time_ms - p1["at"]) / time_diff

        if interpolation == 'spline' and len(actions_list) >= 3:
            # Catmull-rom needs 4 control points: p0, p1, p2, p3
            # Clamp at boundaries (duplicate endpoints)
            i0 = max(0, idx - 2)
            i3 = min(len(actions_list) - 1, idx + 1)
            val = self._catmull_rom(
                actions_list[i0]["pos"],
                p1["pos"],
                p2["pos"],
                actions_list[i3]["pos"],
                t
            )
        else:
            val = p1["pos"] + t * (p2["pos"] - p1["pos"])

        return int(round(np.clip(val, 0, 100)))

    def get_latest_value(self, axis: str = 'primary') -> int:
        if axis == 'primary':
            actions_list = self.primary_actions
        elif axis == 'secondary':
            actions_list = self.secondary_actions
        else:
            actions_list = self.additional_axes.get(axis, [])
        if actions_list:
            return actions_list[-1]["pos"]
        return 50

    def clear(self):
        self.primary_actions = []
        self.secondary_actions = []
        self.last_timestamp_primary = 0
        self.last_timestamp_secondary = 0
        # Clear additional axes
        for axis_name in self.additional_axes:
            self.additional_axes[axis_name].clear()
        self._additional_last_timestamps = {k: 0 for k in self._additional_last_timestamps}
        self._invalidate_cache('both')
        self.logger.info("Cleared all actions from MultiAxisFunscript.")

    # ==================================================================================
    # Multi-axis support (additional axes beyond primary/secondary)
    # ==================================================================================

    def ensure_axis(self, axis_name: str) -> None:
        """Create storage for an additional axis if it doesn't already exist. Idempotent."""
        if axis_name in ('primary', 'secondary'):
            return  # Built-in axes always exist
        if axis_name not in self.additional_axes:
            self.additional_axes[axis_name] = []
            self._additional_timestamps_cache[axis_name] = []
            self._additional_cache_dirty[axis_name] = True
            self._additional_last_timestamps[axis_name] = 0

    def get_axis_actions(self, axis_name: str) -> List[Dict]:
        """Unified accessor for any axis's action list."""
        if axis_name == 'primary':
            return self.primary_actions
        elif axis_name == 'secondary':
            return self.secondary_actions
        elif axis_name in self.additional_axes:
            return self.additional_axes[axis_name]
        return []

    def set_axis_actions(self, axis_name: str, actions: list) -> None:
        """Replace all actions on a given axis with the provided list.

        Handles primary, secondary, and additional axes.
        Invalidates timestamp/numpy/parallel-array caches so readers
        (timeline lines, get_value, gauge, 3D simulator) see the new data.
        """
        if axis_name == 'primary':
            self.primary_actions.clear()
            self.primary_actions.extend(actions)
        elif axis_name == 'secondary':
            self.secondary_actions.clear()
            self.secondary_actions.extend(actions)
        else:
            self.ensure_axis(axis_name)
            self.additional_axes[axis_name] = list(actions)
        self._invalidate_cache(axis_name)

    def add_action_to_axis(self, axis_name: str, timestamp_ms: int, pos: int) -> None:
        """Add an action to any axis (primary, secondary, or additional)."""
        if axis_name == 'primary':
            self.add_action(timestamp_ms=timestamp_ms, primary_pos=pos, secondary_pos=None)
            return
        elif axis_name == 'secondary':
            self.add_action(timestamp_ms=timestamp_ms, primary_pos=None, secondary_pos=pos)
            return

        # Additional axis
        self.ensure_axis(axis_name)
        last_ts = self._additional_last_timestamps.get(axis_name, 0)
        new_last_ts = self._process_action_for_axis(
            actions_target_list=self.additional_axes[axis_name],
            timestamp_ms=timestamp_ms,
            pos=pos,
            min_interval_ms=self.min_interval_ms,
            axis_name=axis_name
        )
        self._additional_last_timestamps[axis_name] = new_last_ts if self.additional_axes[axis_name] else 0

    def get_axis_count(self) -> int:
        """Returns total number of axes (2 built-in + additional)."""
        return 2 + len(self.additional_axes)

    def get_all_axis_names(self) -> List[str]:
        """Returns list of all axis names in order."""
        return ['primary', 'secondary'] + sorted(self.additional_axes.keys())

    def clear_axis(self, axis_name: str) -> None:
        """Clear a specific additional axis."""
        if axis_name in self.additional_axes:
            self.additional_axes[axis_name].clear()
            self._additional_last_timestamps[axis_name] = 0
            self._invalidate_cache(axis_name)

    # ==================================================================================
    # Axis assignment mapping (timeline_num <-> semantic axis name)
    # ==================================================================================

    def assign_axis(self, timeline_num: int, axis_name: str) -> None:
        """Set which semantic axis a timeline represents (e.g. timeline 3 -> 'pitch')."""
        self._axis_assignments[timeline_num] = axis_name

    def get_axis_for_timeline(self, timeline_num: int) -> str:
        """Return the assigned axis name for a timeline, with sensible defaults."""
        return self._axis_assignments.get(timeline_num, f"axis_{timeline_num}")

    def get_timeline_for_axis(self, axis_name: str) -> Optional[int]:
        """Reverse lookup: find which timeline is assigned to a given axis name."""
        for tl_num, name in self._axis_assignments.items():
            if name == axis_name:
                return tl_num
        return None

    def get_axis_assignments(self) -> Dict[int, str]:
        """Return a copy of the current timeline-to-axis assignments."""
        return dict(self._axis_assignments)

    # ==================================================================================
    # Serialization (project save/load)
    # ==================================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all axes to a dict for project storage.

        The action lists are passed by reference; orjson does not mutate.
        """
        axes_data = {
            "primary": self.primary_actions,
            "secondary": self.secondary_actions,
        }
        for name, actions in self.additional_axes.items():
            axes_data[name] = actions

        result: Dict[str, Any] = {
            "axes": axes_data,
            "axis_assignments": {str(k): v for k, v in self._axis_assignments.items()},
        }
        if self.chapters:
            result["chapters"] = [
                c.to_dict() if hasattr(c, 'to_dict') else dict(c)
                for c in self.chapters
            ]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], logger=None) -> "MultiAxisFunscript":
        """Reconstruct from serialized dict."""
        obj = cls(logger=logger)
        axes = data.get("axes", {})
        obj.primary_actions = axes.get("primary", [])
        obj.secondary_actions = axes.get("secondary", [])
        for name, actions in axes.items():
            if name not in ("primary", "secondary"):
                obj.additional_axes[name] = actions
                obj._additional_timestamps_cache[name] = []
                obj._additional_cache_dirty[name] = True
                obj._additional_last_timestamps[name] = 0
        # Restore axis assignments
        raw_assignments = data.get("axis_assignments", {})
        if raw_assignments:
            obj._axis_assignments = {int(k): v for k, v in raw_assignments.items()}
        # Restore chapters
        if "chapters" in data:
            obj.chapters = list(data["chapters"])
        obj._invalidate_cache('both')
        return obj

    def find_next_jump_frame(self, current_frame: int, fps: float, axis: str = 'primary') -> Optional[int]:
        """
        Finds the frame index of the first action that occurs on a frame
        strictly after the current frame.
        """
        result = self.find_next_action_position(current_frame, fps, axis)
        return result[0] if result else None

    def find_prev_jump_frame(self, current_frame: int, fps: float, axis: str = 'primary') -> Optional[int]:
        """
        Finds the frame index of the last action that occurs on a frame
        strictly before the current frame.
        """
        result = self.find_prev_action_position(current_frame, fps, axis)
        return result[0] if result else None

    def find_next_action_position(self, current_frame: int, fps: float, axis: str = 'primary') -> Optional[Tuple[int, int]]:
        """Find the next action after current_frame. Returns (frame, action_ms) or None."""
        if not fps > 0: return None
        timestamps = self._get_timestamps_for_axis(axis)
        if not timestamps: return None

        current_time_ms = current_frame * (1000.0 / fps)
        idx = bisect.bisect_right(timestamps, current_time_ms)
        while idx < len(timestamps):
            target_frame = ms_to_frame(timestamps[idx], fps)
            if target_frame > current_frame:
                return (target_frame, timestamps[idx])
            idx += 1
        return None

    def find_prev_action_position(self, current_frame: int, fps: float, axis: str = 'primary') -> Optional[Tuple[int, int]]:
        """Find the previous action before current_frame. Returns (frame, action_ms) or None."""
        if not fps > 0: return None
        timestamps = self._get_timestamps_for_axis(axis)
        if not timestamps: return None

        current_time_ms = current_frame * (1000.0 / fps)
        idx = bisect.bisect_left(timestamps, current_time_ms) - 1
        while idx >= 0:
            target_frame = ms_to_frame(timestamps[idx], fps)
            if target_frame < current_frame:
                return (target_frame, timestamps[idx])
            idx -= 1
        return None

    @property
    def actions(self) -> List[Dict]:
        return self.primary_actions

    @actions.setter
    def actions(self, value: List[Dict]):
        """
        Sets the primary actions list. Assumes 'value' is a list of action dictionaries.
        The list will be sorted by 'at'. This setter is typically used for loading
        scripts or undo/redo, where the input list is expected to be 'clean'
        (i.e., min_interval_ms and max_history are not re-applied here).
        """
        try:
            if not isinstance(value, list) or \
                    not all(isinstance(item, dict) and "at" in item and "pos" in item for item in value):
                self.logger.error(
                    "Invalid value for actions setter: Must be a list of action dicts {'at': ms, 'pos': val}.")
                self.primary_actions = []
            else:
                # Skip the sort when input is already monotonic by 'at'.
                already = all(value[i]["at"] <= value[i + 1]["at"]
                              for i in range(len(value) - 1))
                self.primary_actions = list(value) if already else sorted(
                    value, key=lambda x: x["at"])

            self.last_timestamp_primary = self.primary_actions[-1]["at"] if self.primary_actions else 0
            self._invalidate_cache('primary') # Invalidate cache

        except Exception as e:
            self.logger.error(f"Error in actions.setter: {e}. Clearing primary actions as a precaution.")
            self.primary_actions = []
            self.last_timestamp_primary = 0
            self._invalidate_cache('primary')  # Invalidate cache

    def _get_default_stats_values(self) -> dict:
        return {
            "num_points": 0, "duration_scripted_s": 0.0, "avg_speed_pos_per_s": 0.0,
            "avg_intensity_percent": 0.0, "min_pos": -1, "max_pos": -1,
            "avg_interval_ms": 0.0, "min_interval_ms": -1, "max_interval_ms": -1,
            "total_travel_dist": 0, "num_strokes": 0
        }

    def get_actions_statistics(self, axis: str = 'primary') -> dict:
        # Vectorized: uses the cached parallel arrays (_pa_times / _pa_values)
        # built by get_arrays(). ~10-20x faster than the per-point Python
        # loop on 10k+ scripts, per bench.
        stats = self._get_default_stats_values()
        t, v = self.get_arrays(axis)
        n = len(t)
        if n == 0:
            return stats
        stats["num_points"] = n
        # v is uint8 — cast to int for Python-dict round-trip so callers
        # don't see a numpy scalar type.
        stats["min_pos"] = int(v.min())
        stats["max_pos"] = int(v.max())
        if n < 2:
            return stats

        stats["duration_scripted_s"] = float(t[-1] - t[0]) / 1000.0

        # Vectorized per-segment deltas. v is uint8 so subtract as int16 to
        # keep negatives intact, then abs().
        dpos = np.abs(np.diff(v.astype(np.int16))).astype(np.int64)
        dt = np.diff(t)
        total_pos_change = int(dpos.sum())
        stats["total_travel_dist"] = total_pos_change

        # Intervals = positive dt values only (same as the old filter)
        pos_dt = dt[dt > 0]
        # Time spent in moving segments (dpos > 0 AND dt > 0)
        moving_mask = (dpos > 0) & (dt > 0)
        total_time_ms_for_speed = int(dt[moving_mask].sum())

        # Direction change counter (== old `num_strokes` logic): compare
        # consecutive non-zero directions and count flips.
        # direction: +1 if v[i+1]>v[i], -1 if <, 0 if ==
        v_int = v.astype(np.int16)
        dv = np.diff(v_int)
        direction = np.sign(dv).astype(np.int8)
        nonzero_idx = np.nonzero(direction)[0]
        if nonzero_idx.size > 1:
            nz_dirs = direction[nonzero_idx]
            num_strokes = int((nz_dirs[:-1] != nz_dirs[1:]).sum())
        else:
            num_strokes = 0
        stats["num_strokes"] = num_strokes if num_strokes > 0 else (
            1 if total_pos_change > 0 and n >= 2 else 0)

        if total_time_ms_for_speed > 0:
            stats["avg_speed_pos_per_s"] = total_pos_change / (total_time_ms_for_speed / 1000.0)

        num_segments = n - 1
        if num_segments > 0:
            stats["avg_intensity_percent"] = total_pos_change / float(num_segments)

        if pos_dt.size > 0:
            stats["avg_interval_ms"] = float(pos_dt.mean())
            stats["min_interval_ms"] = float(pos_dt.min())
            stats["max_interval_ms"] = float(pos_dt.max())
        return stats

    def get_actions_in_range(self, start_time_ms: int, end_time_ms: int, axis: str = 'primary') -> List[Dict]:
        """
        Get all actions within a time range for streaming/query purposes.

        Args:
            start_time_ms: Start of time range (inclusive)
            end_time_ms: End of time range (inclusive)
            axis: 'primary' or 'secondary'

        Returns:
            List of action dictionaries [{'at': timestamp_ms, 'pos': position}, ...]
        """
        actions_list = self.primary_actions if axis == 'primary' else self.secondary_actions
        if not actions_list:
            return []

        indices = self._get_action_indices_in_time_range(actions_list, start_time_ms, end_time_ms)
        if indices[0] is None or indices[1] is None:
            return []

        start_idx, end_idx = indices
        return actions_list[start_idx:end_idx + 1]

    def _get_action_indices_in_time_range(self, actions_list: List[dict],
                                          start_time_ms: int, end_time_ms: int,
                                          axis: str = 'primary') -> Tuple[Optional[int], Optional[int]]:
        if not actions_list: return None, None
        # Use cached timestamps when the list matches our own actions
        if actions_list is self.primary_actions or actions_list is self.secondary_actions:
            ax = 'primary' if actions_list is self.primary_actions else 'secondary'
            action_timestamps = self._get_timestamps_for_axis(ax)
        else:
            action_timestamps = [a['at'] for a in actions_list]

        s_idx = bisect.bisect_left(action_timestamps, start_time_ms)
        e_idx = bisect.bisect_right(action_timestamps, end_time_ms)
        if s_idx >= e_idx: return None, None
        return s_idx, e_idx - 1

    def auto_tune_sg_filter(self, *args, **kwargs):
        return self.signal.auto_tune_sg_filter(*args, **kwargs)

    def recover_missing_strokes(self, *args, **kwargs):
        return self.signal.recover_missing_strokes(*args, **kwargs)

    def find_peaks_and_valleys(self, *args, **kwargs):
        return self.signal.find_peaks_and_valleys(*args, **kwargs)

    def _apply_to_points(self, *args, **kwargs):
        return self.editor._apply_to_points(*args, **kwargs)

    def clear_points(self, *args, **kwargs):
        return self.editor.clear_points(*args, **kwargs)

    def clear_actions_in_time_range(self, *args, **kwargs):
        return self.editor.clear_actions_in_time_range(*args, **kwargs)

    def shift_points_time(self, *args, **kwargs):
        return self.editor.shift_points_time(*args, **kwargs)

    def add_actions_batch(self, *args, **kwargs):
        return self.editor.add_actions_batch(*args, **kwargs)

    def _filter_list_by_interval(self, axis: str):
        return self.signal._filter_list_by_interval(axis)

    def scale_points_to_range(self, *args, **kwargs):
        return self.signal.scale_points_to_range(*args, **kwargs)

    def apply_peak_preserving_resample(self, *args, **kwargs):
        return self.signal.apply_peak_preserving_resample(*args, **kwargs)

    def _simplify_keyframes_vectorized(self, extrema: List[Dict], position_tolerance: int) -> List[Dict]:
        return self.signal._simplify_keyframes_vectorized(extrema, position_tolerance)

    def list_available_plugins(self) -> List[Dict]:
        return self.plugins.list_available_plugins()

    def apply_plugin(self, *args, **kwargs) -> bool:
        return self.plugins.apply_plugin(*args, **kwargs)

    def get_plugin_preview(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plugins.get_plugin_preview(*args, **kwargs)

    def set_chapters_from_segments(self, video_segments: List, video_fps: float):
        """
        Set funscript chapters from video segments.
        
        Args:
            video_segments: List of VideoSegment objects or dictionaries
            video_fps: Video frames per second for timestamp conversion
        """
        self.chapters = []
        
        for segment in video_segments:
            # Handle both VideoSegment objects and dictionaries
            if hasattr(segment, 'start_frame_id'):
                # VideoSegment object
                start_frame_id = segment.start_frame_id
                end_frame_id = segment.end_frame_id
                position_short = segment.position_short_name
                position_long = segment.position_long_name
            elif isinstance(segment, dict):
                # Dictionary representation
                start_frame_id = segment.get('start_frame_id', 0)
                end_frame_id = segment.get('end_frame_id', 0)
                position_short = segment.get('position_short_name', segment.get('major_position', 'Unknown'))
                position_long = segment.get('position_long_name', segment.get('major_position', 'Unknown'))
            else:
                self.logger.warning(f"Unknown segment type: {type(segment)}, skipping")
                continue
            
            start_time_ms = int((start_frame_id / video_fps) * 1000)
            end_time_ms = int((end_frame_id / video_fps) * 1000)
            
            chapter_name = position_long or position_short or "Unknown"
            chapter = {
                "name": chapter_name,
                "start": start_time_ms,
                "end": end_time_ms,
                "startTime": start_time_ms,  # Keep both for compatibility
                "endTime": end_time_ms,
                "position_short": position_short,
                "position_long": position_long
            }
            
            self.chapters.append(chapter)
        
        self.logger.debug(f"Set {len(self.chapters)} chapters from video segments")

    def clear_chapters(self):
        """Clear all chapters from the funscript."""
        self.chapters = []
        self.logger.debug("Cleared all chapters")

    def add_chapter(self, start_time_ms: int, end_time_ms: int, name: str = "Chapter", 
                   position_short: str = "", position_long: str = "", **kwargs):
        """
        Add a chapter to the funscript.
        
        Args:
            start_time_ms: Chapter start time in milliseconds
            end_time_ms: Chapter end time in milliseconds  
            name: Chapter name/title
            position_short: Short position name
            position_long: Long position name
            **kwargs: Additional chapter properties
        """
        chapter = {
            "name": name,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "position_short": position_short,
            "position_long": position_long,
            **kwargs
        }
        self.chapters.append(chapter)
        self.logger.debug(f"Added chapter '{name}' ({start_time_ms}-{end_time_ms}ms)")
