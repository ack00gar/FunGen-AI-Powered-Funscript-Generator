"""Funscript comparison utilities for reference overlay feature.

Provides peak detection, peak matching, and aggregate metrics for
comparing a generated funscript against a reference (manual) one.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple


def detect_peaks(actions: List[Dict], tolerance: int = 2) -> List[Dict]:
    """Detect reversal points (local min/max) in a funscript.

    Args:
        actions: Sorted list of {'at': ms, 'pos': 0-100} dicts.
        tolerance: Minimum position change to count as a direction change.

    Returns:
        List of action dicts that are peaks (reversals).
    """
    if len(actions) < 3:
        return list(actions)

    # Vectorized: extract positions, compute diffs, find sign changes
    pos = np.array([a['pos'] for a in actions], dtype=np.float64)
    d = np.diff(pos)

    # A peak is where consecutive diffs change sign beyond tolerance
    # d_prev > tolerance and d_next < -tolerance  (local max)
    # d_prev < -tolerance and d_next > tolerance  (local min)
    d_prev = d[:-1]
    d_next = d[1:]
    is_max = (d_prev > tolerance) & (d_next < -tolerance)
    is_min = (d_prev < -tolerance) & (d_next > tolerance)
    peak_mask = is_max | is_min

    # Interior peak indices (offset by 1 since d_prev starts at index 0 = actions[1])
    interior_indices = np.flatnonzero(peak_mask) + 1

    # Always include first and last
    peaks = [actions[0]]
    for idx in interior_indices:
        peaks.append(actions[idx])
    peaks.append(actions[-1])
    return peaks


def match_peaks(
    main_peaks: List[Dict],
    ref_peaks: List[Dict],
    max_window_ms: float = 500.0
) -> Tuple[List[Tuple[Dict, Dict, float]], List[Dict], List[Dict]]:
    """Match peaks between main and reference by nearest timestamp.

    Uses greedy nearest-neighbor with binary search within a time window.

    Returns:
        (matched_pairs, unmatched_main, unmatched_ref)
        matched_pairs: List of (main_peak, ref_peak, offset_ms)
        unmatched_main: Main peaks with no reference counterpart
        unmatched_ref: Reference peaks with no main counterpart
    """
    if not main_peaks or not ref_peaks:
        return [], list(main_peaks), list(ref_peaks)

    ref_times = np.array([p['at'] for p in ref_peaks], dtype=np.float64)
    used_ref = set()
    matched = []
    unmatched_main = []

    for mp in main_peaks:
        t = mp['at']
        # Binary search for nearest ref peak — O(log n) per lookup
        idx = int(np.searchsorted(ref_times, t))
        best_idx = None
        best_offset = max_window_ms + 1
        # Check idx-1 and idx (the two nearest candidates)
        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(ref_times) and candidate not in used_ref:
                off = abs(ref_times[candidate] - t)
                if off < best_offset:
                    best_offset = off
                    best_idx = candidate

        if best_idx is not None and best_offset <= max_window_ms:
            matched.append((mp, ref_peaks[best_idx], float(best_offset)))
            used_ref.add(best_idx)
        else:
            unmatched_main.append(mp)

    unmatched_ref = [ref_peaks[i] for i in range(len(ref_peaks)) if i not in used_ref]
    return matched, unmatched_main, unmatched_ref


def classify_match(offset_ms: float, pos_diff: float, fps: float = 30.0) -> str:
    """Classify a peak match by timing offset AND position difference.

    Args:
        offset_ms: Absolute time difference between matched peaks.
        pos_diff: Absolute position difference (0-100 scale).
        fps: Video frame rate for gold threshold.

    Returns: 'gold', 'green', 'yellow', or 'red'
    """
    frame_ms = 1000.0 / fps if fps > 0 else 33.33
    # Both timing and position must be within threshold for each tier
    if offset_ms < frame_ms and pos_diff < 3:
        return 'gold'
    elif offset_ms < 66.0 and pos_diff < 7:
        return 'green'
    elif offset_ms < 133.0 and pos_diff < 15:
        return 'yellow'
    else:
        return 'red'


def compute_comparison_metrics(
    main_actions: List[Dict],
    ref_actions: List[Dict],
    fps: float = 30.0,
    chapters: Optional[List] = None,
    peak_data: Optional[Tuple] = None,
) -> Optional[Dict]:
    """Compute aggregate comparison metrics between main and reference funscripts.

    Args:
        peak_data: Optional pre-computed (matched, unmatched_main, unmatched_ref)
                   to avoid recomputing peaks. Pass from caller if already available.

    Returns dict with: mae, correlation, speed_mae, peak_stats, per_chapter (if chapters).
    Returns None if either input has < 2 actions.
    """
    if len(main_actions) < 2 or len(ref_actions) < 2:
        return None

    # Build numpy arrays
    main_at = np.array([a['at'] for a in main_actions], dtype=np.float64)
    main_pos = np.array([a['pos'] for a in main_actions], dtype=np.float64)
    ref_at = np.array([a['at'] for a in ref_actions], dtype=np.float64)
    ref_pos = np.array([a['pos'] for a in ref_actions], dtype=np.float64)

    # Interpolate reference positions at main timestamps
    overlap_start = max(main_at[0], ref_at[0])
    overlap_end = min(main_at[-1], ref_at[-1])
    if overlap_end <= overlap_start:
        return None

    mask = (main_at >= overlap_start) & (main_at <= overlap_end)
    if mask.sum() < 2:
        return None

    query_at = main_at[mask]
    query_pos = main_pos[mask]
    ref_interp = np.interp(query_at, ref_at, ref_pos)

    # MAE
    mae = float(np.mean(np.abs(query_pos - ref_interp)))

    # Pearson correlation
    if np.std(query_pos) > 0 and np.std(ref_interp) > 0:
        corr = float(np.corrcoef(query_pos, ref_interp)[0, 1])
    else:
        corr = 0.0

    # Speed MAE
    dt_main = np.diff(query_at)
    dt_main_safe = np.where(dt_main > 0, dt_main, 1.0)
    speed_main = np.abs(np.diff(query_pos)) / dt_main_safe * 1000.0
    speed_ref = np.abs(np.diff(ref_interp)) / dt_main_safe * 1000.0
    speed_mae = float(np.mean(np.abs(speed_main - speed_ref)))

    # Peak matching stats — reuse pre-computed data if available
    if peak_data is not None:
        matched, unmatched_main, unmatched_ref = peak_data
    else:
        main_peaks = detect_peaks(main_actions)
        ref_peaks = detect_peaks(ref_actions)
        matched, unmatched_main, unmatched_ref = match_peaks(main_peaks, ref_peaks)

    peak_classes = {'gold': 0, 'green': 0, 'yellow': 0, 'red': 0}
    for mp, rp, offset in matched:
        pos_diff = abs(mp['pos'] - rp['pos'])
        peak_classes[classify_match(offset, pos_diff, fps)] += 1

    total_peaks = len(matched) + len(unmatched_main) + len(unmatched_ref)
    peak_stats = {
        'matched': len(matched),
        'unmatched_main': len(unmatched_main),
        'unmatched_ref': len(unmatched_ref),
        'total': total_peaks,
        'classes': peak_classes,
    }

    # Coverage
    overlap_duration = overlap_end - overlap_start
    ref_duration = ref_at[-1] - ref_at[0]
    coverage = float(overlap_duration / ref_duration) if ref_duration > 0 else 0.0

    result = {
        'mae': mae,
        'correlation': corr,
        'speed_mae': speed_mae,
        'peak_stats': peak_stats,
        'coverage': coverage,
    }

    # Per-chapter stats (MAE + peak accuracy)
    if chapters:
        per_chapter = {}
        # Pre-extract match times for binary search
        matched_times = np.array([m[0]['at'] for m in matched], dtype=np.float64) if matched else np.array([], dtype=np.float64)
        um_main_times = np.array([p['at'] for p in unmatched_main], dtype=np.float64) if unmatched_main else np.array([], dtype=np.float64)
        um_ref_times = np.array([p['at'] for p in unmatched_ref], dtype=np.float64) if unmatched_ref else np.array([], dtype=np.float64)

        for ch in chapters:
            ch_start = ch.get('start_ms') or ch.get('start_time_ms', 0)
            ch_end = ch.get('end_ms') or ch.get('end_time_ms', 0)
            ch_name = ch.get('name', ch.get('label', 'unknown'))
            if ch_end <= ch_start:
                continue

            ch_info = {}

            # MAE for this chapter — use searchsorted for fast range lookup
            i_start = int(np.searchsorted(query_at, ch_start))
            i_end = int(np.searchsorted(query_at, ch_end, side='right'))
            if i_end - i_start >= 2:
                ch_info['mae'] = float(np.mean(np.abs(query_pos[i_start:i_end] - ref_interp[i_start:i_end])))

            # Peak counts via searchsorted
            m_s = int(np.searchsorted(matched_times, ch_start))
            m_e = int(np.searchsorted(matched_times, ch_end, side='right'))
            um_s = int(np.searchsorted(um_main_times, ch_start))
            um_e = int(np.searchsorted(um_main_times, ch_end, side='right'))
            ur_s = int(np.searchsorted(um_ref_times, ch_start))
            ur_e = int(np.searchsorted(um_ref_times, ch_end, side='right'))

            ch_matched_slice = matched[m_s:m_e] if len(matched_times) > 0 else []
            ch_total = (m_e - m_s) + (um_e - um_s) + (ur_e - ur_s)
            if ch_total > 0:
                ch_good = sum(1 for mp, rp, off in ch_matched_slice
                              if classify_match(off, abs(mp['pos'] - rp['pos']), fps) in ('gold', 'green'))
                ch_info['peaks_total'] = ch_total
                ch_info['peaks_good'] = ch_good
                ch_info['peaks_accuracy'] = ch_good / ch_total

            if ch_info:
                per_chapter[ch_name] = ch_info
        if per_chapter:
            result['per_chapter'] = per_chapter

    return result


def detect_problem_sections(
    main_actions: List[Dict],
    ref_actions: List[Dict],
    fps: float = 30.0,
    window_ms: float = 500.0,
    mae_threshold: float = 20.0,
    min_gap_ms: float = 300.0,
) -> List[Dict]:
    """Detect contiguous sections where the generated script deviates significantly.

    Slides a window over the overlap region, computes local MAE, and merges
    consecutive high-error windows into problem sections.

    Args:
        window_ms: Sliding window size in ms.
        mae_threshold: MAE above this marks a window as problematic.
        min_gap_ms: Merge sections closer than this.

    Returns:
        List of {'start_ms', 'end_ms', 'mae', 'duration_ms'} dicts.
    """
    if len(main_actions) < 2 or len(ref_actions) < 2:
        return []

    main_at = np.array([a['at'] for a in main_actions], dtype=np.float64)
    main_pos = np.array([a['pos'] for a in main_actions], dtype=np.float64)
    ref_at = np.array([a['at'] for a in ref_actions], dtype=np.float64)
    ref_pos = np.array([a['pos'] for a in ref_actions], dtype=np.float64)

    overlap_start = max(main_at[0], ref_at[0])
    overlap_end = min(main_at[-1], ref_at[-1])
    if overlap_end - overlap_start < window_ms:
        return []

    mask = (main_at >= overlap_start) & (main_at <= overlap_end)
    if mask.sum() < 2:
        return []

    query_at = main_at[mask]
    query_pos = main_pos[mask]
    ref_interp = np.interp(query_at, ref_at, ref_pos)
    errors = np.abs(query_pos - ref_interp)

    # Slide window using searchsorted for O(log n) per step instead of O(n)
    step_ms = window_ms / 2
    bad_windows = []
    t = overlap_start
    while t + window_ms <= overlap_end:
        i_start = int(np.searchsorted(query_at, t))
        i_end = int(np.searchsorted(query_at, t + window_ms, side='right'))
        if i_end - i_start >= 2:
            local_mae = float(np.mean(errors[i_start:i_end]))
            if local_mae > mae_threshold:
                bad_windows.append((t, t + window_ms, local_mae))
        t += step_ms

    if not bad_windows:
        return []

    # Merge overlapping / close windows
    sections = []
    cur_start, cur_end, cur_maes = bad_windows[0][0], bad_windows[0][1], [bad_windows[0][2]]
    for s, e, m in bad_windows[1:]:
        if s - cur_end <= min_gap_ms:
            cur_end = max(cur_end, e)
            cur_maes.append(m)
        else:
            sections.append({
                'start_ms': cur_start,
                'end_ms': cur_end,
                'mae': float(np.mean(cur_maes)),
                'duration_ms': cur_end - cur_start,
            })
            cur_start, cur_end, cur_maes = s, e, [m]
    sections.append({
        'start_ms': cur_start,
        'end_ms': cur_end,
        'mae': float(np.mean(cur_maes)),
        'duration_ms': cur_end - cur_start,
    })

    return sections
