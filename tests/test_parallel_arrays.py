"""Tests for the parallel-arrays API on MultiAxisFunscript.

Verifies that get_arrays / bisect_at / range_indices / get_values_at_times
return correct, fresh data after every kind of mutation that callers (UI,
trackers, plugins, file load, undo/redo) can do to the action lists.
"""
import numpy as np
import pytest

from funscript.multi_axis_funscript import MultiAxisFunscript


def _build(actions_p=None, actions_s=None):
    fs = MultiAxisFunscript()
    fs.enable_point_simplification = False  # tests want literal counts
    if actions_p:
        for at, pos in actions_p:
            fs.add_action_to_axis('primary', int(at), int(pos))
    if actions_s:
        for at, pos in actions_s:
            fs.add_action_to_axis('secondary', int(at), int(pos))
    return fs


# ---------- get_arrays basic correctness ----------

def test_get_arrays_empty():
    fs = MultiAxisFunscript()
    t, v = fs.get_arrays('primary')
    assert t.dtype == np.int64 and v.dtype == np.uint8
    assert t.size == 0 and v.size == 0


def test_get_arrays_basic():
    fs = _build([(0, 0), (100, 50), (200, 100)])
    t, v = fs.get_arrays('primary')
    assert t.tolist() == [0, 100, 200]
    assert v.tolist() == [0, 50, 100]


def test_get_arrays_secondary():
    fs = _build(actions_s=[(50, 25), (150, 75)])
    t, v = fs.get_arrays('secondary')
    assert t.tolist() == [50, 150]
    assert v.tolist() == [25, 75]


def test_get_arrays_caching_returns_same_object():
    fs = _build([(0, 0), (100, 100)])
    t1, v1 = fs.get_arrays('primary')
    t2, v2 = fs.get_arrays('primary')
    assert t1 is t2 and v1 is v2  # cached


# ---------- mutation invalidation ----------

def test_add_action_invalidates_arrays():
    fs = _build([(0, 0)])
    fs.get_arrays('primary')  # warm cache
    fs.add_action_to_axis('primary', 100, 50)
    t, v = fs.get_arrays('primary')
    assert t.tolist() == [0, 100]
    assert v.tolist() == [0, 50]


def test_in_place_value_update_invalidates_values():
    fs = _build([(0, 50)])
    fs.get_arrays('primary')
    # add_action with same timestamp updates pos in place
    fs.add_action_to_axis('primary', 0, 75)
    _, v = fs.get_arrays('primary')
    assert v.tolist() == [75]


def test_clear_invalidates_arrays():
    fs = _build([(0, 0), (100, 100)])
    fs.get_arrays('primary')
    fs.clear()
    t, _ = fs.get_arrays('primary')
    assert t.size == 0


def test_explicit_invalidate_cache_resets_arrays():
    fs = _build([(0, 0)])
    fs.get_arrays('primary')
    fs.primary_actions.append({'at': 99, 'pos': 99})
    fs._invalidate_cache('primary')
    t, v = fs.get_arrays('primary')
    assert t.tolist() == [0, 99]
    assert v.tolist() == [0, 99]


def test_mark_actions_dirty_resets_after_inplace_edit():
    fs = _build([(0, 0), (100, 50)])
    fs.get_arrays('primary')
    # Simulate plugin doing in-place pos mutation without ts change
    fs.primary_actions[1]['pos'] = 88
    fs.mark_actions_dirty('primary')
    _, v = fs.get_arrays('primary')
    assert v.tolist() == [0, 88]


def test_invalidate_both_axes():
    fs = _build([(0, 0)], [(50, 50)])
    fs.get_arrays('primary')
    fs.get_arrays('secondary')
    fs._invalidate_cache('both')
    fs.primary_actions.append({'at': 100, 'pos': 99})
    fs.secondary_actions.append({'at': 100, 'pos': 1})
    tp, vp = fs.get_arrays('primary')
    ts, vs = fs.get_arrays('secondary')
    assert tp.tolist() == [0, 100]
    assert vp.tolist() == [0, 99]
    assert ts.tolist() == [50, 100]
    assert vs.tolist() == [50, 1]


# ---------- bisect_at / range_indices ----------

def test_bisect_at_basic():
    fs = _build([(0, 0), (100, 50), (200, 100)])
    assert fs.bisect_at('primary', -10) == 0
    assert fs.bisect_at('primary', 0) == 0
    assert fs.bisect_at('primary', 50) == 1
    assert fs.bisect_at('primary', 100) == 1   # default left
    assert fs.bisect_at('primary', 100, side='right') == 2
    assert fs.bisect_at('primary', 999) == 3


def test_bisect_at_empty():
    fs = MultiAxisFunscript()
    assert fs.bisect_at('primary', 100) == 0


def test_range_indices():
    fs = _build([(0, 0), (100, 50), (200, 100), (300, 50), (400, 0)])
    lo, hi = fs.range_indices('primary', 50, 250)
    assert (lo, hi) == (1, 3)
    assert fs.primary_actions[lo:hi] == [
        {'at': 100, 'pos': 50}, {'at': 200, 'pos': 100}]


def test_range_indices_inclusive_endpoints():
    fs = _build([(0, 0), (100, 50), (200, 100)])
    lo, hi = fs.range_indices('primary', 0, 200)
    assert (lo, hi) == (0, 3)


# ---------- get_values_at_times ----------

def test_get_values_at_times_interpolation():
    fs = _build([(0, 0), (100, 100), (200, 0)])
    out = fs.get_values_at_times([0, 50, 100, 150, 200])
    np.testing.assert_allclose(out, [0, 50, 100, 50, 0], rtol=1e-3)


def test_get_values_at_times_clamps_outside():
    fs = _build([(100, 25), (200, 75)])
    out = fs.get_values_at_times([0, 1000])
    np.testing.assert_allclose(out, [25, 75], rtol=1e-3)


def test_get_values_at_times_empty():
    fs = MultiAxisFunscript()
    out = fs.get_values_at_times([0, 100])
    np.testing.assert_array_equal(out, [50.0, 50.0])


def test_get_values_at_times_single_action():
    fs = _build([(100, 42)])
    out = fs.get_values_at_times([0, 100, 999])
    np.testing.assert_array_equal(out, [42, 42, 42])


# ---------- consistency after many edits ----------

def test_arrays_consistent_after_many_inserts():
    fs = MultiAxisFunscript()
    rng = np.random.default_rng(42)
    times = sorted(rng.integers(0, 1_000_000, size=500).tolist())
    poss = rng.integers(0, 101, size=500).tolist()
    for t, p in zip(times, poss):
        fs.add_action_to_axis('primary', int(t), int(p))
    arr_t, arr_v = fs.get_arrays('primary')
    actual_ts = [a['at'] for a in fs.primary_actions]
    actual_ps = [a['pos'] for a in fs.primary_actions]
    assert arr_t.tolist() == actual_ts
    assert arr_v.tolist() == actual_ps


def test_arrays_match_get_value_interpolation():
    fs = _build([(0, 10), (100, 90), (200, 30), (300, 70)])
    times = np.linspace(0, 300, 50, dtype=np.int64)
    arr_vals = fs.get_values_at_times(times)
    scalar_vals = np.array([fs.get_value(int(tm), 'primary') for tm in times])
    # get_value returns int; arr_vals is float — compare with tolerance
    np.testing.assert_allclose(arr_vals, scalar_vals, atol=1.0)


# ---------- additional axes ----------

def test_additional_axis():
    fs = MultiAxisFunscript()
    fs.enable_point_simplification = False
    fs.ensure_axis('twist')
    fs.add_action_to_axis('twist', 0, 0)
    fs.add_action_to_axis('twist', 100, 100)
    t, v = fs.get_arrays('twist')
    assert t.tolist() == [0, 100]
    assert v.tolist() == [0, 100]
    fs.add_action_to_axis('twist', 50, 50)
    t, v = fs.get_arrays('twist')
    assert t.tolist() == [0, 50, 100]
    assert v.tolist() == [0, 50, 100]


def test_additional_axis_invalidate_both():
    fs = MultiAxisFunscript()
    fs.enable_point_simplification = False
    fs.ensure_axis('twist')
    fs.add_action_to_axis('twist', 0, 0)
    fs.get_arrays('twist')
    fs.primary_actions.append({'at': 0, 'pos': 0})
    fs.additional_axes['twist'].append({'at': 50, 'pos': 50})
    fs._invalidate_cache('both')
    t, _ = fs.get_arrays('twist')
    assert t.tolist() == [0, 50]


# ---------- get_value bracket cache invariant ----------

def test_arrays_consistent_after_clear_axis():
    fs = MultiAxisFunscript()
    fs.enable_point_simplification = False
    fs.ensure_axis('twist')
    fs.add_action_to_axis('twist', 0, 0)
    fs.add_action_to_axis('twist', 100, 50)
    fs.get_arrays('twist')
    fs.clear_axis('twist')
    t, _ = fs.get_arrays('twist')
    assert t.size == 0


def test_arrays_consistent_after_set_axis_actions():
    fs = _build([(0, 0)])
    fs.get_arrays('primary')
    fs.set_axis_actions('primary', [{'at': 0, 'pos': 10}, {'at': 100, 'pos': 90}])
    t, v = fs.get_arrays('primary')
    assert t.tolist() == [0, 100]
    assert v.tolist() == [10, 90]


def test_arrays_consistent_after_apply_plugin_invert():
    """Invert plugin mutates in place; verify arrays reflect new values."""
    fs = _build([(0, 30), (100, 70)])
    pre_t, pre_v = fs.get_arrays('primary')
    assert pre_v.tolist() == [30, 70]
    ok = fs.apply_plugin('Invert', axis='primary')
    if not ok:
        pytest.skip("Invert plugin not available")
    _, post_v = fs.get_arrays('primary')
    # 100 - x for each
    assert post_v.tolist() == [70, 30]


def test_get_value_bracket_cache_reset_on_invalidate():
    fs = _build([(0, 0), (100, 100), (200, 0)])
    # Prime the bracket cache
    fs.get_value(50, 'primary')
    # Now mutate
    fs.add_action_to_axis('primary', 50, 99)
    # Next get_value must see new point
    assert fs.get_value(50, 'primary') == 99
