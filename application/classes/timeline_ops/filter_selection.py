"""Narrow the current selection to extrema (peaks / valleys / mid).

Selection is filtered against neighbours INSIDE the current selection,
not the whole timeline, matches the right-click "Keep Only..." menu.
"""


def filter_selection(tl, mode: str) -> None:
    if mode not in ('top', 'bottom', 'mid'):
        return
    actions = tl._get_actions()
    if len(tl.multi_selected_action_indices) < 3:
        return
    indices = tl._resolve_selected_indices()
    if len(indices) < 3:
        return
    subset = [actions[i] for i in indices]
    keep_keys = set()
    for k, idx in enumerate(indices):
        current = actions[idx]['pos']
        prev_val = subset[k - 1]['pos'] if k > 0 else -1
        next_val = subset[k + 1]['pos'] if k < len(subset) - 1 else -1
        is_peak = (current > prev_val) and (current >= next_val)
        is_valley = (current < prev_val) and (current <= next_val)
        if mode == 'top' and is_peak:
            keep_keys.add(tl._action_key(actions[idx]))
        elif mode == 'bottom' and is_valley:
            keep_keys.add(tl._action_key(actions[idx]))
        elif mode == 'mid' and not is_peak and not is_valley:
            keep_keys.add(tl._action_key(actions[idx]))
    tl.multi_selected_action_indices = keep_keys
