"""Pattern library for saving, loading, and applying motion patterns.

Patterns are normalized funscript action sequences that can be scaled
in time and amplitude, then applied to any position in the script.
Each pattern is saved as an individual JSON file for easy sharing.
"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class MotionPattern:
    """A reusable motion pattern with normalized timing."""
    name: str
    description: str = ""
    actions: List[Dict] = field(default_factory=list)  # Normalized: t starts at 0
    duration_ms: float = 0.0
    tags: List[str] = field(default_factory=list)


class PatternLibrary:
    """Save, load, browse, and apply motion patterns.

    Patterns are stored as individual JSON files in a configurable directory.
    """

    def __init__(self, patterns_dir: str = None):
        if patterns_dir is None:
            # Default: patterns/ in app data directory
            base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), 'patterns')
            patterns_dir = base
        self._patterns_dir = patterns_dir
        os.makedirs(self._patterns_dir, exist_ok=True)

    @property
    def patterns_dir(self) -> str:
        return self._patterns_dir

    def save_pattern(self, name: str, actions: List[Dict],
                     description: str = "", tags: List[str] = None) -> MotionPattern:
        """Save a pattern from a selection of actions.

        Actions are normalized so the first action starts at t=0.

        Args:
            name: Pattern name (used as filename)
            actions: Raw funscript actions to save
            description: Optional description
            tags: Optional tags list

        Returns:
            The saved MotionPattern
        """
        if not actions or len(actions) < 2:
            raise ValueError("Need at least 2 actions to create a pattern")

        # Normalize: shift time so first action is at t=0
        t_offset = actions[0]['at']
        normalized = [{'at': a['at'] - t_offset, 'pos': a['pos']} for a in actions]
        duration_ms = normalized[-1]['at']

        pattern = MotionPattern(
            name=name,
            description=description,
            actions=normalized,
            duration_ms=duration_ms,
            tags=tags or [],
        )

        # Save to file
        filepath = self._pattern_filepath(name)
        data = {
            'name': pattern.name,
            'description': pattern.description,
            'actions': pattern.actions,
            'duration_ms': pattern.duration_ms,
            'tags': pattern.tags,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return pattern

    def load_pattern(self, name: str) -> Optional[MotionPattern]:
        """Load a pattern by name."""
        filepath = self._pattern_filepath(name)
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return MotionPattern(
                name=data.get('name', name),
                description=data.get('description', ''),
                actions=data.get('actions', []),
                duration_ms=data.get('duration_ms', 0),
                tags=data.get('tags', []),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def apply_pattern(self, pattern: MotionPattern, start_time_ms: float,
                      speed_factor: float = 1.0,
                      amplitude_factor: float = 1.0) -> List[Dict]:
        """Scale and position a pattern for insertion into a script.

        Args:
            pattern: The pattern to apply
            start_time_ms: Where to insert (ms)
            speed_factor: Time scaling (>1 = faster, <1 = slower)
            amplitude_factor: Position scaling (1.0 = original, 0.5 = half range)

        Returns:
            List of scaled funscript actions ready for insertion
        """
        if not pattern.actions:
            return []

        speed_factor = max(0.1, speed_factor)
        center = 50.0  # Center position for amplitude scaling

        result = []
        for a in pattern.actions:
            # Scale time
            scaled_time = start_time_ms + (a['at'] / speed_factor)

            # Scale amplitude around center
            pos = center + (a['pos'] - center) * amplitude_factor
            pos = max(0, min(100, int(round(pos))))

            result.append({'at': int(round(scaled_time)), 'pos': pos})

        return result

    def list_patterns(self) -> List[str]:
        """Return list of available pattern names."""
        patterns = []
        if os.path.isdir(self._patterns_dir):
            for f in sorted(os.listdir(self._patterns_dir)):
                if f.endswith('.json'):
                    patterns.append(f[:-5])  # Strip .json
        return patterns

    def delete_pattern(self, name: str) -> bool:
        """Delete a pattern by name. Returns True if deleted."""
        filepath = self._pattern_filepath(name)
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def _pattern_filepath(self, name: str) -> str:
        """Get filesystem path for a pattern name."""
        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in ' _-').strip()
        if not safe_name:
            safe_name = "unnamed"
        return os.path.join(self._patterns_dir, f"{safe_name}.json")
