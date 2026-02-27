"""Funscript quality validation and scoring.

Checks for common issues: speed limit violations, dead zones, poor coverage,
imperceptible movements, and timing anomalies. Produces a 0-100 quality score.
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


class IssueSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityIssue:
    severity: IssueSeverity
    category: str
    message: str
    time_range_ms: Optional[Tuple[float, float]] = None


@dataclass
class QualityReport:
    issues: List[QualityIssue] = field(default_factory=list)
    score: int = 100
    stats: Dict = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.INFO)


class FunscriptQualityValidator:
    """Validates funscript quality and produces a scored report.

    Args:
        speed_limit: Maximum acceptable speed in units/second (default: 400)
        dead_zone_threshold_ms: Gap threshold for dead zone detection (default: 5000ms)
        min_movement_threshold: Minimum position change to be perceptible (default: 5)
        min_interval_ms: Minimum acceptable interval between points (default: 20ms)
        max_interval_ms: Maximum acceptable interval during active sections (default: 5000ms)
    """

    def __init__(
        self,
        speed_limit: float = 400.0,
        dead_zone_threshold_ms: float = 5000.0,
        min_movement_threshold: float = 5.0,
        min_interval_ms: float = 20.0,
        max_interval_ms: float = 5000.0,
    ):
        self.speed_limit = speed_limit
        self.dead_zone_threshold_ms = dead_zone_threshold_ms
        self.min_movement_threshold = min_movement_threshold
        self.min_interval_ms = min_interval_ms
        self.max_interval_ms = max_interval_ms

    def validate(self, actions: List[Dict], duration_ms: float = 0) -> QualityReport:
        """Run all quality checks and return a scored report.

        Args:
            actions: Funscript actions [{'at': ms, 'pos': 0-100}, ...]
            duration_ms: Total video/script duration in ms (0 = auto from last action)

        Returns:
            QualityReport with issues, score, and statistics
        """
        report = QualityReport()

        if not actions:
            report.issues.append(QualityIssue(
                IssueSeverity.ERROR, "empty", "Script has no actions"
            ))
            report.score = 0
            return report

        if len(actions) < 2:
            report.issues.append(QualityIssue(
                IssueSeverity.ERROR, "insufficient", "Script has fewer than 2 actions"
            ))
            report.score = 10
            return report

        # Auto-detect duration from last action if not provided
        if duration_ms <= 0:
            duration_ms = actions[-1]['at']

        # --- Pre-compute shared arrays ONCE (saves 8+ redundant O(n) extractions) ---
        ats = np.array([a['at'] for a in actions], dtype=np.float64)
        poss = np.array([a['pos'] for a in actions], dtype=np.float64)
        dt = np.diff(ats)
        dp = np.abs(np.diff(poss))
        dt_safe = np.where(dt > 0, dt, 1.0)
        speeds = (dp / dt_safe) * 1000.0

        # Run individual checks with pre-computed arrays
        self._check_speed_limits(ats, speeds, report)
        self._check_dead_zones(ats, dp, report)
        self._check_coverage(ats, duration_ms, report)
        self._check_min_movement(dp, report)
        self._check_interval_anomalies(ats, dt, dp, report)

        # Compute statistics from pre-computed arrays
        report.stats = self._compute_stats(ats, poss, dt, speeds, duration_ms, len(actions))

        # Compute final score
        report.score = self._compute_score(report)

        return report

    def _check_speed_limits(self, ats, speeds, report: QualityReport):
        """Check for segments exceeding the speed limit — emits ONE aggregated issue."""
        violations = np.where(speeds > self.speed_limit)[0]
        n_violations = len(violations)

        if n_violations == 0:
            return

        max_speed = float(np.max(speeds[violations]))
        avg_speed = float(np.mean(speeds[violations]))

        if n_violations > 50:
            severity = IssueSeverity.ERROR
        elif n_violations > 10:
            severity = IssueSeverity.WARNING
        else:
            severity = IssueSeverity.INFO

        report.issues.append(QualityIssue(
            severity,
            "speed_limit",
            f"{n_violations} speed violations (max {max_speed:.0f} u/s, avg {avg_speed:.0f} u/s, limit {self.speed_limit:.0f} u/s)",
        ))

    def _check_dead_zones(self, ats, dp, report: QualityReport):
        """Check for extended periods with no movement."""
        n = len(dp)
        i = 0
        while i < n:
            if dp[i] < self.min_movement_threshold:
                zone_start = ats[i]
                j = i
                while j < n and dp[j] < self.min_movement_threshold:
                    j += 1
                zone_end = ats[min(j, len(ats) - 1)]
                zone_duration = zone_end - zone_start

                if zone_duration >= self.dead_zone_threshold_ms:
                    report.issues.append(QualityIssue(
                        IssueSeverity.INFO,
                        "dead_zone",
                        f"Dead zone ({zone_duration / 1000:.1f}s with no movement)",
                        time_range_ms=(float(zone_start), float(zone_end))
                    ))
                i = j
            else:
                i += 1

    def _check_coverage(self, ats, duration_ms: float, report: QualityReport):
        """Check what percentage of the video duration is scripted."""
        if duration_ms <= 0:
            return

        script_start = ats[0]
        script_end = ats[-1]
        scripted_duration = script_end - script_start
        coverage = (scripted_duration / duration_ms) * 100.0

        report.stats['coverage_pct'] = coverage

        if coverage < 50:
            report.issues.append(QualityIssue(
                IssueSeverity.WARNING,
                "coverage",
                f"Only {coverage:.0f}% of video duration is scripted"
            ))
        elif coverage < 80:
            report.issues.append(QualityIssue(
                IssueSeverity.INFO,
                "coverage",
                f"{coverage:.0f}% of video duration is scripted"
            ))

    def _check_min_movement(self, dp, report: QualityReport):
        """Check for imperceptible position changes."""
        small_moves = np.where((dp > 0) & (dp < self.min_movement_threshold))[0]

        if len(small_moves) > 0:
            pct = (len(small_moves) / max(1, len(dp))) * 100
            if pct > 10:
                report.issues.append(QualityIssue(
                    IssueSeverity.WARNING,
                    "min_movement",
                    f"{len(small_moves)} segments ({pct:.0f}%) have imperceptible movement (<{self.min_movement_threshold} units)"
                ))
            elif len(small_moves) > 5:
                report.issues.append(QualityIssue(
                    IssueSeverity.INFO,
                    "min_movement",
                    f"{len(small_moves)} segments have very small movement (<{self.min_movement_threshold} units)"
                ))

    def _check_interval_anomalies(self, ats, dt, dp, report: QualityReport):
        """Check for points too close together or too far apart."""
        # Too close
        too_close = np.where(dt < self.min_interval_ms)[0]
        if len(too_close) > 0:
            report.issues.append(QualityIssue(
                IssueSeverity.INFO,
                "interval_close",
                f"{len(too_close)} point pairs are very close together (<{self.min_interval_ms:.0f}ms)"
            ))

        # Too far apart (in active sections only — skip if it's a dead zone)
        too_far = np.where((dt > self.max_interval_ms) & (dp > self.min_movement_threshold))[0]
        if len(too_far) > 0:
            for idx in too_far[:5]:  # Report up to 5
                report.issues.append(QualityIssue(
                    IssueSeverity.WARNING,
                    "interval_far",
                    f"Large gap ({dt[idx] / 1000:.1f}s) with movement — may cause device stalling",
                    time_range_ms=(float(ats[idx]), float(ats[idx + 1]))
                ))

    def _compute_stats(self, ats, poss, dt, speeds, duration_ms: float, action_count: int) -> Dict:
        """Compute summary statistics from pre-computed arrays."""
        return {
            'action_count': action_count,
            'duration_ms': duration_ms,
            'avg_speed': float(np.mean(speeds)) if len(speeds) > 0 else 0,
            'max_speed': float(np.max(speeds)) if len(speeds) > 0 else 0,
            'min_pos': float(np.min(poss)),
            'max_pos': float(np.max(poss)),
            'avg_interval_ms': float(np.mean(dt)) if len(dt) > 0 else 0,
        }

    def _compute_score(self, report: QualityReport) -> int:
        """Compute weighted quality score 0-100 from issues."""
        score = 100

        for issue in report.issues:
            if score <= 0:
                break

            if issue.severity == IssueSeverity.ERROR:
                if issue.category == "empty":
                    score -= 100
                elif issue.category == "insufficient":
                    score -= 80
                else:
                    score -= 20
            elif issue.severity == IssueSeverity.WARNING:
                if issue.category == "speed_limit":
                    # Parse violation count from aggregated message for proportional penalty
                    try:
                        n = int(issue.message.split()[0])
                        score -= min(30, n * 0.5)
                    except (ValueError, IndexError):
                        score -= 5
                elif issue.category == "coverage":
                    score -= 10
                elif issue.category == "min_movement":
                    score -= 5
                elif issue.category == "interval_far":
                    score -= 3
                else:
                    score -= 2
            elif issue.severity == IssueSeverity.INFO:
                score -= 1

        return max(0, min(100, int(score)))
