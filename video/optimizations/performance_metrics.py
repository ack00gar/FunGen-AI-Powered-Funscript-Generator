#!/usr/bin/env python3
"""
Performance Metrics Tracking

Provides comprehensive performance monitoring for video processor
with minimal overhead using efficient data structures.

Features:
- Latency tracking (decode, seek, unwrap, YOLO)
- Memory usage monitoring
- CPU usage tracking
- Cache hit rate statistics
- Percentile calculations (p50, p95, p99)
- Automatic metric rotation to prevent unbounded growth
"""

import time
import psutil
import numpy as np
import threading
from collections import deque
from typing import Dict, List, Optional, Any


class PerformanceMetrics:
    """
    Lightweight performance monitoring for video processor.

    Uses deques with maxlen for automatic rotation and constant memory.
    Thread-safe with minimal locking overhead.
    """

    def __init__(self, max_samples: int = 1000):
        """
        Initialize performance metrics tracker.

        Args:
            max_samples: Maximum samples to keep per metric (auto-rotates)
        """
        self.max_samples = max_samples

        # Latency metrics (milliseconds)
        self.decode_latency_ms = deque(maxlen=max_samples)
        self.seek_latency_ms = deque(maxlen=max_samples)
        self.unwarp_latency_ms = deque(maxlen=max_samples)
        self.yolo_latency_ms = deque(maxlen=max_samples)

        # Memory metrics (megabytes) - sampled periodically
        self.memory_usage_mb = deque(maxlen=max_samples)

        # CPU metrics (percent) - sampled periodically
        self.cpu_usage_percent = deque(maxlen=max_samples)

        # Cache metrics
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        # Frame drop tracking
        self.frame_drop_count = 0
        self.total_frame_count = 0

        # Thread safety
        self.lock = threading.Lock()

        # Process handle for resource monitoring
        self.process = psutil.Process()

        # Last sample time (for rate-limiting)
        self.last_memory_sample = time.time()
        self.last_cpu_sample = time.time()
        self.sample_interval = 1.0  # Sample every 1 second

    def record_decode(self, latency_ms: float) -> None:
        """Record frame decode latency."""
        with self.lock:
            self.decode_latency_ms.append(latency_ms)

    def record_seek(self, latency_ms: float) -> None:
        """Record seek operation latency."""
        with self.lock:
            self.seek_latency_ms.append(latency_ms)

    def record_unwarp(self, latency_ms: float) -> None:
        """Record VR unwarp operation latency."""
        with self.lock:
            self.unwarp_latency_ms.append(latency_ms)

    def record_yolo(self, latency_ms: float) -> None:
        """Record YOLO inference latency."""
        with self.lock:
            self.yolo_latency_ms.append(latency_ms)

    def record_cache_hit(self) -> None:
        """Record cache hit."""
        with self.lock:
            self.cache_hit_count += 1

    def record_cache_miss(self) -> None:
        """Record cache miss."""
        with self.lock:
            self.cache_miss_count += 1

    def record_frame_drop(self) -> None:
        """Record dropped frame."""
        with self.lock:
            self.frame_drop_count += 1
            self.total_frame_count += 1

    def record_frame_processed(self) -> None:
        """Record successfully processed frame."""
        with self.lock:
            self.total_frame_count += 1

    def sample_resource_usage(self) -> None:
        """
        Sample memory and CPU usage (rate-limited).

        Call this periodically during playback.
        """
        now = time.time()

        # Sample memory
        if now - self.last_memory_sample > self.sample_interval:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            with self.lock:
                self.memory_usage_mb.append(memory_mb)
            self.last_memory_sample = now

        # Sample CPU
        if now - self.last_cpu_sample > self.sample_interval:
            cpu_percent = self.process.cpu_percent(interval=None)
            with self.lock:
                self.cpu_usage_percent.append(cpu_percent)
            self.last_cpu_sample = now

    def _calculate_percentiles(self, values: deque) -> Dict[str, float]:
        """Calculate percentiles for a metric."""
        if not values:
            return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'min': 0, 'max': 0}

        arr = np.array(values)
        return {
            'mean': np.mean(arr),
            'p50': np.percentile(arr, 50),
            'p95': np.percentile(arr, 95),
            'p99': np.percentile(arr, 99),
            'min': np.min(arr),
            'max': np.max(arr)
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all metrics.

        Returns:
            Dictionary with statistics for each metric
        """
        with self.lock:
            # Calculate cache hit rate
            total_cache_requests = self.cache_hit_count + self.cache_miss_count
            cache_hit_rate = (
                (self.cache_hit_count / total_cache_requests * 100)
                if total_cache_requests > 0 else 0
            )

            # Calculate frame drop rate
            frame_drop_rate = (
                (self.frame_drop_count / self.total_frame_count * 100)
                if self.total_frame_count > 0 else 0
            )

            return {
                'decode_latency': self._calculate_percentiles(self.decode_latency_ms),
                'seek_latency': self._calculate_percentiles(self.seek_latency_ms),
                'unwarp_latency': self._calculate_percentiles(self.unwarp_latency_ms),
                'yolo_latency': self._calculate_percentiles(self.yolo_latency_ms),
                'memory_usage': self._calculate_percentiles(self.memory_usage_mb),
                'cpu_usage': self._calculate_percentiles(self.cpu_usage_percent),
                'cache_hit_rate_percent': cache_hit_rate,
                'frame_drop_rate_percent': frame_drop_rate,
                'total_frames': self.total_frame_count,
                'samples': {
                    'decode': len(self.decode_latency_ms),
                    'seek': len(self.seek_latency_ms),
                    'unwarp': len(self.unwarp_latency_ms),
                    'yolo': len(self.yolo_latency_ms),
                    'memory': len(self.memory_usage_mb),
                    'cpu': len(self.cpu_usage_percent)
                }
            }

    def get_summary(self) -> str:
        """
        Get human-readable summary of performance metrics.

        Returns:
            Formatted string with key metrics
        """
        stats = self.get_stats()

        lines = [
            "Performance Metrics Summary",
            "=" * 60,
            "",
            "Latency (p95):",
            f"  Decode:  {stats['decode_latency']['p95']:.2f} ms",
            f"  Seek:    {stats['seek_latency']['p95']:.2f} ms",
            f"  Unwarp:  {stats['unwarp_latency']['p95']:.2f} ms",
            f"  YOLO:    {stats['yolo_latency']['p95']:.2f} ms",
            "",
            "Resource Usage (mean):",
            f"  Memory:  {stats['memory_usage']['mean']:.2f} MB",
            f"  CPU:     {stats['cpu_usage']['mean']:.1f}%",
            "",
            "Cache & Frame Stats:",
            f"  Cache Hit Rate:   {stats['cache_hit_rate_percent']:.1f}%",
            f"  Frame Drop Rate:  {stats['frame_drop_rate_percent']:.2f}%",
            f"  Total Frames:     {stats['total_frames']}",
        ]

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.decode_latency_ms.clear()
            self.seek_latency_ms.clear()
            self.unwarp_latency_ms.clear()
            self.yolo_latency_ms.clear()
            self.memory_usage_mb.clear()
            self.cpu_usage_percent.clear()
            self.cache_hit_count = 0
            self.cache_miss_count = 0
            self.frame_drop_count = 0
            self.total_frame_count = 0


# Example usage and demonstration
if __name__ == '__main__':
    import random

    print("Performance Metrics Demonstration\n" + "="*60 + "\n")

    # Create metrics tracker
    metrics = PerformanceMetrics(max_samples=1000)

    # Simulate video playback with varying performance
    print("Simulating video playback (1000 frames)...\n")

    for i in range(1000):
        # Simulate decode latency (normally distributed around 50ms)
        decode_time = max(10, random.gauss(50, 15))
        metrics.record_decode(decode_time)

        # Simulate occasional seeks (every 100 frames)
        if i % 100 == 0:
            seek_time = random.uniform(200, 500)
            metrics.record_seek(seek_time)

        # Simulate VR unwarp (if enabled, ~10ms)
        if i % 2 == 0:  # 50% of frames
            unwarp_time = random.gauss(10, 3)
            metrics.record_unwarp(unwarp_time)

        # Simulate YOLO inference (every 5 frames, ~80ms)
        if i % 5 == 0:
            yolo_time = random.gauss(80, 20)
            metrics.record_yolo(yolo_time)

        # Simulate cache hits/misses
        if random.random() < 0.85:  # 85% hit rate
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

        # Simulate occasional frame drops
        if random.random() < 0.02:  # 2% drop rate
            metrics.record_frame_drop()
        else:
            metrics.record_frame_processed()

        # Sample resources every 10 frames
        if i % 10 == 0:
            metrics.sample_resource_usage()

    # Print summary
    print(metrics.get_summary())

    # Print detailed stats
    print("\n" + "="*60)
    print("Detailed Statistics:")
    print("="*60 + "\n")

    stats = metrics.get_stats()

    for metric_name, metric_stats in stats.items():
        if isinstance(metric_stats, dict) and 'mean' in metric_stats:
            print(f"{metric_name}:")
            print(f"  Mean: {metric_stats['mean']:.2f}")
            print(f"  P50:  {metric_stats['p50']:.2f}")
            print(f"  P95:  {metric_stats['p95']:.2f}")
            print(f"  P99:  {metric_stats['p99']:.2f}")
            print(f"  Min:  {metric_stats['min']:.2f}")
            print(f"  Max:  {metric_stats['max']:.2f}")
            print()
