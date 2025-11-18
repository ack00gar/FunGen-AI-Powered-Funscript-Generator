#!/usr/bin/env python3
"""
Unified Frame Buffer Manager (Quick Win #4)

Combines three separate buffering systems into one:
1. LRU cache for random access
2. Backward navigation buffer (deque)
3. Statistics and monitoring

Benefits:
- Simpler code (single lock, single interface)
- Better cache coherency
- Unified statistics
- Easier to test and maintain
"""

import cv2
import numpy as np
import threading
from collections import OrderedDict, deque
from typing import Optional, Dict, Any, Tuple


class UnifiedFrameBuffer:
    """
    Unified frame buffer with multiple access patterns.

    Features:
    - LRU cache for random access
    - Deque for backward navigation
    - JPEG compression for memory efficiency
    - Thread-safe operations
    - Comprehensive statistics
    """

    def __init__(
        self,
        lru_cache_size: int = 50,
        backward_buffer_size: int = 600,
        compression_quality: int = 95,
        refill_threshold: int = 120
    ):
        """
        Initialize unified frame buffer.

        Args:
            lru_cache_size: Maximum frames in LRU cache
            backward_buffer_size: Maximum frames in backward navigation buffer
            compression_quality: JPEG compression quality (0-100)
            refill_threshold: Backward buffer refill threshold
        """
        # LRU cache for random access
        self.lru_cache = OrderedDict()
        self.lru_cache_size = lru_cache_size

        # Deque for backward navigation
        self.backward_buffer = deque(maxlen=backward_buffer_size)
        self.backward_buffer_size = backward_buffer_size
        self.refill_threshold = refill_threshold

        # Compression settings
        self.compression_quality = compression_quality

        # Single lock for all operations
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            'lru_hits': 0,
            'lru_misses': 0,
            'backward_hits': 0,
            'backward_misses': 0,
            'total_frames_added': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'refill_count': 0
        }

        # Refill state
        self.is_refilling = False

    def add_frame(self, frame_idx: int, frame: np.ndarray) -> None:
        """
        Add frame to both LRU cache and backward buffer.

        Args:
            frame_idx: Frame index
            frame: BGR24 numpy array
        """
        with self.lock:
            # Compress frame
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality]
            success, encoded = cv2.imencode('.jpg', frame, encode_params)

            if not success:
                return

            cached_frame = {
                'compressed': encoded,
                'shape': frame.shape,
                'dtype': frame.dtype
            }

            # Add to LRU cache
            self.lru_cache[frame_idx] = cached_frame
            if len(self.lru_cache) > self.lru_cache_size:
                self.lru_cache.popitem(last=False)

            # Add to backward buffer
            self.backward_buffer.append((frame_idx, cached_frame))

            # Update statistics
            self.stats['total_frames_added'] += 1
            self.stats['total_original_bytes'] += frame.nbytes
            self.stats['total_compressed_bytes'] += encoded.nbytes

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get frame from buffer (tries LRU cache first, then backward buffer).

        Args:
            frame_idx: Frame index to retrieve

        Returns:
            Decompressed BGR24 frame or None if not found
        """
        with self.lock:
            # Try LRU cache first
            cached = self.lru_cache.get(frame_idx)
            if cached:
                self.stats['lru_hits'] += 1
                self.lru_cache.move_to_end(frame_idx)  # Update LRU order
                return cv2.imdecode(cached['compressed'], cv2.IMREAD_COLOR)

            # Try backward buffer
            for idx, cached in reversed(self.backward_buffer):
                if idx == frame_idx:
                    self.stats['backward_hits'] += 1

                    # Promote to LRU cache
                    self.lru_cache[frame_idx] = cached
                    if len(self.lru_cache) > self.lru_cache_size:
                        self.lru_cache.popitem(last=False)

                    return cv2.imdecode(cached['compressed'], cv2.IMREAD_COLOR)

            # Not found in either buffer
            self.stats['lru_misses'] += 1
            self.stats['backward_misses'] += 1
            return None

    def check_backward_buffer_refill(self) -> bool:
        """
        Check if backward buffer needs refilling.

        Returns:
            True if refill needed and not already refilling
        """
        with self.lock:
            buffer_size = len(self.backward_buffer)
            needs_refill = (
                buffer_size < self.refill_threshold and
                not self.is_refilling
            )

            if needs_refill:
                self.is_refilling = True
                self.stats['refill_count'] += 1

            return needs_refill

    def refill_complete(self) -> None:
        """Mark backward buffer refill as complete."""
        with self.lock:
            self.is_refilling = False

    def get_backward_buffer_info(self) -> Tuple[int, int]:
        """
        Get backward buffer information.

        Returns:
            Tuple of (current_size, max_size)
        """
        with self.lock:
            return len(self.backward_buffer), self.backward_buffer_size

    def clear(self) -> None:
        """Clear all buffers."""
        with self.lock:
            self.lru_cache.clear()
            self.backward_buffer.clear()
            self.is_refilling = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        with self.lock:
            total_requests = (
                self.stats['lru_hits'] +
                self.stats['lru_misses']
            )

            lru_hit_rate = (
                (self.stats['lru_hits'] / total_requests * 100)
                if total_requests > 0 else 0
            )

            backward_hit_rate = (
                (self.stats['backward_hits'] / total_requests * 100)
                if total_requests > 0 else 0
            )

            total_hit_rate = (
                ((self.stats['lru_hits'] + self.stats['backward_hits']) /
                 total_requests * 100)
                if total_requests > 0 else 0
            )

            compression_ratio = (
                (self.stats['total_compressed_bytes'] /
                 self.stats['total_original_bytes'] * 100)
                if self.stats['total_original_bytes'] > 0 else 0
            )

            return {
                # Buffer sizes
                'lru_cache_size': len(self.lru_cache),
                'lru_cache_max': self.lru_cache_size,
                'backward_buffer_size': len(self.backward_buffer),
                'backward_buffer_max': self.backward_buffer_size,

                # Hit rates
                'lru_hit_rate_percent': lru_hit_rate,
                'backward_hit_rate_percent': backward_hit_rate,
                'total_hit_rate_percent': total_hit_rate,

                # Request counts
                'total_requests': total_requests,
                'lru_hits': self.stats['lru_hits'],
                'lru_misses': self.stats['lru_misses'],
                'backward_hits': self.stats['backward_hits'],

                # Memory usage
                'total_original_mb': self.stats['total_original_bytes'] / 1024 / 1024,
                'total_compressed_mb': self.stats['total_compressed_bytes'] / 1024 / 1024,
                'compression_ratio_percent': compression_ratio,
                'memory_saved_mb': (
                    (self.stats['total_original_bytes'] -
                     self.stats['total_compressed_bytes']) / 1024 / 1024
                ),

                # Operational stats
                'total_frames_added': self.stats['total_frames_added'],
                'refill_count': self.stats['refill_count'],
                'is_refilling': self.is_refilling
            }


# Example usage and benchmark
if __name__ == '__main__':
    import time
    import random

    print("Unified Frame Buffer Benchmark\n" + "="*60)

    # Create test frames
    test_frames = {
        i: np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for i in range(100)
    }

    # Initialize buffer
    buffer = UnifiedFrameBuffer(
        lru_cache_size=50,
        backward_buffer_size=600,
        compression_quality=95
    )

    # Simulate sequential playback with backward navigation
    print("\nSimulating video playback with backward navigation...")

    # Forward playback (frames 0-99)
    for i in range(100):
        buffer.add_frame(i, test_frames[i])

    # Random backward seeks (simulating arrow key navigation)
    for _ in range(50):
        current_frame = random.randint(10, 99)
        backward_frame = random.randint(0, current_frame - 1)

        # Try to get backward frame (should hit backward buffer)
        frame = buffer.get_frame(backward_frame)

    # Random seeks (simulating timeline scrubbing)
    for _ in range(50):
        random_frame = random.randint(0, 99)
        frame = buffer.get_frame(random_frame)

    # Print statistics
    stats = buffer.get_stats()
    print("\n" + "="*60)
    print("Buffer Statistics:")
    print("="*60)
    print(f"LRU Cache: {stats['lru_cache_size']}/{stats['lru_cache_max']}")
    print(f"Backward Buffer: {stats['backward_buffer_size']}/{stats['backward_buffer_max']}")
    print(f"\nHit Rates:")
    print(f"  LRU Cache: {stats['lru_hit_rate_percent']:.1f}%")
    print(f"  Backward Buffer: {stats['backward_hit_rate_percent']:.1f}%")
    print(f"  Total: {stats['total_hit_rate_percent']:.1f}%")
    print(f"\nMemory:")
    print(f"  Original: {stats['total_original_mb']:.2f} MB")
    print(f"  Compressed: {stats['total_compressed_mb']:.2f} MB")
    print(f"  Compression: {stats['compression_ratio_percent']:.1f}%")
    print(f"  Saved: {stats['memory_saved_mb']:.2f} MB")
    print(f"\nOperations:")
    print(f"  Frames Added: {stats['total_frames_added']}")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Refills: {stats['refill_count']}")
