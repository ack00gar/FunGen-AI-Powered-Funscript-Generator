# Video Processor Optimization - Implementation Guide

**Related Document:** [VIDEO_PROCESSOR_REVIEW.md](VIDEO_PROCESSOR_REVIEW.md)

---

## Quick Start

This guide provides step-by-step instructions for implementing the video processor optimizations.

### File Structure

```
video/
├── optimizations/           # NEW - Ready-to-use optimization modules
│   ├── frame_cache_compressed.py      # Quick Win #1 (83% memory reduction)
│   ├── unified_frame_buffer.py        # Quick Win #4 (simplified architecture)
│   ├── performance_metrics.py         # Performance monitoring
│   ├── ffmpeg_command_builder.py      # Quick Win #2 (20-30% faster)
│   └── async_video_processor.py       # Stage 3 (async/await pattern)
├── video_processor.py       # Main processor (to be refactored)
├── dual_frame_processor.py
├── gpu_unwarp_worker.py
└── thumbnail_extractor.py
```

---

## Phase 1: Quick Wins (Week 1)

### Step 1: Integrate Compressed Frame Cache

**Impact:** 83% memory reduction (60MB → 10MB for 50-frame cache)

**Instructions:**

1. Import the compressed cache module in `video_processor.py`:

```python
from video.optimizations.frame_cache_compressed import CompressedFrameCache
```

2. Replace existing cache initialization (line ~144):

```python
# OLD:
self.frame_cache = OrderedDict()
self.frame_cache_max_size = cache_size

# NEW:
self.frame_cache = CompressedFrameCache(
    max_size=cache_size,
    compression_quality=95  # High quality, minimal artifacts
)
```

3. Update cache methods:

```python
# OLD: _add_frame_to_cache
def _add_frame_to_cache(self, frame_idx: int, frame: np.ndarray) -> None:
    with self.frame_cache_lock:
        self.frame_cache[frame_idx] = frame.copy()
        if len(self.frame_cache) > self.frame_cache_max_size:
            self.frame_cache.popitem(last=False)

# NEW: _add_frame_to_cache
def _add_frame_to_cache(self, frame_idx: int, frame: np.ndarray) -> None:
    # No lock needed - CompressedFrameCache is thread-safe
    self.frame_cache.add(frame_idx, frame)

# OLD: _get_frame_from_cache
def _get_frame_from_cache(self, frame_idx: int) -> Optional[np.ndarray]:
    with self.frame_cache_lock:
        return self.frame_cache.get(frame_idx)

# NEW: _get_frame_from_cache
def _get_frame_from_cache(self, frame_idx: int) -> Optional[np.ndarray]:
    return self.frame_cache.get(frame_idx)
```

4. **Test:** Run existing tests to ensure no regression
5. **Benchmark:** Check memory usage before/after with `get_stats()`

---

### Step 2: Optimize FFmpeg Command Construction

**Impact:** 20-30% faster command building

**Instructions:**

1. Import the builder module in `video_processor.py`:

```python
from video.optimizations.ffmpeg_command_builder import (
    FFmpegCommandBuilder,
    VideoConfig
)
```

2. Initialize builder in `__init__`:

```python
def __init__(self, ...):
    # ... existing code ...
    self.ffmpeg_builder = FFmpegCommandBuilder()
```

3. Replace `build_ffmpeg_command` method (line ~500-800):

```python
# OLD: build_ffmpeg_command (complex string concatenation)

# NEW: build_ffmpeg_command (using builder)
def build_ffmpeg_command(self, start_frame: int = 0) -> List[str]:
    """Build FFmpeg command using optimized builder."""
    config = VideoConfig(
        video_path=self._active_video_source_path,
        output_size=self.yolo_input_size,
        video_type=self.determined_video_type,
        vr_format=self.vr_input_format,
        vr_fov=self.vr_fov,
        vr_pitch=self.vr_pitch,
        start_time=start_frame / self.fps if start_frame > 0 else None
    )

    return self.ffmpeg_builder.build_command(config)
```

4. **Test:** Verify FFmpeg commands match previous output
5. **Benchmark:** Time command construction (should be 20-30% faster)

---

### Step 3: Add Lazy GPU Worker Loading

**Impact:** Faster startup for 2D videos, reduced memory when not needed

**Instructions:**

1. Find GPU unwarp worker initialization (line ~1100-1200):

```python
# OLD: Eager initialization
def __init__(self, ...):
    # ...
    from video.gpu_unwarp_worker import GPUUnwarpWorker
    self.gpu_unwarp_worker = GPUUnwarpWorker(...)

# NEW: Lazy initialization
def __init__(self, ...):
    # ...
    self.gpu_unwarp_worker = None
    self.gpu_unwarp_enabled = False
```

2. Add lazy loader method:

```python
def _ensure_gpu_unwarp_worker(self):
    """Lazy-load GPU unwarp worker only when needed."""
    if self.gpu_unwarp_worker is None and self.determined_video_type == 'vr':
        from video.gpu_unwarp_worker import GPUUnwarpWorker

        self.logger.info("🔧 Initializing GPU unwarp worker...")

        self.gpu_unwarp_worker = GPUUnwarpWorker(
            video_processor=self,
            unwarp_method=self.vr_unwarp_method_override,
            logger=self.logger
        )
        self.gpu_unwarp_enabled = self.gpu_unwarp_worker.start()

        if self.gpu_unwarp_enabled:
            self.logger.info("✅ GPU unwarp worker started")
        else:
            self.logger.warning("⚠️ GPU unwarp unavailable, using v360 filter")
```

3. Call lazy loader before use:

```python
# In process_frame or similar:
if self.determined_video_type == 'vr':
    self._ensure_gpu_unwarp_worker()
    if self.gpu_unwarp_enabled:
        # Use GPU unwarp
        ...
```

4. **Test:** Load 2D video (should be faster), then load VR video (GPU worker should initialize)

---

### Step 4: Integrate Performance Metrics

**Impact:** Better visibility, data-driven optimization

**Instructions:**

1. Import metrics module:

```python
from video.optimizations.performance_metrics import PerformanceMetrics
```

2. Initialize in `__init__`:

```python
def __init__(self, ...):
    # ...
    self.performance_metrics = PerformanceMetrics(max_samples=1000)
```

3. Record metrics in processing loop:

```python
# In _read_frame_from_ffmpeg or similar:
import time

start = time.perf_counter()
# ... decode frame ...
decode_time_ms = (time.perf_counter() - start) * 1000

self.performance_metrics.record_decode(decode_time_ms)

# Sample resources periodically
if frame_index % 10 == 0:
    self.performance_metrics.sample_resource_usage()
```

4. Add metrics endpoint for UI:

```python
def get_performance_stats(self) -> dict:
    """Get performance statistics for UI display."""
    return self.performance_metrics.get_stats()

def get_performance_summary(self) -> str:
    """Get human-readable performance summary."""
    return self.performance_metrics.get_summary()
```

5. **Test:** Run video playback, print summary at end

---

### Step 5: Reduce Logging Overhead

**Impact:** 5-10% CPU reduction during playback

**Instructions:**

1. Add rate-limited logger helper:

```python
class RateLimitedLogger:
    """Rate-limited logging to reduce overhead."""
    def __init__(self, logger, min_interval=1.0):
        self.logger = logger
        self.min_interval = min_interval
        self.last_log = {}

    def debug(self, key: str, message: str):
        """Log message if enough time has passed."""
        now = time.time()
        if now - self.last_log.get(key, 0) > self.min_interval:
            self.logger.debug(message)
            self.last_log[key] = now
```

2. Initialize in `__init__`:

```python
def __init__(self, ...):
    # ...
    self.rate_limited_logger = RateLimitedLogger(self.logger, min_interval=1.0)
```

3. Replace debug logs in hot paths:

```python
# OLD: Log every frame
self.logger.debug(f"Frame {frame_idx} decoded in {decode_time:.2f}ms")

# NEW: Log once per second
self.rate_limited_logger.debug(
    'decode_timing',
    f"Frame {frame_idx} decoded in {decode_time:.2f}ms"
)
```

4. **Benchmark:** CPU usage should drop 5-10% during playback

---

## Phase 2: Unified Frame Buffer (Week 2)

### Step 1: Replace Multiple Buffers with Unified Buffer

**Impact:** Simpler code, better cache coherency

**Instructions:**

1. Import unified buffer:

```python
from video.optimizations.unified_frame_buffer import UnifiedFrameBuffer
```

2. Replace cache and backward buffer initialization (line ~144-156):

```python
# OLD:
self.frame_cache = OrderedDict()
self.frame_cache_max_size = cache_size
self.frame_cache_lock = threading.Lock()

from collections import deque
self.arrow_nav_backward_buffer = deque(maxlen=buffer_size)
self.arrow_nav_backward_buffer_lock = threading.Lock()

# NEW:
self.frame_buffer = UnifiedFrameBuffer(
    lru_cache_size=cache_size,
    backward_buffer_size=buffer_size,
    compression_quality=95,
    refill_threshold=max(120, buffer_size // 5)
)
```

3. Update all cache/buffer access points:

```python
# OLD:
def _add_frame_to_cache(self, frame_idx, frame):
    with self.frame_cache_lock:
        self.frame_cache[frame_idx] = frame
    with self.arrow_nav_backward_buffer_lock:
        self.arrow_nav_backward_buffer.append((frame_idx, frame))

# NEW:
def _add_frame_to_buffer(self, frame_idx, frame):
    self.frame_buffer.add_frame(frame_idx, frame)

# OLD:
def _get_frame_from_cache(self, frame_idx):
    # Complex logic checking both cache and buffer

# NEW:
def _get_frame_from_buffer(self, frame_idx):
    return self.frame_buffer.get_frame(frame_idx)
```

4. Update refill logic:

```python
# OLD: Complex refill checking

# NEW:
if self.frame_buffer.check_backward_buffer_refill():
    # Start refill
    self._start_backward_buffer_refill()
```

5. **Test:** Verify cache hit rates match or exceed previous implementation

---

## Phase 3: Testing & Validation

### Regression Tests

Create test file `tests/test_video_processor_optimizations.py`:

```python
import pytest
import numpy as np
from video.video_processor import VideoProcessor

def test_compressed_cache_quality():
    """Ensure compressed cache maintains quality."""
    # Create test frame
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Add to cache
    cache = CompressedFrameCache(max_size=10)
    cache.add(0, frame)

    # Retrieve
    cached_frame = cache.get(0)

    # Check similarity (JPEG compression may have minor differences)
    mse = np.mean((frame.astype(float) - cached_frame.astype(float)) ** 2)
    assert mse < 10, "Compressed frame differs too much from original"

def test_cache_memory_reduction():
    """Verify memory reduction from compression."""
    cache = CompressedFrameCache(max_size=50)

    # Add 50 frames
    for i in range(50):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cache.add(i, frame)

    stats = cache.get_stats()

    # Should achieve at least 70% compression
    assert stats['compression_ratio_percent'] < 30, "Compression ratio too low"
    assert stats['memory_saved_mb'] > 40, "Memory savings too low"

def test_ffmpeg_command_consistency():
    """Ensure new builder produces valid commands."""
    builder = FFmpegCommandBuilder()

    config = VideoConfig(
        video_path='/test.mp4',
        output_size=640,
        video_type='2d'
    )

    cmd = builder.build_command(config)

    # Verify essential arguments
    assert 'ffmpeg' in cmd
    assert '-i' in cmd
    assert '/test.mp4' in cmd
    assert 'scale=640:640' in ' '.join(cmd)
    assert 'pipe:1' in cmd
```

Run tests:

```bash
pytest tests/test_video_processor_optimizations.py -v
```

### Performance Benchmarks

Create benchmark file `benchmarks/benchmark_optimizations.py`:

```python
import time
import psutil
from video.video_processor import VideoProcessor

def benchmark_cache_performance():
    """Benchmark cache operations."""
    from video.optimizations.frame_cache_compressed import CompressedFrameCache

    cache = CompressedFrameCache(max_size=50)
    test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Benchmark add
    start = time.perf_counter()
    for i in range(100):
        cache.add(i, test_frame)
    add_time = (time.perf_counter() - start) * 1000

    # Benchmark get
    start = time.perf_counter()
    for i in range(50):
        frame = cache.get(i)
    get_time = (time.perf_counter() - start) * 1000

    stats = cache.get_stats()

    print(f"Cache Performance:")
    print(f"  Add: {add_time:.2f}ms for 100 frames ({add_time/100:.2f}ms avg)")
    print(f"  Get: {get_time:.2f}ms for 50 frames ({get_time/50:.2f}ms avg)")
    print(f"  Memory: {stats['total_compressed_mb']:.2f}MB")
    print(f"  Compression: {stats['compression_ratio_percent']:.1f}%")

if __name__ == '__main__':
    benchmark_cache_performance()
```

Run benchmarks:

```bash
python benchmarks/benchmark_optimizations.py
```

---

## Phase 4: Advanced Optimizations (Future)

### Async/Await Migration (Stage 3)

See `video/optimizations/async_video_processor.py` for reference implementation.

**Steps:**
1. Identify I/O-heavy operations (FFmpeg pipe reads, file I/O)
2. Convert to async methods using `asyncio`
3. Replace threading with coroutines
4. Test thoroughly (async introduces different error patterns)

### PyAV Integration (Stage 3)

**Steps:**
1. Install PyAV: `pip install av`
2. Create wrapper class (see VIDEO_PROCESSOR_REVIEW.md for example)
3. Benchmark against subprocess approach
4. Gradually migrate if benefits are clear

---

## Monitoring & Metrics

### Key Metrics to Track

Before and after each optimization, track:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Decode latency (p95) | < 70ms | `performance_metrics.get_stats()['decode_latency']['p95']` |
| Seek latency (p95) | < 400ms | `performance_metrics.get_stats()['seek_latency']['p95']` |
| Memory usage | < 640MB | `performance_metrics.get_stats()['memory_usage']['mean']` |
| CPU usage (idle) | < 18% | `performance_metrics.get_stats()['cpu_usage']['mean']` |
| Cache hit rate | > 90% | `frame_buffer.get_stats()['total_hit_rate_percent']` |

### Performance Dashboard

Add to UI (optional):

```python
def render_performance_dashboard(self):
    """Render performance metrics in UI."""
    stats = self.video_processor.performance_metrics.get_stats()

    imgui.text("Performance Metrics")
    imgui.separator()

    # Latency
    imgui.text(f"Decode (p95): {stats['decode_latency']['p95']:.1f}ms")
    imgui.text(f"Seek (p95):   {stats['seek_latency']['p95']:.1f}ms")

    # Resources
    imgui.text(f"Memory:       {stats['memory_usage']['mean']:.0f}MB")
    imgui.text(f"CPU:          {stats['cpu_usage']['mean']:.1f}%")

    # Cache
    buffer_stats = self.video_processor.frame_buffer.get_stats()
    imgui.text(f"Cache Hit:    {buffer_stats['total_hit_rate_percent']:.1f}%")
```

---

## Troubleshooting

### Common Issues

**1. JPEG Compression Artifacts**

*Symptom:* Visual quality degradation in cached frames

*Solution:* Increase compression quality

```python
# Change from:
cache = CompressedFrameCache(compression_quality=85)

# To:
cache = CompressedFrameCache(compression_quality=95)  # Minimal artifacts
```

**2. Cache Misses Increasing**

*Symptom:* Lower hit rate after optimization

*Solution:* Increase cache size or check seek patterns

```python
# Increase LRU cache size
frame_buffer = UnifiedFrameBuffer(
    lru_cache_size=100,  # Was 50
    backward_buffer_size=600
)
```

**3. Performance Degradation**

*Symptom:* Slower after optimization

*Solution:* Profile to find bottleneck

```python
import cProfile

profiler = cProfile.Profile()
profiler.enable()

# ... run video processing ...

profiler.disable()
profiler.print_stats(sort='cumtime')
```

---

## Rollback Plan

If optimizations cause issues:

1. **Quick Rollback:** Revert individual changes using git

```bash
git diff video/video_processor.py
git checkout video/video_processor.py  # Revert file
```

2. **Gradual Rollback:** Disable optimizations via flags

```python
USE_COMPRESSED_CACHE = False  # Add flag to disable

if USE_COMPRESSED_CACHE:
    self.frame_cache = CompressedFrameCache(...)
else:
    self.frame_cache = OrderedDict()  # Original
```

---

## Success Criteria

✅ **Phase 1 Complete When:**
- Memory usage reduced by > 15%
- CPU usage reduced by > 5%
- All tests passing
- No visual quality degradation
- Seek/playback performance maintained or improved

✅ **Phase 2 Complete When:**
- Code simplified (fewer classes, single buffer interface)
- Cache hit rate > 90%
- All tests passing
- Performance metrics match or exceed Phase 1

---

## Next Steps

After completing quick wins:

1. **Gather Metrics:** Run for 1 week, collect performance data
2. **Analyze Results:** Compare before/after metrics
3. **Plan Stage 2:** Based on data, prioritize next optimizations
4. **Iterate:** Continuous improvement based on real-world usage

---

## References

- [VIDEO_PROCESSOR_REVIEW.md](VIDEO_PROCESSOR_REVIEW.md) - Full analysis and recommendations
- [video/optimizations/](video/optimizations/) - Ready-to-use optimization modules
- FFmpeg documentation: https://ffmpeg.org/documentation.html
- PyAV documentation: https://pyav.org/docs/

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Status:** Ready for Implementation
