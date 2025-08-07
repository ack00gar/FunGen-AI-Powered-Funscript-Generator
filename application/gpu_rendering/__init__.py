"""
GPU Timeline Rendering System
High-performance GPU-accelerated timeline rendering for VR Funscript AI Generator.
"""

# Version info
__version__ = "1.0.0"
__author__ = "VR Funscript AI Generator"

# Optional imports with fallbacks
try:
    from .gpu_timeline_renderer import GPUTimelineRenderer, RenderingMode
    from .texture_cache import TimelineTextureCache, CacheLayer
    from .gpu_timeline_integration import GPUTimelineIntegration, RenderBackend
    
    GPU_RENDERING_AVAILABLE = True
    
except ImportError as e:
    # Graceful fallback if OpenGL dependencies are missing
    GPU_RENDERING_AVAILABLE = False
    
    # Create mock classes for compatibility
    class GPUTimelineRenderer:
        def __init__(self, *args, **kwargs):
            pass
        def initialize_gpu_resources(self):
            return False
    
    class GPUTimelineIntegration:
        def __init__(self, *args, **kwargs):
            pass
        def render_timeline_optimized(self, *args, **kwargs):
            return False
    
    class RenderBackend:
        AUTO = "auto"
        GPU_INSTANCED = "gpu"
        CPU_IMGUI = "cpu"

__all__ = [
    'GPUTimelineRenderer',
    'TimelineTextureCache', 
    'GPUTimelineIntegration',
    'RenderBackend',
    'GPU_RENDERING_AVAILABLE'
]