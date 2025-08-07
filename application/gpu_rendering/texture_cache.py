#!/usr/bin/env python3
"""
Timeline Texture Caching System
Optimizes timeline rendering by caching static content to textures.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

try:
    import OpenGL.GL as gl
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

@dataclass
class TextureCacheEntry:
    """Represents a cached texture with metadata"""
    texture_id: int
    width: int
    height: int
    data_hash: str
    creation_time: float
    last_access_time: float
    usage_count: int

class CacheLayer(Enum):
    """Different layers of the timeline that can be cached"""
    STATIC_BACKGROUND = "static_background"  # Grid, labels, static elements
    FUNSCRIPT_LINES = "funscript_lines"     # Connecting lines between points
    WAVEFORM_DATA = "waveform_data"         # Audio waveform if present
    UI_OVERLAYS = "ui_overlays"             # Selection boxes, hover effects

class TimelineTextureCache:
    """
    High-performance texture caching system for timeline rendering.
    
    Features:
    - Layer-based caching (background, lines, overlays)
    - Dirty region tracking for partial updates
    - Automatic cache invalidation
    - Memory-efficient texture management
    - LRU eviction for memory pressure
    """
    
    def __init__(self, 
                 max_cache_size_mb: int = 512,
                 max_texture_age_seconds: int = 300,
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or logging.getLogger(__name__)
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_texture_age = max_texture_age_seconds
        
        # Cache storage
        self.cache_entries: Dict[str, TextureCacheEntry] = {}
        self.current_cache_size = 0
        
        # Dirty region tracking
        self.dirty_regions: Dict[CacheLayer, List[Tuple[int, int, int, int]]] = {
            layer: [] for layer in CacheLayer
        }
        
        # Performance metrics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_used_mb': 0,
            'render_time_saved_ms': 0
        }
        
        # OpenGL state
        self.gl_initialized = False
        
    def initialize(self) -> bool:
        """Initialize texture cache system"""
        
        if not OPENGL_AVAILABLE:
            self.logger.warning("OpenGL not available - texture caching disabled")
            return False
        
        try:
            # Initialize OpenGL texture resources
            self.gl_initialized = True
            self.logger.info("Texture cache system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize texture cache: {e}")
            return False
    
    def get_or_create_texture(self,
                            cache_key: str,
                            layer: CacheLayer,
                            width: int,
                            height: int,
                            render_func: callable) -> Optional[int]:
        """
        Get cached texture or create new one if not cached.
        
        Args:
            cache_key: Unique identifier for this texture content
            layer: Which layer this texture belongs to
            width: Texture width in pixels
            height: Texture height in pixels
            render_func: Function to call if texture needs to be rendered
            
        Returns:
            OpenGL texture ID or None if failed
        """
        
        if not self.gl_initialized:
            return None
        
        current_time = time.time()
        
        # Check if texture exists in cache
        if cache_key in self.cache_entries:
            entry = self.cache_entries[cache_key]
            
            # Update access time
            entry.last_access_time = current_time
            entry.usage_count += 1
            
            # Check if texture is still valid
            if self._is_texture_valid(entry, current_time):
                self.cache_stats['hits'] += 1
                return entry.texture_id
            else:
                # Texture expired, remove from cache
                self._evict_texture(cache_key)
        
        # Cache miss - need to create new texture
        self.cache_stats['misses'] += 1
        
        # Check if we need to free space
        estimated_size = width * height * 4  # RGBA
        if self.current_cache_size + estimated_size > self.max_cache_size_bytes:
            self._evict_old_textures(estimated_size)
        
        # Create new texture
        texture_id = self._create_texture(width, height, render_func)
        
        if texture_id is not None:
            # Add to cache
            data_hash = self._generate_content_hash(cache_key, width, height)
            entry = TextureCacheEntry(
                texture_id=texture_id,
                width=width,
                height=height,
                data_hash=data_hash,
                creation_time=current_time,
                last_access_time=current_time,
                usage_count=1
            )
            
            self.cache_entries[cache_key] = entry
            self.current_cache_size += estimated_size
            self._update_stats()
        
        return texture_id
    
    def invalidate_layer(self, layer: CacheLayer):
        """Invalidate all textures in a specific layer"""
        
        keys_to_remove = []
        for key, entry in self.cache_entries.items():
            if key.startswith(layer.value):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._evict_texture(key)
        
        # Clear dirty regions for this layer
        self.dirty_regions[layer] = []
        
        self.logger.debug(f"Invalidated {len(keys_to_remove)} textures from layer {layer.value}")
    
    def add_dirty_region(self, layer: CacheLayer, x: int, y: int, width: int, height: int):
        """Mark a region as dirty for incremental updates"""
        
        self.dirty_regions[layer].append((x, y, width, height))
        
        # Merge overlapping regions for efficiency
        self._merge_dirty_regions(layer)
    
    def get_dirty_regions(self, layer: CacheLayer) -> List[Tuple[int, int, int, int]]:
        """Get all dirty regions for a layer"""
        return self.dirty_regions[layer].copy()
    
    def clear_dirty_regions(self, layer: CacheLayer):
        """Clear dirty regions after they've been processed"""
        self.dirty_regions[layer] = []
    
    def update_partial_texture(self,
                             cache_key: str,
                             x: int, y: int, width: int, height: int,
                             render_func: callable) -> bool:
        """
        Update part of a cached texture (for incremental updates).
        
        This is much faster than re-rendering the entire texture.
        """
        
        if cache_key not in self.cache_entries:
            return False
        
        entry = self.cache_entries[cache_key]
        
        try:
            # Render partial content to temporary buffer
            partial_data = render_func(x, y, width, height)
            
            if OPENGL_AVAILABLE and partial_data is not None:
                # Update texture subregion on GPU
                # gl.glBindTexture(gl.GL_TEXTURE_2D, entry.texture_id)
                # gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, x, y, width, height,
                #                   gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, partial_data)
                
                # Update entry metadata
                entry.last_access_time = time.time()
                entry.usage_count += 1
                
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to update partial texture: {e}")
        
        return False
    
    def _create_texture(self, width: int, height: int, render_func: callable) -> Optional[int]:
        """Create new OpenGL texture"""
        
        if not OPENGL_AVAILABLE:
            return None
        
        try:
            # Render content to CPU buffer
            texture_data = render_func(width, height)
            
            if texture_data is None:
                return None
            
            # Create OpenGL texture
            # texture_id = gl.glGenTextures(1)
            # gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height,
            #                0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texture_data)
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            
            # For demo purposes, return a mock texture ID
            texture_id = len(self.cache_entries) + 1000
            
            self.logger.debug(f"Created texture {texture_id} ({width}x{height})")
            return texture_id
        
        except Exception as e:
            self.logger.error(f"Failed to create texture: {e}")
            return None
    
    def _evict_texture(self, cache_key: str):
        """Remove texture from cache and free GPU memory"""
        
        if cache_key in self.cache_entries:
            entry = self.cache_entries[cache_key]
            
            # Free GPU memory
            if OPENGL_AVAILABLE and entry.texture_id:
                # gl.glDeleteTextures([entry.texture_id])
                pass
            
            # Update cache size
            texture_size = entry.width * entry.height * 4
            self.current_cache_size -= texture_size
            
            # Remove from cache
            del self.cache_entries[cache_key]
            self.cache_stats['evictions'] += 1
    
    def _evict_old_textures(self, bytes_needed: int):
        """Evict old textures to make space for new ones"""
        
        current_time = time.time()
        
        # Sort textures by access time (LRU eviction)
        sorted_entries = sorted(
            self.cache_entries.items(),
            key=lambda x: x[1].last_access_time
        )
        
        bytes_freed = 0
        for cache_key, entry in sorted_entries:
            if bytes_freed >= bytes_needed:
                break
            
            # Don't evict very recently used textures
            if current_time - entry.last_access_time < 5.0:
                continue
            
            texture_size = entry.width * entry.height * 4
            self._evict_texture(cache_key)
            bytes_freed += texture_size
        
        self.logger.debug(f"Evicted textures to free {bytes_freed} bytes")
    
    def _is_texture_valid(self, entry: TextureCacheEntry, current_time: float) -> bool:
        """Check if cached texture is still valid"""
        
        # Check age
        if current_time - entry.creation_time > self.max_texture_age:
            return False
        
        # Check if texture still exists on GPU
        if OPENGL_AVAILABLE and entry.texture_id:
            # is_valid = gl.glIsTexture(entry.texture_id)
            is_valid = True  # Mock for demo
            return is_valid
        
        return True
    
    def _merge_dirty_regions(self, layer: CacheLayer):
        """Merge overlapping dirty regions to reduce update overhead"""
        
        regions = self.dirty_regions[layer]
        if len(regions) <= 1:
            return
        
        merged = []
        sorted_regions = sorted(regions, key=lambda r: (r[0], r[1]))
        
        current = sorted_regions[0]
        for next_region in sorted_regions[1:]:
            # Check if regions overlap or are adjacent
            if (current[0] + current[2] >= next_region[0] and
                current[1] + current[3] >= next_region[1]):
                # Merge regions
                right = max(current[0] + current[2], next_region[0] + next_region[2])
                bottom = max(current[1] + current[3], next_region[1] + next_region[3])
                current = (
                    min(current[0], next_region[0]),
                    min(current[1], next_region[1]),
                    right - min(current[0], next_region[0]),
                    bottom - min(current[1], next_region[1])
                )
            else:
                merged.append(current)
                current = next_region
        
        merged.append(current)
        self.dirty_regions[layer] = merged
    
    def _generate_content_hash(self, cache_key: str, width: int, height: int) -> str:
        """Generate hash for texture content"""
        content = f"{cache_key}_{width}_{height}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_stats(self):
        """Update performance statistics"""
        self.cache_stats['memory_used_mb'] = self.current_cache_size / (1024 * 1024)
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        hit_rate = 0
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_requests > 0:
            hit_rate = (self.cache_stats['hits'] / total_requests) * 100
        
        stats = self.cache_stats.copy()
        stats['hit_rate_percent'] = hit_rate
        stats['cached_textures'] = len(self.cache_entries)
        
        return stats
    
    def cleanup(self):
        """Clean up all cached textures and free GPU memory"""
        
        for cache_key in list(self.cache_entries.keys()):
            self._evict_texture(cache_key)
        
        self.gl_initialized = False
        self.logger.info("Texture cache cleaned up")

# Timeline-specific texture rendering functions
class TimelineTextureRenderer:
    """Renders timeline content to textures for caching"""
    
    @staticmethod
    def render_static_background(width: int, height: int) -> Optional[np.ndarray]:
        """Render static background elements (grid, labels, etc.)"""
        
        # Create RGBA buffer
        buffer = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Draw grid lines
        for x in range(0, width, 50):  # Vertical lines every 50px
            buffer[:, x:x+1, :3] = [64, 64, 64]  # Gray grid
            buffer[:, x:x+1, 3] = 128  # Semi-transparent
        
        for y in range(0, height, 25):  # Horizontal lines every 25px
            buffer[y:y+1, :, :3] = [64, 64, 64]
            buffer[y:y+1, :, 3] = 128
        
        return buffer
    
    @staticmethod
    def render_funscript_lines(actions_list: list, width: int, height: int,
                             zoom_factor: float, pan_offset: int) -> Optional[np.ndarray]:
        """Render funscript connecting lines to texture"""
        
        buffer = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Draw lines between points
        for i in range(len(actions_list) - 1):
            x1 = int((actions_list[i]["at"] - pan_offset) / zoom_factor)
            y1 = int(height - (actions_list[i]["pos"] / 100.0) * height)
            x2 = int((actions_list[i+1]["at"] - pan_offset) / zoom_factor)
            y2 = int(height - (actions_list[i+1]["pos"] / 100.0) * height)
            
            # Simple line drawing (in production, use proper line algorithm)
            if 0 <= x1 < width and 0 <= x2 < width:
                # Draw line (simplified)
                buffer[y1:y1+2, x1:x2, :3] = [200, 200, 255]  # Light blue
                buffer[y1:y1+2, x1:x2, 3] = 255  # Opaque
        
        return buffer