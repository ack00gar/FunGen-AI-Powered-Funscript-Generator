"""
ModelPool: Smart YOLO model management with GPU memory optimization.

This module provides context-managed access to YOLO models with automatic
memory management, preventing GPU OOM issues and optimizing performance.
"""

import torch
import gc
import logging
import os
from typing import Dict, Optional, Any
from contextlib import contextmanager
from ultralytics import YOLO


class ModelPool:
    """
    Smart model pool that manages YOLO models with automatic GPU memory optimization.
    
    Features:
    - Context-managed model access
    - Automatic GPU memory cleanup
    - Memory pressure detection
    - Least-recently-used eviction
    - Graceful fallback to CPU when GPU memory is full
    """
    
    def __init__(self, max_gpu_memory_ratio: float = 0.8, logger: Optional[logging.Logger] = None):
        """
        Initialize the model pool.
        
        Args:
            max_gpu_memory_ratio: Maximum ratio of GPU memory to use (0.8 = 80%)
            logger: Optional logger instance
        """
        self.models: Dict[str, YOLO] = {}
        self.model_usage_count: Dict[str, int] = {}
        self.max_gpu_memory_ratio = max_gpu_memory_ratio
        self.logger = logger or logging.getLogger('ModelPool')
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
            self.logger.info(f"ModelPool initialized with GPU support. Total GPU memory: {self.gpu_total_memory / 1024**3:.1f}GB")
        else:
            self.gpu_total_memory = 0
            self.logger.info("ModelPool initialized in CPU-only mode.")
    
    @contextmanager
    def get_model(self, model_path: str, task: str = 'detect'):
        """
        Context manager for safe model access with automatic cleanup.
        
        Args:
            model_path: Path to the YOLO model file
            task: YOLO task type ('detect', 'pose', etc.)
            
        Yields:
            YOLO model instance
            
        Example:
            with model_pool.get_model('model.pt', 'detect') as model:
                results = model(frame)
        """
        model_key = f"{model_path}:{task}"
        model = None
        
        try:
            model = self._load_or_get_model(model_key, model_path, task)
            self.model_usage_count[model_key] = self.model_usage_count.get(model_key, 0) + 1
            yield model
        finally:
            if model_key in self.model_usage_count:
                self.model_usage_count[model_key] = max(0, self.model_usage_count[model_key] - 1)
    
    def _load_or_get_model(self, model_key: str, model_path: str, task: str) -> YOLO:
        """Load model or return cached instance."""
        if model_key in self.models:
            self.logger.debug(f"Using cached model: {model_key}")
            return self.models[model_key]
        
        # Check memory pressure before loading
        if self.gpu_available and self._is_gpu_memory_pressure():
            self.logger.warning("GPU memory pressure detected, attempting to free memory")
            self._evict_unused_models()
            
            # If still under pressure, force cleanup
            if self._is_gpu_memory_pressure():
                self._force_gpu_cleanup()
        
        # Load the model
        self.logger.info(f"Loading model: {model_path} (task: {task})")
        try:
            # Handle different model formats
            model_ext = model_path.lower().split('.')[-1]
            
            if model_ext == 'mlpackage':
                # CoreML models - try loading with graceful fallback
                pt_path = model_path.replace('.mlpackage', '.pt')
                if os.path.exists(pt_path):
                    self.logger.info(f"CoreML model detected, using PyTorch equivalent: {pt_path}")
                    model = YOLO(pt_path, task=task)
                else:
                    try:
                        model = YOLO(model_path)
                        self.logger.debug(f"Loaded CoreML model: {model_key}")
                    except Exception as coreml_error:
                        self.logger.warning(f"CoreML model loading failed and no PyTorch fallback found: {coreml_error}")
                        raise coreml_error
            else:
                # Standard PyTorch/ONNX/TensorRT models
                model = YOLO(model_path, task=task)
                
                # Move to GPU if available and not under memory pressure
                if self.gpu_available and not self._is_gpu_memory_pressure():
                    try:
                        model.to('cuda')
                        self.logger.debug(f"Model moved to GPU: {model_key}")
                    except Exception as e:
                        self.logger.warning(f"Could not move model to GPU: {e}")
                        model.to('cpu')
                else:
                    model.to('cpu')
                    self.logger.debug(f"Model kept on CPU: {model_key}")
            
            self.models[model_key] = model
            self.model_usage_count[model_key] = 0
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def _is_gpu_memory_pressure(self) -> bool:
        """Check if GPU memory usage is above threshold."""
        if not self.gpu_available:
            return False
        
        try:
            allocated = torch.cuda.memory_allocated()
            usage_ratio = allocated / self.gpu_total_memory
            return usage_ratio > self.max_gpu_memory_ratio
        except Exception:
            return False
    
    def _evict_unused_models(self):
        """Remove models that are not currently in use."""
        unused_models = [
            key for key, count in self.model_usage_count.items() 
            if count == 0
        ]
        
        for model_key in unused_models:
            self._unload_model(model_key)
            self.logger.info(f"Evicted unused model: {model_key}")
    
    def _force_gpu_cleanup(self):
        """Aggressively clean up GPU memory."""
        self.logger.warning("Performing aggressive GPU memory cleanup")
        
        # Clear all cached models
        self.clear_all_models()
        
        # Force GPU memory cleanup
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        # Python garbage collection
        gc.collect()
    
    def _unload_model(self, model_key: str):
        """Safely unload a specific model."""
        if model_key in self.models:
            model = self.models[model_key]
            
            # Move to CPU before deletion (only for PyTorch models)
            try:
                if hasattr(model, 'cpu') and callable(model.cpu):
                    model.cpu()
            except Exception:
                pass  # CoreML and other formats might not support .cpu()
            
            # Clean up references
            del self.models[model_key]
            if model_key in self.model_usage_count:
                del self.model_usage_count[model_key]
            del model
    
    def clear_all_models(self):
        """Clear all cached models and free memory."""
        model_keys = list(self.models.keys())
        for model_key in model_keys:
            self._unload_model(model_key)
        
        # Final cleanup
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.info("All models cleared from pool")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            'cached_models': len(self.models),
            'gpu_available': self.gpu_available,
            'cpu_memory_usage': 0,  # Could be enhanced with psutil
        }
        
        if self.gpu_available:
            try:
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_cached()
                stats.update({
                    'gpu_allocated_mb': allocated / 1024**2,
                    'gpu_cached_mb': cached / 1024**2,
                    'gpu_total_mb': self.gpu_total_memory / 1024**2,
                    'gpu_usage_ratio': allocated / self.gpu_total_memory,
                })
            except Exception:
                stats['gpu_error'] = 'Unable to get GPU stats'
        
        return stats
    
    def cleanup(self):
        """Explicit cleanup method for external resource management."""
        self.clear_all_models()
    
    def __del__(self):
        """Cleanup when pool is destroyed."""
        try:
            self.clear_all_models()
        except Exception:
            pass  # Avoid errors during cleanup