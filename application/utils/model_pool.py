"""
SmartModelPool: Intelligent YOLO model instance management with resource optimization.

This module provides sustainable model instance management that prevents OOM by limiting
concurrent instances rather than moving models between devices. Maintains warm instance 
pools for reuse while respecting memory constraints.
"""

import torch
import gc
import logging
import os
import threading
import time
import queue
from typing import Dict, Optional, Any, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class ModelInstance:
    """Represents a model instance with metadata."""
    model: YOLO
    model_key: str
    device: str
    last_used: float
    in_use: bool
    creation_time: float


class SmartModelPool:
    """
    Smart model pool that manages YOLO model instances with sustainable resource management.
    
    Key Features:
    - Concurrent instance limiting (prevents OOM)
    - Warm instance pools for reuse
    - Device-aware loading (.engine stays GPU, others optimally placed)
    - Queue-based access with timeout handling
    - Memory pressure adaptation
    - No unnecessary create/destroy cycles
    """
    
    def __init__(self, 
                 max_concurrent_instances: int = 3,
                 max_instances_per_model: int = 2,
                 instance_timeout_seconds: float = 300.0,
                 max_gpu_memory_ratio: float = 0.85,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the smart model pool.
        
        Args:
            max_concurrent_instances: Total concurrent instances across all models
            max_instances_per_model: Maximum instances per unique model
            instance_timeout_seconds: Time before unused instances are eligible for cleanup
            max_gpu_memory_ratio: Memory threshold for adaptation (0.85 = 85%)
            logger: Optional logger instance
        """
        self.max_concurrent_instances = max_concurrent_instances
        self.max_instances_per_model = max_instances_per_model
        self.instance_timeout_seconds = instance_timeout_seconds
        self.max_gpu_memory_ratio = max_gpu_memory_ratio
        self.logger = logger or logging.getLogger('SmartModelPool')
        
        # Instance tracking
        self.instances: Dict[str, List[ModelInstance]] = {}  # model_key -> list of instances
        self.active_instances: int = 0
        self.waiting_queue: queue.Queue = queue.Queue()
        
        # Thread safety
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        
        # Background maintenance
        self.cleanup_thread = None
        self.shutdown_event = threading.Event()
        
        # Device detection and setup
        self.optimal_device, self.gpu_available, self.gpu_total_memory = self._detect_optimal_device()
        
        if self.gpu_available:
            memory_gb = self.gpu_total_memory / 1024**3 if self.gpu_total_memory > 0 else 0
            self.logger.info(f"SmartModelPool initialized with {self.optimal_device.upper()} support. "
                           f"Memory: {memory_gb:.1f}GB")
        else:
            self.logger.info("SmartModelPool initialized in CPU-only mode.")
        
        self.logger.info(f"Max concurrent instances: {max_concurrent_instances}, "
                       f"Max per model: {max_instances_per_model}")
        
        # Start background cleanup
        self._start_cleanup_thread()
    
    def _detect_optimal_device(self) -> Tuple[str, bool, int]:
        """
        Detect the optimal device for model execution.
        
        Returns:
            Tuple of (device_string, gpu_available, total_memory)
        """
        try:
            # Check for NVIDIA CUDA
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return 'cuda', True, total_memory
        except Exception as e:
            self.logger.debug(f"CUDA check failed: {e}")
        
        try:
            # Check for Apple Metal Performance Shaders (MPS) on Mac Silicon
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't expose memory info easily, estimate based on system
                import platform
                if 'arm64' in platform.machine().lower():  # Mac Silicon
                    estimated_memory = 8 * 1024**3  # 8GB estimate for unified memory
                    return 'mps', True, estimated_memory
        except Exception as e:
            self.logger.debug(f"MPS check failed: {e}")
        
        try:
            # Check for other GPU backends (OpenCL, ROCm, etc.)
            # Note: This is more limited as PyTorch doesn't expose these as easily
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return 'xpu', True, 0  # Intel GPU
        except Exception as e:
            self.logger.debug(f"XPU check failed: {e}")
        
        # Fallback to CPU
        return 'cpu', False, 0
    
    @contextmanager
    def get_model(self, model_path: str, task: str = 'detect', timeout: float = 30.0):
        """
        Context manager for safe model instance access with queuing and resource management.
        
        Args:
            model_path: Path to the YOLO model file
            task: YOLO task type ('detect', 'pose', etc.)
            timeout: Maximum wait time for an available instance
            
        Yields:
            YOLO model instance
            
        Example:
            with model_pool.get_model('model.pt', 'detect') as model:
                results = model(frame)
        """
        model_key = f"{model_path}:{task}"
        instance = None
        
        try:
            # Get or wait for available instance
            instance = self._acquire_instance(model_key, model_path, task, timeout)
            if instance is None:
                raise TimeoutError(f"No model instance available within {timeout}s for {model_key}")
            
            # Mark as in use and update last used time
            with self.lock:
                instance.in_use = True
                instance.last_used = time.time()
            
            yield instance.model
            
        finally:
            # Release the instance
            if instance:
                self._release_instance(instance)
    
    def _acquire_instance(self, model_key: str, model_path: str, task: str, timeout: float) -> Optional[ModelInstance]:
        """Acquire a model instance, creating if necessary or waiting for availability."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.condition:
                # Try to find available instance
                available_instance = self._find_available_instance(model_key)
                if available_instance:
                    return available_instance
                
                # Check if we can create a new instance
                if self._can_create_instance(model_key):
                    return self._create_instance(model_key, model_path, task)
                
                # Wait for an instance to become available
                self.logger.debug(f"Waiting for available instance: {model_key}")
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout > 0:
                    self.condition.wait(remaining_timeout)
        
        return None
    
    def _find_available_instance(self, model_key: str) -> Optional[ModelInstance]:
        """Find an available (not in use) instance for the given model."""
        if model_key not in self.instances:
            return None
        
        for instance in self.instances[model_key]:
            if not instance.in_use:
                return instance
        
        return None
    
    def _can_create_instance(self, model_key: str) -> bool:
        """Check if we can create a new instance without exceeding limits."""
        # Check global concurrent limit
        if self.active_instances >= self.max_concurrent_instances:
            return False
        
        # Check per-model limit
        model_instance_count = len(self.instances.get(model_key, []))
        if model_instance_count >= self.max_instances_per_model:
            return False
        
        # Check memory pressure (adaptive limit)
        if self.gpu_available and self._is_gpu_memory_pressure():
            self.logger.warning("GPU memory pressure detected, deferring instance creation")
            return False
        
        return True
    
    def _create_instance(self, model_key: str, model_path: str, task: str) -> ModelInstance:
        """Create a new model instance with device-aware placement."""
        self.logger.info(f"Creating new model instance: {model_key}")
        
        try:
            # Load model with device-aware logic
            model = self._load_model_device_aware(model_path, task)
            
            # Determine target device
            device = self._get_target_device(model_path)
            
            # Create instance
            instance = ModelInstance(
                model=model,
                model_key=model_key,
                device=device,
                last_used=time.time(),
                in_use=False,
                creation_time=time.time()
            )
            
            # Add to tracking
            if model_key not in self.instances:
                self.instances[model_key] = []
            self.instances[model_key].append(instance)
            self.active_instances += 1
            
            self.logger.info(f"Created instance on {device}: {model_key} "
                           f"(active: {self.active_instances}/{self.max_concurrent_instances})")
            
            return instance
            
        except Exception as e:
            self.logger.error(f"Failed to create model instance {model_key}: {e}")
            raise
    
    def _load_model_device_aware(self, model_path: str, task: str) -> YOLO:
        """Load model with intelligent device placement."""
        model_ext = model_path.lower().split('.')[-1]
        
        if model_ext == 'mlpackage':
            # CoreML models - try PyTorch fallback first
            pt_path = model_path.replace('.mlpackage', '.pt')
            if os.path.exists(pt_path):
                self.logger.info(f"Using PyTorch equivalent for CoreML: {pt_path}")
                model = YOLO(pt_path, task=task)
            else:
                try:
                    model = YOLO(model_path, task=task)
                except Exception as e:
                    self.logger.warning(f"CoreML loading failed, no fallback: {e}")
                    raise
        else:
            # Standard models (.pt, .onnx, .engine)
            model = YOLO(model_path, task=task)
        
        # Device placement with multi-device support
        target_device = self._get_target_device(model_path)
        
        if model_ext == 'engine':
            # TensorRT engines are CUDA-specific and GPU-bound
            if target_device != 'cuda':
                raise RuntimeError(f"TensorRT .engine models require CUDA, but optimal device is {target_device}")
            self.logger.debug(f"TensorRT engine loaded (CUDA-bound): {model_path}")
        elif target_device != 'cpu':
            # Move to optimal GPU device (cuda/mps/xpu)
            try:
                model.to(target_device)
                self.logger.debug(f"Model moved to {target_device.upper()}: {model_path}")
            except Exception as e:
                self.logger.warning(f"{target_device.upper()} placement failed, using CPU: {e}")
                model.to('cpu')
                target_device = 'cpu'
        else:
            # CPU placement
            model.to('cpu')
            self.logger.debug(f"Model kept on CPU: {model_path}")
        
        return model
    
    def _get_target_device(self, model_path: str) -> str:
        """Determine target device for model based on format and resources."""
        model_ext = model_path.lower().split('.')[-1]
        
        # TensorRT engines must stay on CUDA specifically
        if model_ext == 'engine':
            if self.optimal_device == 'cuda':
                return 'cuda'
            else:
                raise RuntimeError(f"TensorRT .engine models require CUDA, but only {self.optimal_device} is available")
        
        # For other models, use optimal device if no memory pressure
        if self.gpu_available and not self._is_gpu_memory_pressure():
            return self.optimal_device
        else:
            return 'cpu'
    
    def _release_instance(self, instance: ModelInstance):
        """Release an instance back to the pool."""
        with self.condition:
            instance.in_use = False
            instance.last_used = time.time()
            # Notify waiting threads
            self.condition.notify_all()
        
        self.logger.debug(f"Released instance: {instance.model_key}")
    
    def _is_gpu_memory_pressure(self) -> bool:
        """Check if GPU memory usage is above threshold."""
        if not self.gpu_available:
            return False
        
        try:
            if self.optimal_device == 'cuda':
                allocated = torch.cuda.memory_allocated()
                if self.gpu_total_memory > 0:
                    usage_ratio = allocated / self.gpu_total_memory
                    return usage_ratio > self.max_gpu_memory_ratio
            elif self.optimal_device == 'mps':
                # MPS doesn't provide detailed memory stats, use conservative estimate
                # Based on number of active instances
                if self.active_instances >= self.max_concurrent_instances * 0.8:
                    return True
            # For other devices, use instance count as proxy
            return self.active_instances >= self.max_concurrent_instances
        except Exception:
            return False
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup of unused instances."""
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker that periodically cleans up unused instances."""
        while not self.shutdown_event.is_set():
            try:
                # Sleep for cleanup interval
                self.shutdown_event.wait(timeout=60.0)  # Check every minute
                
                if self.shutdown_event.is_set():
                    break
                
                self._cleanup_expired_instances()
                
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
    
    def _cleanup_expired_instances(self):
        """Remove instances that haven't been used for the timeout period."""
        current_time = time.time()
        instances_removed = 0
        
        with self.lock:
            for model_key in list(self.instances.keys()):
                instances_list = self.instances[model_key]
                
                # Find expired unused instances
                expired_instances = []
                for instance in instances_list:
                    if (not instance.in_use and 
                        current_time - instance.last_used > self.instance_timeout_seconds):
                        expired_instances.append(instance)
                
                # Remove expired instances (but keep at least 1 per model for warmth)
                for instance in expired_instances:
                    if len(instances_list) > 1:  # Keep at least one warm instance
                        self._destroy_instance(instance)
                        instances_list.remove(instance)
                        instances_removed += 1
                
                # Clean up empty model entries
                if not instances_list:
                    del self.instances[model_key]
        
        if instances_removed > 0:
            self.logger.info(f"Cleaned up {instances_removed} expired instances")
    
    def _destroy_instance(self, instance: ModelInstance):
        """Safely destroy a model instance and free its resources."""
        try:
            model = instance.model
            
            # Move to CPU before deletion (for PyTorch models)
            if hasattr(model, 'cpu') and callable(model.cpu):
                try:
                    model.cpu()
                except Exception:
                    pass  # Some models might not support .cpu()
            
            # Clean up references
            del instance.model
            self.active_instances -= 1
            
            self.logger.debug(f"Destroyed instance: {instance.model_key} "
                            f"(active: {self.active_instances}/{self.max_concurrent_instances})")
            
        except Exception as e:
            self.logger.warning(f"Error destroying instance {instance.model_key}: {e}")
    
    def clear_all_instances(self):
        """Clear all instances and free memory - for shutdown or emergency cleanup."""
        with self.lock:
            total_instances = 0
            for model_key, instances_list in self.instances.items():
                for instance in instances_list:
                    self._destroy_instance(instance)
                    total_instances += 1
            
            self.instances.clear()
            self.active_instances = 0
        
        # Force cleanup based on device
        if self.gpu_available:
            try:
                if self.optimal_device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.optimal_device == 'mps':
                    # MPS cleanup
                    torch.mps.empty_cache()
                # Other devices don't typically have explicit cache cleanup
            except Exception as e:
                self.logger.debug(f"Cache cleanup failed: {e}")
        gc.collect()
        
        try:
            self.logger.info(f"Cleared {total_instances} instances from pool")
        except ValueError:
            pass # Ignore I/O operation on closed file during shutdown
    
    def adapt_to_memory_pressure(self):
        """Dynamically adapt instance limits based on current memory pressure."""
        if not self.gpu_available:
            return
        
        try:
            usage_ratio = 0.0
            device_name = self.optimal_device.upper()
            
            if self.optimal_device == 'cuda':
                allocated = torch.cuda.memory_allocated()
                if self.gpu_total_memory > 0:
                    usage_ratio = allocated / self.gpu_total_memory
            elif self.optimal_device == 'mps':
                # For MPS, use instance count as proxy for memory usage
                usage_ratio = self.active_instances / self.max_concurrent_instances
            else:
                # For other devices, use instance count
                usage_ratio = self.active_instances / self.max_concurrent_instances
            
            if usage_ratio > 0.9:  # Critical memory pressure
                # Reduce concurrent limit temporarily
                self.max_concurrent_instances = max(1, self.max_concurrent_instances - 1)
                self.logger.warning(f"Critical memory pressure detected on {device_name} ({usage_ratio:.1%}), "
                                  f"reducing concurrent limit to {self.max_concurrent_instances}")
                
                # Force cleanup of unused instances
                with self.lock:
                    self._cleanup_expired_instances()
                
            elif usage_ratio < 0.7 and self.max_concurrent_instances < 3:
                # Memory available, can increase limit
                self.max_concurrent_instances = min(3, self.max_concurrent_instances + 1)
                self.logger.info(f"Memory pressure reduced on {device_name} ({usage_ratio:.1%}), "
                               f"increasing concurrent limit to {self.max_concurrent_instances}")
                
        except Exception as e:
            self.logger.warning(f"Error adapting to memory pressure: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage and instance statistics."""
        with self.lock:
            total_instances = sum(len(instances) for instances in self.instances.values())
            in_use_instances = sum(
                sum(1 for instance in instances if instance.in_use)
                for instances in self.instances.values()
            )
            unique_models = len(self.instances)
        
        stats = {
            'total_instances': total_instances,
            'active_instances': self.active_instances,
            'in_use_instances': in_use_instances,
            'unique_models': unique_models,
            'max_concurrent': self.max_concurrent_instances,
            'max_per_model': self.max_instances_per_model,
            'optimal_device': self.optimal_device,
            'gpu_available': self.gpu_available,
            'memory_pressure': self._is_gpu_memory_pressure(),
        }
        
        if self.gpu_available:
            try:
                if self.optimal_device == 'cuda':
                    allocated = torch.cuda.memory_allocated()
                    cached = torch.cuda.memory_cached()
                    stats.update({
                        'cuda_allocated_mb': allocated / 1024**2,
                        'cuda_cached_mb': cached / 1024**2,
                        'cuda_total_mb': self.gpu_total_memory / 1024**2,
                        'cuda_usage_ratio': allocated / self.gpu_total_memory if self.gpu_total_memory > 0 else 0,
                    })
                elif self.optimal_device == 'mps':
                    # MPS doesn't expose detailed memory stats
                    stats.update({
                        'mps_available': True,
                        'mps_estimated_memory_gb': self.gpu_total_memory / 1024**3 if self.gpu_total_memory > 0 else 8,
                        'mps_instance_usage': self.active_instances / self.max_concurrent_instances,
                    })
                else:
                    stats.update({
                        f'{self.optimal_device}_available': True,
                        f'{self.optimal_device}_instance_usage': self.active_instances / self.max_concurrent_instances,
                    })
            except Exception as e:
                stats['device_error'] = f'Unable to get {self.optimal_device.upper()} stats: {e}'
        
        return stats
    
    def cleanup(self):
        """Explicit cleanup method for external resource management."""
        self.shutdown_event.set()
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        self.clear_all_instances()
    
    def __del__(self):
        """Cleanup when pool is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid errors during cleanup


# Backward compatibility alias
ModelPool = SmartModelPool