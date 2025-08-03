"""
ProcessingThreadManager: Advanced threading architecture for GPU-intensive operations.

This module provides thread-safe GPU processing with proper context management,
progress reporting, and UI responsiveness during heavy video analysis tasks.
"""

import threading
import queue
import time
import logging
import traceback
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

# GPU context management imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TaskType(Enum):
    """Enumeration of processing task types."""
    VIDEO_ANALYSIS = "video_analysis"
    OBJECT_DETECTION = "object_detection" 
    OPTICAL_FLOW = "optical_flow"
    STAGE_PROCESSING = "stage_processing"
    LIVE_TRACKING = "live_tracking"
    PREVIEW_GENERATION = "preview_generation"


class TaskPriority(Enum):
    """Task priority levels for processing queue."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Container for processing task information."""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    function: Callable
    args: tuple = ()
    kwargs: dict = None
    progress_callback: Optional[Callable[[float, str], None]] = None
    completion_callback: Optional[Callable[[Any], None]] = None
    error_callback: Optional[Callable[[Exception], None]] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    created_time: float = None
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_time is None:
            self.created_time = time.time()


class ProcessingThreadManager:
    """
    Advanced thread manager for GPU-intensive processing operations.
    
    Features:
    - Priority-based task queue
    - Thread-safe GPU context management
    - Real-time progress reporting
    - Automatic error handling and recovery
    - Cancellation support
    - Resource monitoring
    """
    
    def __init__(self, max_worker_threads: int = 2, logger: Optional[logging.Logger] = None):
        """
        Initialize the processing thread manager.
        
        Args:
            max_worker_threads: Maximum number of concurrent worker threads
            logger: Optional logger instance
        """
        self.max_worker_threads = max_worker_threads
        self.logger = logger or logging.getLogger('ProcessingThreadManager')
        
        # Thread management
        self.worker_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingTask] = {}
        
        # Thread safety
        self.task_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        
        # Progress tracking
        self.progress_callbacks: Dict[str, Callable[[str, float, str], None]] = {}
        self.global_progress_callback: Optional[Callable[[str, float, str], None]] = None
        
        # Performance monitoring
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0
        }
        
        # GPU context management
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.gpu_context_lock = threading.RLock()
        self.thread_gpu_contexts: Dict[str, bool] = {}
        
        if self.gpu_available:
            self.primary_gpu_device = torch.cuda.current_device()
            self.logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        else:
            self.primary_gpu_device = None
            self.logger.info("Running in CPU-only mode")
        
        # Start worker threads
        self._start_worker_threads()
        
        self.logger.info(f"ProcessingThreadManager initialized with {max_worker_threads} worker threads")
    
    def _start_worker_threads(self):
        """Start the worker threads."""
        for i in range(self.max_worker_threads):
            thread = threading.Thread(
                target=self._worker_thread_loop,
                name=f"ProcessingWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
            self.logger.debug(f"Started worker thread: {thread.name}")
    
    def _setup_thread_gpu_context(self, thread_name: str):
        """Setup GPU context for a worker thread."""
        if not self.gpu_available:
            return
        
        with self.gpu_context_lock:
            if thread_name not in self.thread_gpu_contexts:
                try:
                    # Set the GPU device for this thread
                    torch.cuda.set_device(self.primary_gpu_device)
                    
                    # Create a CUDA context for this thread
                    torch.cuda.current_stream()
                    
                    self.thread_gpu_contexts[thread_name] = True
                    self.logger.debug(f"GPU context setup for thread {thread_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to setup GPU context for thread {thread_name}: {e}")
                    self.thread_gpu_contexts[thread_name] = False
    
    def _cleanup_thread_gpu_context(self, thread_name: str):
        """Cleanup GPU context for a worker thread."""
        if not self.gpu_available:
            return
        
        with self.gpu_context_lock:
            if thread_name in self.thread_gpu_contexts and self.thread_gpu_contexts[thread_name]:
                try:
                    # Clear GPU cache for this thread
                    torch.cuda.empty_cache()
                    self.logger.debug(f"GPU context cleaned up for thread {thread_name}")
                except Exception as e:
                    self.logger.warning(f"Error cleaning up GPU context for thread {thread_name}: {e}")
                finally:
                    del self.thread_gpu_contexts[thread_name]
    
    def _worker_thread_loop(self):
        """Main loop for worker threads."""
        thread_name = threading.current_thread().name
        self.logger.debug(f"Worker thread {thread_name} started")
        
        # Setup GPU context for this thread
        self._setup_thread_gpu_context(thread_name)
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get next task with timeout
                    priority_item = self.task_queue.get(timeout=1.0)
                    priority, task_id, task = priority_item
                    
                    if task is None:  # Shutdown signal
                        break
                    
                    self._execute_task(task, thread_name)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error in worker thread {thread_name}: {e}", exc_info=True)
        finally:
            # Cleanup GPU context
            self._cleanup_thread_gpu_context(thread_name)
        
        self.logger.debug(f"Worker thread {thread_name} shutting down")
    
    def _execute_task(self, task: ProcessingTask, thread_name: str):
        """Execute a single task with error handling and progress tracking."""
        task_id = task.task_id
        
        try:
            with self.task_lock:
                task.status = TaskStatus.RUNNING
                task.started_time = time.time()
                self.active_tasks[task_id] = task
            
            self.logger.info(f"[{thread_name}] Starting task {task_id} ({task.task_type.value})")
            
            # Setup progress callback wrapper
            def progress_wrapper(progress: float, message: str = ""):
                self._report_progress(task_id, progress, message)
            
            # Add progress callback to kwargs if task supports it
            if task.progress_callback:
                task.kwargs['progress_callback'] = progress_wrapper
            
            # Execute the task function
            result = task.function(*task.args, **task.kwargs)
            
            # Task completed successfully
            with self.task_lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_time = time.time()
                task_duration = task.completed_time - task.started_time
                
                # Update statistics
                with self.stats_lock:
                    self.stats['tasks_completed'] += 1
                    self.stats['total_processing_time'] += task_duration
                    self.stats['average_task_time'] = (
                        self.stats['total_processing_time'] / self.stats['tasks_completed']
                    )
                
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            self.logger.info(f"[{thread_name}] Completed task {task_id} in {task_duration:.2f}s")
            
            # Call completion callback
            if task.completion_callback:
                try:
                    task.completion_callback(result)
                except Exception as callback_error:
                    self.logger.error(f"Error in completion callback for task {task_id}: {callback_error}")
            
        except Exception as e:
            # Task failed
            with self.task_lock:
                task.status = TaskStatus.FAILED
                task.error = e
                task.completed_time = time.time()
                
                with self.stats_lock:
                    self.stats['tasks_failed'] += 1
                
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            self.logger.error(f"[{thread_name}] Task {task_id} failed: {e}", exc_info=True)
            
            # Call error callback
            if task.error_callback:
                try:
                    task.error_callback(e)
                except Exception as callback_error:
                    self.logger.error(f"Error in error callback for task {task_id}: {callback_error}")
        
        finally:
            self.task_queue.task_done()
    
    def submit_task(
        self,
        task_id: str,
        task_type: TaskType,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        completion_callback: Optional[Callable[[Any], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> str:
        """
        Submit a task for processing.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of processing task
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            progress_callback: Optional progress reporting callback
            completion_callback: Optional completion callback
            error_callback: Optional error callback
            
        Returns:
            Task ID for tracking
        """
        if self.shutdown_event.is_set():
            raise RuntimeError("ProcessingThreadManager is shutting down")
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            function=function,
            args=args,
            kwargs=kwargs or {},
            progress_callback=progress_callback,
            completion_callback=completion_callback,
            error_callback=error_callback
        )
        
        # Add to queue with priority (lower number = higher priority)
        priority_value = 5 - priority.value  # Invert priority for queue
        self.task_queue.put((priority_value, task_id, task))
        
        with self.stats_lock:
            self.stats['tasks_submitted'] += 1
        
        self.logger.debug(f"Submitted task {task_id} ({task_type.value}) with priority {priority.name}")
        return task_id
    
    def _report_progress(self, task_id: str, progress: float, message: str = ""):
        """Report task progress to registered callbacks."""
        # Call task-specific progress callback
        if task_id in self.progress_callbacks:
            try:
                self.progress_callbacks[task_id](task_id, progress, message)
            except Exception as e:
                self.logger.error(f"Error in progress callback for task {task_id}: {e}")
        
        # Call global progress callback
        if self.global_progress_callback:
            try:
                self.global_progress_callback(task_id, progress, message)
            except Exception as e:
                self.logger.error(f"Error in global progress callback: {e}")
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        with self.task_lock:
            # Check if task is in active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status == TaskStatus.RUNNING:
                    # Task is already running, cannot cancel
                    self.logger.warning(f"Cannot cancel running task {task_id}")
                    return False
                
                task.status = TaskStatus.CANCELLED
                with self.stats_lock:
                    self.stats['tasks_cancelled'] += 1
                
                self.completed_tasks[task_id] = task
                del self.active_tasks[task_id]
                
                self.logger.info(f"Cancelled task {task_id}")
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        with self.task_lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].status
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id].status
        return None
    
    def get_task_result(self, task_id: str) -> Any:
        """Get the result of a completed task."""
        with self.task_lock:
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise task.error
        return None
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """
        Wait for a task to complete.
        
        Args:
            task_id: ID of task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if task completed, False if timeout
        """
        start_time = time.time()
        
        while True:
            status = self.get_task_status(task_id)
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        with self.task_lock:
            stats['active_tasks'] = len(self.active_tasks)
            stats['pending_tasks'] = self.task_queue.qsize()
            stats['completed_tasks_stored'] = len(self.completed_tasks)
        
        return stats
    
    def register_progress_callback(
        self, 
        task_id: str, 
        callback: Callable[[str, float, str], None]
    ):
        """Register a progress callback for a specific task."""
        self.progress_callbacks[task_id] = callback
    
    def set_global_progress_callback(self, callback: Callable[[str, float, str], None]):
        """Set a global progress callback for all tasks."""
        self.global_progress_callback = callback
    
    def clear_completed_tasks(self, max_age_seconds: float = 3600):
        """Clear old completed tasks to prevent memory buildup."""
        current_time = time.time()
        to_remove = []
        
        with self.task_lock:
            for task_id, task in self.completed_tasks.items():
                if task.completed_time and (current_time - task.completed_time) > max_age_seconds:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.completed_tasks[task_id]
        
        if to_remove:
            self.logger.debug(f"Cleared {len(to_remove)} old completed tasks")
    
    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown the thread manager and wait for all tasks to complete.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        self.logger.info("Shutting down ProcessingThreadManager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Send shutdown signals to worker threads
        for _ in self.worker_threads:
            self.task_queue.put((0, "shutdown", None))
        
        # Wait for threads to finish
        start_time = time.time()
        for thread in self.worker_threads:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                thread.join(remaining_time)
            
            if thread.is_alive():
                self.logger.warning(f"Worker thread {thread.name} did not shutdown cleanly")
        
        self.logger.info("ProcessingThreadManager shutdown complete")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if not self.shutdown_event.is_set():
                self.shutdown(timeout=2.0)
        except Exception:
            pass  # Avoid errors during cleanup