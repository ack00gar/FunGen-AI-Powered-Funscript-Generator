import psutil
import threading
import time
import platform
import subprocess
import json
from collections import deque

# Try to import pynvml for GPU monitoring
try:
    from pynvml.smi import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except (ImportError, Exception):
    NVIDIA_SMI_AVAILABLE = False

class SystemMonitor:
    def __init__(self, update_interval=1, history_size=100):
        self.update_interval = update_interval
        self.history_size = history_size
        self.is_running = False
        self.thread = None
        self.os_type = platform.system()

        # Data storage
        self.cpu_load = deque([0] * history_size, maxlen=history_size)
        self.cpu_core_count = psutil.cpu_count(logical=True)
        self.cpu_physical_cores = psutil.cpu_count(logical=False)
        self.cpu_per_core = deque([[] * history_size], maxlen=history_size)  # Per-core usage
        self.cpu_freq = 0  # CPU frequency
        self.cpu_temp = 0  # CPU temperature (if available)
        
        self.ram_usage_percent = deque([0] * history_size, maxlen=history_size)
        self.ram_usage_gb = deque([0] * history_size, maxlen=history_size)
        self.ram_total_gb = psutil.virtual_memory().total / (1024**3)
        self.swap_usage_percent = 0  # Swap usage
        self.swap_usage_gb = 0
        
        self.gpu_load = deque([0] * history_size, maxlen=history_size)
        self.gpu_mem_usage_percent = deque([0] * history_size, maxlen=history_size)
        self.gpu_info = None
        self.gpu_name = "Unknown GPU"
        self.gpu_temp = 0  # GPU temperature (if available)
        
        # Note: removed memory_pressure, disk_usage, uptime - not displayed in UI

        # GPU setup
        self.nvsmi = None
        self.gpu_available = False
        self._setup_gpu_monitoring()

    def _setup_gpu_monitoring(self):
        """Set up GPU monitoring for various GPU types."""
        global NVIDIA_SMI_AVAILABLE
        
        # Try NVIDIA first
        if NVIDIA_SMI_AVAILABLE:
            try:
                self.nvsmi = nvidia_smi.getInstance()
                self.gpu_info = self.nvsmi.DeviceQuery('name,utilization.gpu,memory.total,memory.used')
                if self.gpu_info and 'gpu' in self.gpu_info:
                    self.gpu_name = self.gpu_info['gpu'][0].get('product_name', 'NVIDIA GPU')
                    self.gpu_available = True
            except Exception:
                self.nvsmi = None
                NVIDIA_SMI_AVAILABLE = False
        
        # Try Apple Silicon GPU detection (macOS)
        if not self.gpu_available and self.os_type == 'Darwin':
            try:
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType', '-json'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    displays = data.get('SPDisplaysDataType', [])
                    for display in displays:
                        gpu_model = display.get('sppci_model', '')
                        if 'Apple' in gpu_model or 'M1' in gpu_model or 'M2' in gpu_model or 'M3' in gpu_model or 'M4' in gpu_model:
                            self.gpu_name = gpu_model
                            self.gpu_available = True
                            break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
                pass
            
            # If still not detected, try alternative detection for Apple Silicon
            if not self.gpu_available:
                try:
                    # Check for Apple GPU through ioreg
                    result = subprocess.run(['ioreg', '-l', '-w', '0'], 
                                          capture_output=True, text=True, timeout=3)
                    if 'AppleM' in result.stdout or 'Apple GPU' in result.stdout:
                        # Try to extract the specific GPU model
                        import re
                        match = re.search(r'"model" = <"([^"]*Apple[^"]*M[0-9][^"]*GPU[^"]*)">', result.stdout)
                        if match:
                            self.gpu_name = match.group(1)
                        else:
                            self.gpu_name = "Apple Silicon GPU"
                        self.gpu_available = True
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    pass
        
        if not self.gpu_available:
            self.gpu_name = "No GPU detected"

    def _update_stats(self):
        """Internal method to update all system stats."""
        # CPU
        cpu_overall = psutil.cpu_percent()
        cpu_per_core = psutil.cpu_percent(percpu=True)
        self.cpu_load.append(cpu_overall)
        self.cpu_per_core.append(cpu_per_core)

        # RAM
        mem = psutil.virtual_memory()
        self.ram_usage_percent.append(mem.percent)
        self.ram_usage_gb.append(mem.used / (1024**3))
        
        # Swap
        try:
            swap = psutil.swap_memory()
            self.swap_usage_percent = swap.percent
            self.swap_usage_gb = swap.used / (1024**3)
        except (RuntimeError, OSError, AttributeError):
            # Windows performance counters may be disabled
            self.swap_usage_percent = 0
            self.swap_usage_gb = 0
        
        # CPU Frequency (safe check)
        try:
            cpu_freq_info = psutil.cpu_freq()
            if cpu_freq_info:
                self.cpu_freq = cpu_freq_info.current
        except (AttributeError, OSError):
            self.cpu_freq = 0
        
        # Removed disk usage and uptime collection - not displayed in UI

        # GPU Monitoring
        gpu_usage = 0
        gpu_mem_usage = 0
        
        if self.nvsmi:  # NVIDIA GPU
            try:
                gpu_query = self.nvsmi.DeviceQuery('utilization.gpu,memory.used,memory.total')
                if gpu_query and 'gpu' in gpu_query:
                    gpu = gpu_query['gpu'][0]
                    gpu_usage = gpu['utilization']['gpu_util']
                    gpu_mem_usage = (gpu['fb_memory_usage']['used'] / gpu['fb_memory_usage']['total']) * 100
            except Exception:
                pass
        elif self.gpu_available and self.os_type == 'Darwin':  # Apple Silicon GPU
            # For Apple Silicon, we detect the GPU but cannot get reliable metrics without admin access
            # Do not provide estimates - either we have real data or we don't
            
            # Try powermetrics for real GPU data (requires admin, likely to fail)
            try:
                # Attempt to use powermetrics for real GPU utilization (requires sudo)
                result = subprocess.run(['powermetrics', '--samplers', 'gpu_power', '-n', '1', '--show-process-gpu'], 
                                      capture_output=True, text=True, timeout=3)
                if result.returncode == 0 and 'GPU' in result.stdout:
                    # Parse actual GPU usage from powermetrics output
                    import re
                    gpu_match = re.search(r'GPU\s+usage:\s*([0-9.]+)%', result.stdout)
                    if gpu_match:
                        gpu_usage = float(gpu_match.group(1))
                    
                    # Parse GPU memory usage if available
                    mem_match = re.search(r'GPU\s+memory:\s*([0-9.]+)%', result.stdout)
                    if mem_match:
                        gpu_mem_usage = float(mem_match.group(1))
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, Exception):
                # powermetrics failed (expected without admin access)
                # Leave gpu_usage and gpu_mem_usage at 0 to indicate no metrics available
                pass
        
        self.gpu_load.append(gpu_usage)
        self.gpu_mem_usage_percent.append(gpu_mem_usage)

        # Removed memory pressure collection - not working properly and not displayed


    def _run(self):
        """The loop that runs in the background thread."""
        while self.is_running:
            self._update_stats()
            time.sleep(self.update_interval)  # This will use the current value

    def start(self):
        """Starts the monitoring thread."""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def get_stats(self):
        """Returns the latest stats."""
        return {
            "cpu_load": list(self.cpu_load),
            "cpu_core_count": self.cpu_core_count,
            "cpu_physical_cores": self.cpu_physical_cores,
            "cpu_per_core": list(self.cpu_per_core)[-1] if self.cpu_per_core else [],  # Latest per-core data
            "cpu_freq": self.cpu_freq,
            "cpu_temp": self.cpu_temp,
            "ram_usage_percent": list(self.ram_usage_percent),
            "ram_usage_gb": list(self.ram_usage_gb),
            "ram_total_gb": self.ram_total_gb,
            "swap_usage_percent": self.swap_usage_percent,
            "swap_usage_gb": self.swap_usage_gb,
            "gpu_load": list(self.gpu_load),
            "gpu_mem_usage_percent": list(self.gpu_mem_usage_percent),
            "gpu_info": self.gpu_info,
            "gpu_name": self.gpu_name,
            "gpu_available": self.gpu_available,
            "gpu_temp": self.gpu_temp,
            # Removed: memory_pressure, disk_usage, uptime
            "os": self.os_type,
        }
