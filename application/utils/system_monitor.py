import json
import platform
import re
import subprocess
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import psutil

# Try to import pynvml for GPU monitoring
try:
    from pynvml.smi import nvidia_smi

    NVIDIA_SMI_AVAILABLE = True
except Exception:
    NVIDIA_SMI_AVAILABLE = False


def _bytes_to_gb(value_bytes: float) -> float:
    return value_bytes / (1024**3)


def _bytes_to_mb(value_bytes: float) -> float:
    return value_bytes / (1024**2)





class SystemMonitor:
    """
    Cross-platform system monitor collecting CPU, RAM, swap, disk I/O, and GPU stats.
    Runs an internal background thread at a given interval and keeps history.
    """

    def __init__(
        self,
        update_interval: float = 1.0,
        history_size: int = 100,
        per_disk_io: bool = True,
    ):
        self.update_interval: float = max(0.1, float(update_interval))
        self.history_size: int = max(1, int(history_size))
        self.per_disk_io: bool = per_disk_io

        self._is_running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self.os_type: str = platform.system()

        # CPU metadata
        self.cpu_core_count: int = psutil.cpu_count(logical=True) or 0
        self.cpu_physical_cores: int = psutil.cpu_count(logical=False) or 0

        # Rolling histories
        self.cpu_load: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        initial_per_core = [0.0] * max(1, self.cpu_core_count)
        self.cpu_per_core: Deque[List[float]] = deque(
            [initial_per_core.copy()] * self.history_size, maxlen=self.history_size
        )

        self.cpu_freq: float = 0.0
        self.cpu_temp: Optional[float] = None  # Celsius

        self.ram_usage_percent: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        self.ram_usage_gb: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        self.ram_total_gb: float = _bytes_to_gb(psutil.virtual_memory().total)

        self.swap_usage_percent: float = 0.0
        self.swap_usage_gb: float = 0.0

        # Disk I/O histories (aggregate)
        self.disk_read_bytes_delta: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        self.disk_write_bytes_delta: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        self.disk_read_mb_s: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        self.disk_write_mb_s: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        # Cumulative totals
        self.disk_cumulative_read_bytes: float = 0.0
        self.disk_cumulative_write_bytes: float = 0.0

        # Per-disk latest snapshot (display name -> metrics)
        self.per_disk_latest: Dict[str, Dict[str, float]] = {}

        # Previous counters for delta computation
        self._prev_disk_counters: Optional[psutil._common.sdiskio] = None
        self._prev_per_disk_counters: Dict[str, psutil._common.sdiskio] = {}

        # GPU fields
        self.gpu_load: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        self.gpu_mem_usage_percent: Deque[float] = deque(
            [0.0] * self.history_size, maxlen=self.history_size
        )
        self.gpu_temp: Optional[float] = None
        self.gpu_info: Optional[Dict[str, Any]] = None
        self.gpu_name: str = "Unknown GPU"
        self.gpu_available: bool = False

        # GPU setup
        self._nvsmi = None
        self._setup_gpu_monitoring()

        # Prime cpu_percent
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(percpu=True, interval=None)
        except Exception:
            pass

        # Prime disk counters
        try:
            self._prev_disk_counters = psutil.disk_io_counters(nowrap=True)
            if self.per_disk_io:
                self._prev_per_disk_counters = psutil.disk_io_counters(
                    perdisk=True, nowrap=True
                )
        except Exception:
            self._prev_disk_counters = None
            self._prev_per_disk_counters = {}

    def _setup_gpu_monitoring(self) -> None:
        if NVIDIA_SMI_AVAILABLE:
            try:
                self._nvsmi = nvidia_smi.getInstance()
                info = self._nvsmi.DeviceQuery(
                    "name,utilization.gpu,memory.total,memory.used,temperature.gpu"
                )
                self.gpu_info = info
                if info and isinstance(info, dict) and "gpu" in info and info["gpu"]:
                    gpu0 = info["gpu"][0]
                    self.gpu_name = (
                        gpu0.get("product_name") or gpu0.get("name") or "NVIDIA GPU"
                    )
                    self.gpu_available = True
                    return
            except Exception:
                self._nvsmi = None

        if self.os_type == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0 and result.stdout:
                    data = json.loads(result.stdout)
                    displays = data.get("SPDisplaysDataType", [])
                    for display in displays:
                        model = display.get("sppci_model", "") or display.get(
                            "spdisplays_vendor", ""
                        )
                        if any(x in model for x in ["Apple", "M1", "M2", "M3", "M4"]):
                            self.gpu_name = model or "Apple Silicon GPU"
                            self.gpu_available = True
                            return
            except Exception:
                pass

            try:
                result = subprocess.run(
                    ["ioreg", "-l", "-w", "0"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                out = result.stdout or ""
                if ("AppleM" in out) or ("Apple GPU" in out):
                    match = re.search(
                        r'"model"\s*=\s*<"([^"]*Apple[^"]*M[0-9][^"]*GPU[^"]*)">', out
                    )
                    if match:
                        self.gpu_name = match.group(1)
                    else:
                        self.gpu_name = "Apple Silicon GPU"
                    self.gpu_available = True
                    return
            except Exception:
                pass

        self.gpu_name = "No GPU detected"
        self.gpu_available = False

    def _read_cpu_frequency(self) -> None:
        try:
            freq = psutil.cpu_freq()
            if freq:
                self.cpu_freq = float(freq.current)
        except Exception:
            self.cpu_freq = 0.0

    def _read_cpu_temperature(self) -> None:
        try:
            temps = psutil.sensors_temperatures()
        except Exception:
            self.cpu_temp = None
            return
        if not temps:
            self.cpu_temp = None
            return
        samples: List[float] = []
        for _, entries in temps.items():
            for entry in entries:
                if entry.current is not None:
                    samples.append(float(entry.current))
        self.cpu_temp = float(sum(samples) / len(samples)) if samples else None

    def _read_gpu_stats_nvidia(self) -> Tuple[float, float, Optional[float]]:
        """
        Returns (gpu_usage_percent, gpu_mem_usage_percent, gpu_temp_celsius) for NVIDIA GPUs.
        """
        usage_percent = 0.0
        mem_usage_percent = 0.0
        temp_celsius: Optional[float] = None

        if not self._nvsmi:
            return usage_percent, mem_usage_percent, temp_celsius

        try:
            query = self._nvsmi.DeviceQuery(
                "utilization.gpu,temperature.gpu,memory.used,memory.total"
            )
            if query and "gpu" in query and query["gpu"]:
                gpu0 = query["gpu"][0]

                # Utilization
                util = gpu0.get("utilization", {})
                raw_util = util.get("gpu_util") or util.get("gpu")
                if isinstance(raw_util, (int, float)):
                    usage_percent = float(raw_util)
                elif isinstance(raw_util, str):
                    try:
                        usage_percent = float(raw_util.strip().strip("%"))
                    except Exception:
                        pass

                # Memory percent from used/total
                fb = gpu0.get("fb_memory_usage") or {}
                used = fb.get("used")
                total = fb.get("total")

                def _to_mib(x):
                    if x is None:
                        return None
                    if isinstance(x, (int, float)):
                        return float(x)
                    m = re.search(r"([0-9.]+)", str(x))
                    return float(m.group(1)) if m else None

                used_mib = _to_mib(used)
                total_mib = _to_mib(total)
                if used_mib and total_mib and total_mib > 0:
                    mem_usage_percent = (used_mib / total_mib) * 100.0

                # Temperature
                temp_block = gpu0.get("temperature") or {}
                raw_temp = temp_block.get("gpu_temp") or temp_block.get("gpu")
                if isinstance(raw_temp, (int, float)):
                    temp_celsius = float(raw_temp)
                elif isinstance(raw_temp, str):
                    try:
                        temp_celsius = float(raw_temp.strip().strip("C").strip("°"))
                    except Exception:
                        temp_celsius = None
        except Exception:
            pass

        return usage_percent, mem_usage_percent, temp_celsius

    def _read_gpu_stats_apple(self) -> Tuple[float, float, Optional[float]]:
        """
        Returns (gpu_usage_percent, gpu_mem_usage_percent, gpu_temp_celsius) on Apple Silicon.
        Uses powermetrics if available (often requires sudo). Best-effort parsing.
        """
        usage_percent = 0.0
        mem_usage_percent = 0.0
        temp_celsius: Optional[float] = None

        try:
            result = subprocess.run(
                ["powermetrics", "--samplers", "gpu_power", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                out = result.stdout
                m_usage = re.search(r"GPU\s+usage:\s*([0-9.]+)\s*%", out, re.I)
                if m_usage:
                    usage_percent = float(m_usage.group(1))
                m_mem = re.search(r"GPU\s+memory:\s*([0-9.]+)\s*%", out, re.I)
                if m_mem:
                    mem_usage_percent = float(m_mem.group(1))
                m_temp = re.search(r"GPU\s+die\s+temperature:\s*([0-9.]+)", out, re.I)
                if m_temp:
                    temp_celsius = float(m_temp.group(1))
        except FileNotFoundError:
            # powermetrics not available
            pass
        except Exception:
            # Any other error; leave defaults
            pass

        return usage_percent, mem_usage_percent, temp_celsius

    def _update_disk_io(self, elapsed_seconds: float) -> None:
        """
        Update aggregate and per-disk I/O stats.
        elapsed_seconds: seconds since last sample (for throughput).
        """
        try:
            aggregate_counters = psutil.disk_io_counters(nowrap=True)
        except Exception:
            aggregate_counters = None

        read_delta_bytes = 0.0
        write_delta_bytes = 0.0

        if aggregate_counters and self._prev_disk_counters:
            current_read = float(aggregate_counters.read_bytes)
            current_write = float(aggregate_counters.write_bytes)
            prev_read = float(self._prev_disk_counters.read_bytes)
            prev_write = float(self._prev_disk_counters.write_bytes)

            read_delta_bytes = current_read - prev_read
            write_delta_bytes = current_write - prev_write

            if read_delta_bytes < 0:
                read_delta_bytes = max(0.0, current_read)
            if write_delta_bytes < 0:
                write_delta_bytes = max(0.0, current_write)

            self.disk_cumulative_read_bytes = current_read
            self.disk_cumulative_write_bytes = current_write
        elif aggregate_counters:
            self.disk_cumulative_read_bytes = float(aggregate_counters.read_bytes)
            self.disk_cumulative_write_bytes = float(aggregate_counters.write_bytes)

        self.disk_read_bytes_delta.append(read_delta_bytes)
        self.disk_write_bytes_delta.append(write_delta_bytes)

        if elapsed_seconds > 0:
            self.disk_read_mb_s.append(_bytes_to_mb(read_delta_bytes) / elapsed_seconds)
            self.disk_write_mb_s.append(
                _bytes_to_mb(write_delta_bytes) / elapsed_seconds
            )
        else:
            self.disk_read_mb_s.append(0.0)
            self.disk_write_mb_s.append(0.0)

        if aggregate_counters:
            self._prev_disk_counters = aggregate_counters

        # Per-disk details - aggregate all drives
        if self.per_disk_io:
            try:
                per_disk_counters = psutil.disk_io_counters(perdisk=True, nowrap=True)
            except Exception:
                per_disk_counters = {}

            # Aggregate all drives into a single entry
            total_read_delta = 0.0
            total_write_delta = 0.0
            total_read_bytes = 0.0
            total_write_bytes = 0.0

            for dev_key, current in per_disk_counters.items():
                prev = self._prev_per_disk_counters.get(dev_key)
                
                if prev:
                    read_delta = float(current.read_bytes) - float(prev.read_bytes)
                    write_delta = float(current.write_bytes) - float(prev.write_bytes)
                    if read_delta < 0:
                        read_delta = max(0.0, float(current.read_bytes))
                    if write_delta < 0:
                        write_delta = max(0.0, float(current.write_bytes))
                else:
                    read_delta = 0.0
                    write_delta = 0.0

                total_read_delta += read_delta
                total_write_delta += write_delta
                total_read_bytes += float(current.read_bytes)
                total_write_bytes += float(current.write_bytes)

            # Store as single aggregate entry
            self.per_disk_latest = {
                "All Drives": {
                    "read_mb_s": (_bytes_to_mb(total_read_delta) / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
                    "write_mb_s": (_bytes_to_mb(total_write_delta) / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
                    "read_bytes_total": total_read_bytes,
                    "write_bytes_total": total_write_bytes,
                }
            }
            self._prev_per_disk_counters = per_disk_counters

    def _update_stats(self) -> None:
        t0 = time.monotonic()
        with self._lock:
            # CPU
            try:
                cpu_overall = psutil.cpu_percent(interval=None)
                per_core = psutil.cpu_percent(percpu=True, interval=None)
            except Exception:
                cpu_overall = 0.0
                per_core = [0.0] * max(1, self.cpu_core_count)
            if len(per_core) != self.cpu_core_count:
                self.cpu_core_count = len(per_core)
            self.cpu_load.append(float(cpu_overall))
            self.cpu_per_core.append([float(x) for x in per_core])

            # RAM
            try:
                mem = psutil.virtual_memory()
                self.ram_usage_percent.append(float(mem.percent))
                self.ram_usage_gb.append(_bytes_to_gb(float(mem.used)))
                self.ram_total_gb = _bytes_to_gb(float(mem.total))
            except Exception:
                self.ram_usage_percent.append(0.0)
                self.ram_usage_gb.append(0.0)

            # Swap
            try:
                swap = psutil.swap_memory()
                self.swap_usage_percent = float(swap.percent)
                self.swap_usage_gb = _bytes_to_gb(float(swap.used))
            except Exception:
                self.swap_usage_percent = 0.0
                self.swap_usage_gb = 0.0

            # CPU freq/temp
            self._read_cpu_frequency()
            self._read_cpu_temperature()

            # GPU
            gpu_usage = 0.0
            gpu_mem_usage = 0.0
            gpu_temp = None
            if self._nvsmi:
                u, m, t = self._read_gpu_stats_nvidia()
                gpu_usage, gpu_mem_usage, gpu_temp = u, m, t
            elif self.gpu_available and self.os_type == "Darwin":
                u, m, t = self._read_gpu_stats_apple()
                gpu_usage, gpu_mem_usage, gpu_temp = u, m, t
            self.gpu_load.append(float(gpu_usage))
            self.gpu_mem_usage_percent.append(float(gpu_mem_usage))
            self.gpu_temp = gpu_temp

        elapsed = max(0.0, time.monotonic() - t0)
        with self._lock:
            self._update_disk_io(elapsed)

    def _run(self) -> None:
        next_tick = time.monotonic()
        while self._is_running:
            self._update_stats()
            next_tick += self.update_interval
            sleep_time = max(0.0, next_tick - time.monotonic())
            if sleep_time > 5 * self.update_interval:
                next_tick = time.monotonic() + self.update_interval
                sleep_time = self.update_interval
            time.sleep(sleep_time)
            if self.update_interval <= 0:
                self.update_interval = 1.0

    def start(self) -> None:
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        with self._lock:
            self._is_running = False
            thread = self._thread
            self._thread = None
        if thread:
            thread.join(timeout=timeout)

    def set_update_interval(self, interval: float) -> None:
        with self._lock:
            self.update_interval = max(0.1, float(interval))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cpu_load": list(self.cpu_load),
                "cpu_core_count": self.cpu_core_count,
                "cpu_physical_cores": self.cpu_physical_cores,
                "cpu_per_core": list(self.cpu_per_core)[-1] if self.cpu_per_core else [],
                "cpu_freq": self.cpu_freq,
                "cpu_temp": self.cpu_temp,
                "ram_usage_percent": list(self.ram_usage_percent),
                "ram_usage_gb": list(self.ram_usage_gb),
                "ram_total_gb": self.ram_total_gb,
                "swap_usage_percent": self.swap_usage_percent,
                "swap_usage_gb": self.swap_usage_gb,
                # Disk I/O (aggregate)
                "disk_read_bytes_delta": list(self.disk_read_bytes_delta),
                "disk_write_bytes_delta": list(self.disk_write_bytes_delta),
                "disk_read_mb_s": list(self.disk_read_mb_s),
                "disk_write_mb_s": list(self.disk_write_mb_s),
                "disk_cumulative_read_bytes": self.disk_cumulative_read_bytes,
                "disk_cumulative_write_bytes": self.disk_cumulative_write_bytes,
                # Per-disk snapshot
                "per_disk_latest": self.per_disk_latest if self.per_disk_io else {},
                # GPU
                "gpu_load": list(self.gpu_load),
                "gpu_mem_usage_percent": list(self.gpu_mem_usage_percent),
                "gpu_info": self.gpu_info,
                "gpu_name": self.gpu_name,
                "gpu_available": self.gpu_available,
                "gpu_temp": self.gpu_temp,
                "os": self.os_type,
            }