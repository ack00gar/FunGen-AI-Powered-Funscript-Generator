import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional

@dataclass
class Task:
    id: int = field(init=False)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    profile: Dict[str, float] = field(default_factory=dict)  # Timing/profiling info
    _id_counter: int = 0
    _id_lock: Lock = Lock()
    _lock: Lock = field(default_factory=Lock, repr=False, init=False)

    def __post_init__(self):
        # Use the class-level lock for thread safety
        with self.__class__._id_lock:
            self.__class__._id_counter += 1
            self.id = self.__class__._id_counter

    def start(self, process_type: str):
        self._update_profile(process_type, "start")

    def end(self, process_type: str):
        self._update_profile(process_type, "end")
        self._calculate_duration(process_type)

    def duration(self, process_type: str, duration: int):
        key = f"{process_type}_duration"
        with self._lock:
            self.profile[key] = duration

    def _update_profile(self, process_type: str, action: str):
        key = f"{process_type}_{action}"
        with self._lock:
            self.profile[key] = time.time()

    def _calculate_duration(self, process_type: str):
        start_key = f"{process_type}_start"
        end_key = f"{process_type}_end"
        duration_key = f"{process_type}_duration"

        with self._lock:
            if start_key in self.profile and end_key in self.profile:
                self.profile[duration_key] = self.profile[end_key] - self.profile[start_key]



