import threading
from typing import Dict, Any

class JobCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Any:
        with self._lock:
            return self._cache.get(job_id)

    def set(self, job_id: str, value: Any):
        with self._lock:
            self._cache[job_id] = value

    def update(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self._cache:
                self._cache[job_id].update(kwargs)

JOB_CACHE = JobCache() 