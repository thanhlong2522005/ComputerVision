import time


class PerformanceTracker:
    def __init__(self):
        self._start = time.time()
        self._frames = 0

    def update(self) -> float:
        self._frames += 1
        return self._frames / (time.time() - self._start)
