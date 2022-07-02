import time


class FitTimer:
    """A simple timer to measure time taken fitting a model."""

    def __init__(self, init_offset: float = 0.0):
        self._start_time = None
        self._elapsed: float = 0
        self._on: bool = False
        self.reset(init_offset)

    def reset(self, init_offset: float = 0.0):
        self._on = False
        self._elapsed = 0.0 + init_offset
        return self

    def stop(self):
        if self._start_time and self._on:
            self._on = False
            self._elapsed += time.time() - self._start_time
        return self

    def start(self):
        if not self._on:
            self._on = True
            self._start_time = time.time()
        return self

    def get_elapsed(self) -> float:
        if self._on:
            timestamp = time.time()
            return timestamp - self._start_time
        return self._elapsed

    def is_on(self) -> bool:
        return self._on
