import timeit


class FitTimer:
    """A simple timer to measure time taken fitting a model."""

    def __init__(self, init_offset: float = 0.0):
        self._elapsed = 0.0 + init_offset
        self._on = False

    def reset(self, init_offset: float = 0.0):
        self._on = False
        self._elapsed = 0.0 + init_offset
        return self

    def stop(self):
        if self.start_time and self._on:
            self._on = False
            self._elapsed += timeit.default_timer() - self.start_time
        return self

    def start(self):
        if not self._on:
            self._on = True
            self.start_time = timeit.default_timer()
        return self

    def get_elapsed(self) -> float:
        if self._on:
            round = timeit.default_timer()
            return round - self.start_time
        return self._elapsed

    def is_on(self) -> bool:
        return self._on
