import time
from contextlib import contextmanager
from typing import Dict, Generator


class TimingResult:
    """Stores elapsed time for named stages."""

    def __init__(self) -> None:
        self._times: Dict[str, float] = {}

    def record(self, stage: str, elapsed_ms: float) -> None:
        self._times[stage] = elapsed_ms

    def get(self, stage: str) -> float:
        return self._times.get(stage, 0.0)

    def total(self) -> float:
        return sum(self._times.values())

    def as_dict(self) -> Dict[str, float]:
        return dict(self._times)


@contextmanager
def timer(result: TimingResult, stage: str) -> Generator[None, None, None]:
    """Context manager that records wall-clock milliseconds for a named stage."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result.record(stage, elapsed_ms)


def measure_ms(func, *args, **kwargs):
    """Run *func* once and return (return_value, elapsed_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms
