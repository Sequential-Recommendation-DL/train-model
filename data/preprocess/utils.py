import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{name}] {elapsed:.2f}s")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
