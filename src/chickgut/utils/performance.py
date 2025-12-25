import time
from functools import wraps
from contextlib import contextmanager

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {func.__name__} took {end - start:.6f}s")
        return result
    return wrapper

@contextmanager
def time_block(label="Block"):
    start = time.perf_counter()
    try:
        yield  # This is where your code runs
    finally:
        end = time.perf_counter()
        print(f"{label} took {end - start:.6f} seconds")