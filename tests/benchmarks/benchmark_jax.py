import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.append('src')

from chickgut.simulation import run_simulation, k_absp, k_digestrate, Kp_endog_min
from chickgut.io import get_ing_const
from chickgut.main import t_eval, t_span

def benchmark_jax():
    print("Initializing Benchmark...")
    ingr_name = "sbm"
    constants = get_ing_const(ingr_name)
    
    # Warmup Run (includes JIT compilation)
    print("\n--- Warmup Run (Compiling JAX graphs) ---")
    start = time.perf_counter()
    run_simulation(ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span)
    warmup_time = time.perf_counter() - start
    print(f"Warmup Time: {warmup_time:.4f} seconds")
    
    # Benchmark Run (Compiled)
    print("\n--- Benchmark Run (Pure Execution) ---")
    start = time.perf_counter()
    run_simulation(ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span)
    bench_time = time.perf_counter() - start
    print(f"Benchmark Time: {bench_time:.4f} seconds")
    
    # Baseline comparison 
    # (The original SciPy version was reported taking ~20-30 minutes, or roughly 1200+ seconds)
    speedup = 1200 / bench_time if bench_time > 0 else 0
    print(f"\nEstimated Speedup vs SciPy (assuming 20 min SciPy): {speedup:.1f}x")

if __name__ == "__main__":
    benchmark_jax()
