import sys
import os
import time

# Add src to Python path
sys.path.insert(0, os.path.abspath('src'))

from chickgut.simulation import GIT_simulation
import pandas as pd

def main():
    print("Running Baseline SciPy Simulation...")
    start_time = time.time()
    
    # Run the simulation for 'Soybean meal' (index 0)
    # GIT_simulation returns res_df, GIT_df, last_t
    res_df, GIT_df, last_t = GIT_simulation(0)
    
    end_time = time.time()
    
    print(f"\n--- BASELINE TIMING (SciPy) ---")
    print(f"Total Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Last t: {last_t}")
    print(f"Results shape: {res_df.shape}")
    
if __name__ == "__main__":
    main()
