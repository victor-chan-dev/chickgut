import os
import pickle
import time
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.pool import ThreadPool

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from chickgut.simulation import run_simulation

class ChickgutProblem(ElementwiseProblem):
    """
    This class defines our specific optimization problem for pymoo.
    It tells the optimizer how many parameters we have, what their limits are,
    and how to score a specific set of guessed parameters.
    """
    def __init__(self, t_eval, t_span, ingr_name, constants, **kwargs):
        # We define 3 parameters to optimize (n_var=3): k_absp, k_digestrate, Kp_endog_min.
        # We are trying to minimize 1 objective (n_obj=1), which is the Sum of Squared Errors.
        # xl is the lower limit (0 for all), and xu is the upper limit for our parameters.
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=0, xl=np.array([0.0, 0.0, 0.0]), xu=np.array([100.0, 0.1, 0.1]), **kwargs)
        self.t_eval = t_eval
        self.t_span = t_span
        self.ingr_name = ingr_name
        self.constants = constants
        self.protein_data = constants['target_v_sdis']
        
    def _evaluate(self, x, out, *args, **kwargs):
        # This function scores how "good" the optimizer's guess is.
        # x contains the current guessed values for the 3 parameters.
        k_absp, k_digestrate, Kp_endog_min = x
        
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] [Ingredient: {self.ingr_name.upper()}] Starting Evaluation: k_absp={k_absp:.4f}, k_dig={k_digestrate:.4f}, Kp_endog={Kp_endog_min:.4f}")
        
        try:
            # 1. Run the digestive simulation with these guessed parameters.
            # export_dataframes=False skips the heavy spreadsheet generation, making the loop 4x-5x faster!
            duodenum_instance, jejunum_instance, ileum_instance, df_fore = run_simulation(
                self.ingr_name, self.constants, k_absp, k_digestrate, Kp_endog_min, self.t_eval, self.t_span, export_dataframes=False
            )
            
            # 2. Collect the total amount of unabsorbed protein flowing out of the ileum
            sum_UI_flux, sum_SlI_flux, sum_RI_flux = ileum_instance.get_flux_sums(limit=2001)
            sum_flux_total = sum_RI_flux + sum_UI_flux + sum_SlI_flux 
            
            # 3. Calculate the predicted Apparent Digestibility (%)
            # This is the percentage of the original feed protein that successfully absorbed
            pred_protein = (1 - (sum_flux_total / self.constants['FI_24h_gbird'])) * 100
            
            # 4. Calculate the error between our prediction and the real-world lab data
            # The optimizer will try to make this number (SSE_protein) as close to 0 as possible.
            SSE_protein = np.sum((pred_protein - self.protein_data) ** 2)
            out["F"] = SSE_protein
            
        except Exception as e:
            # If the ODE solver crashes (e.g., the guessed parameters caused extreme stiffness),
            # we return a massive penalty error (1e10). This teaches the optimizer to avoid these parameters.
            out["F"] = 1e10

class CheckpointCallback(Callback):
    """
    This callback runs at the end of every generation (population evaluation).
    It saves the current state so that if your computer turns off, you don't lose days of progress.
    """
    def __init__(self, checkpoint_path, protein_data, ingr_name, absolute_gen_offset, total_gen):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.protein_data = protein_data
        self.ingr_name = ingr_name
        self.absolute_gen_offset = absolute_gen_offset
        self.total_gen = total_gen
        self.last_gen_time = time.time()
        
    def notify(self, algorithm):
        abs_gen = self.absolute_gen_offset + algorithm.n_gen
        
        # We save only the minimal essential data: the current absolute generation number,
        # the population of parameters, and the best scores.
        state = {
            'n_gen': abs_gen,
            'pop': algorithm.pop,
            'opt': algorithm.opt
        }
        
        # Save this state to disk in a binary format (pickle)
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
            
        # Logging progress to the terminal for the user
        import datetime
        import time
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        current_time = time.time()
        time_per_gen = current_time - self.last_gen_time
        
        # PyMoo fires notify() immediately on startup/resume before doing any math. 
        # If it took less than 1 second, it's a false reading and we shouldn't update the ETA.
        if time_per_gen < 1.0:
            eta_str = "Calculating..."
        else:
            self.last_gen_time = current_time
            gens_remaining = max(0, self.total_gen - abs_gen)
            eta_seconds = time_per_gen * gens_remaining
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        best_sse = float(algorithm.opt.get("F")[0])
        best_x = algorithm.opt.get("X")[0]
        print(f"[{now}] [Ingredient: {self.ingr_name.upper()}] Checkpoint saved successfully!")
        print(f"[{now}] [Ingredient: {self.ingr_name.upper()}] Generation {abs_gen}/{self.total_gen} complete | ETA: {eta_str} | Best SSE: {best_sse:.4f} | Best X: k_absp={best_x[0]:.4f}, k_dig={best_x[1]:.4f}, Kp_endog={best_x[2]:.4f}\n")

def optimize_params(t_eval, t_span, ingr_name, constants, output_dir=".", n_threads=None, pop_size=40, n_gen=150):
    """
    Fits the model parameters (k_absp, k_digestrate, Kp_endog_min) using pymoo DE.
    Saves optimization results and checkpoints to the specified output directory.
    """
    if n_threads is None:
        # Automatically use all available CPU cores on your computer
        n_threads = os.cpu_count() or 1
        
    print(f"Starting optimization for {ingr_name} using pymoo with {n_threads} threads...")
    
    checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")
    
    # We use a ThreadPool instead of a ProcessPool because JAX is highly optimized.
    # Normally, Python threads block each other (GIL), but JAX skips this lock, 
    # letting us run parallel ODEs blazing fast without eating up all your RAM.
    pool = ThreadPool(n_threads)
    
    problem = ChickgutProblem(
        t_eval=t_eval, 
        t_span=t_span, 
        ingr_name=ingr_name, 
        constants=constants
    )
    
    # Set up the Differential Evolution algorithm
    algorithm = DE(pop_size=pop_size)
    absolute_gen_offset = 0
    target_gen = n_gen
    
    # Checkpoint recovery: Did we already run this partially and crash?
    if os.path.exists(checkpoint_path):
        print(f"Resuming optimization from existing checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
            # Pick up exactly where we left off by loading the saved population
            algorithm = DE(pop_size=pop_size, sampling=state['pop'])
            absolute_gen_offset = state['n_gen']
            # Don't run the full n_gen, only run the generations we have left
            target_gen = max(1, n_gen - absolute_gen_offset)
        
    callback = CheckpointCallback(checkpoint_path, constants['target_v_sdis'], ingr_name, absolute_gen_offset, n_gen)
    
    # This runs the heavy mathematical optimization loop
    res = minimize(
        problem,
        algorithm,
        ('n_gen', target_gen),
        seed=1,
        callback=callback,
        save_history=False,
        verbose=False,
        copy_algorithm=False,
        copy_termination=False
    )
    
    pool.close()
    
    # Extract the best result
    k_absp_opt = res.X[0]
    k_digestrate_opt = res.X[1]
    Kp_endog_min_opt = res.X[2]
    
    print("\nOptimization Complete!")
    print("best K_absp_opt:", k_absp_opt)
    print("best K_digestrate_opt:", k_digestrate_opt)
    print("best K_endog_opt:", Kp_endog_min_opt)
    
    print("\nRefining best result...")
    try:
        # Re-run simulation with full DataFrames to get the final trajectory
        duodenum_2, jejunum_2, ileum_2, df_fore = run_simulation(
            ingr_name, constants, k_absp_opt, k_digestrate_opt, Kp_endog_min_opt, t_eval, t_span, export_dataframes=True
        )
        
        opt_results = [[k_absp_opt, k_digestrate_opt, Kp_endog_min_opt]]
        column_names = [f"k_absp_{ingr_name}", f"k_digestrate_{ingr_name}", f"k_endog_{ingr_name}"]
        df_opt_results = pd.DataFrame(opt_results, columns=column_names)
        
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "optimization_results.xlsx")
        df_opt_results.to_excel(results_file, index=False)
        print(f"All results saved to '{results_file}'")
        
        return (k_absp_opt, k_digestrate_opt, Kp_endog_min_opt, res.X, duodenum_2, jejunum_2, ileum_2)
    except Exception as e:
        print(f"Refinement simulation failed: {e}")
        return None

