import os
import pickle
import time
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.pool import ThreadPool

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from chickgut.simulation import run_simulation, precompute_foregut, pure_sim_flux
import jax
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize

class ChickgutProblem(Problem):
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
        self.t_eval = tuple(t_eval) # Convert to tuple for JAX static_argnames
        self.t_span = t_span
        self.ingr_name = ingr_name
        self.constants = constants
        self.protein_data = constants['target_v_sdis']
        
        # Precompute foregut once per optimization run
        print(f"[{ingr_name.upper()}] Precomputing foregut for PyMoo JAX execution...")
        result_fore, self.iDuo_g, self.BWeight_kgb, self.Kp_PVG_min = precompute_foregut(constants, t_eval, t_span)
        self.fore_t = jnp.array(result_fore.t)
        self.fore_y = jnp.array(result_fore.y.T)
        
    def _evaluate(self, x, out, *args, **kwargs):
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pop_size = x.shape[0]
        print(f"[{now}] [Ingredient: {self.ingr_name.upper()}] Evaluating population of {pop_size} via JAX vmap...")
        
        import functools
        @functools.partial(jax.jit, static_argnames=['t_eval', 't_span', 'constants_tuple', 'iDuo_g', 'BWeight_kgb', 'Kp_PVG_min'])
        def batched_sim(params_batch, t_eval, t_span, constants_tuple, fore_t, fore_y, iDuo_g, BWeight_kgb, Kp_PVG_min):
            constants_dict = dict(constants_tuple)
            return jax.vmap(pure_sim_flux, in_axes=(0, None, None, None, None, None, None, None, None))(
                params_batch, jnp.array(t_eval), t_span, constants_dict, fore_t, fore_y, list(iDuo_g), BWeight_kgb, Kp_PVG_min
            )
            
        try:
            flux_totals = batched_sim(
                jnp.array(x), self.t_eval, self.t_span, tuple(self.constants.items()), 
                self.fore_t, self.fore_y, tuple(self.iDuo_g), self.BWeight_kgb, self.Kp_PVG_min
            )
            
            pred_protein = (1 - (flux_totals / self.constants['FI_24h_gbird'])) * 100
            
            # SSE calculation broadcasted over population
            SSE_protein = jnp.sum((pred_protein[:, None] - self.protein_data) ** 2, axis=1)
            
            out["F"] = np.array(SSE_protein).reshape(-1, 1)
            
        except Exception as e:
            print("Vmap execution failed:", e)
            out["F"] = np.full((pop_size, 1), 1e10)

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
    
    # We don't need a ThreadPool anymore because JAX vmap naturally parallelizes
    # across the CPU/GPU with vectorized instructions!
    # pool = ThreadPool(n_threads)
    
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
    
    # pool.close()
    
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

def optimize_params_autodiff(t_eval, t_span, ingr_name, constants, output_dir="."):
    """
    Fits the model parameters using JAX Autodiff and Scipy L-BFGS-B gradient descent.
    """
    print(f"\nStarting AUTODIFF gradient optimization for {ingr_name}...")
    
    result_fore, iDuo_g, BWeight_kgb, Kp_PVG_min = precompute_foregut(constants, t_eval, t_span)
    fore_t = jnp.array(result_fore.t)
    fore_y = jnp.array(result_fore.y.T)
    
    target_protein = constants['target_v_sdis']
    FI_24h_gbird = constants['FI_24h_gbird']
    
    import functools
    @functools.partial(jax.jit, static_argnames=['t_eval', 't_span', 'constants_tuple', 'iDuo_g', 'BWeight_kgb', 'Kp_PVG_min'])
    def loss_fn(params, t_eval, t_span, constants_tuple, fore_t, fore_y, iDuo_g, BWeight_kgb, Kp_PVG_min):
        constants_dict = dict(constants_tuple)
        sum_flux_total = pure_sim_flux(params, jnp.array(t_eval), t_span, constants_dict, fore_t, fore_y, list(iDuo_g), BWeight_kgb, Kp_PVG_min)
        pred_protein = (1 - (sum_flux_total / FI_24h_gbird)) * 100
        return jnp.sum((pred_protein - target_protein) ** 2)
        
    import datetime
    eval_state = {'count': 0, 'start_time': time.time()}
    
    def scipy_objective(x):
        eval_state['count'] += 1
        elapsed = time.time() - eval_state['start_time']
        
        # Powell averages ~200-300 evaluations. We estimate 250 for the ETA.
        avg_time = elapsed / eval_state['count']
        remaining = max(0, 250 - eval_state['count'])
        eta_sec = remaining * avg_time
        
        eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[{current_time}] Eval {eval_state['count']:03d} | X: k_absp={x[0]:.4f}, k_dig={x[1]:.4f}, Kp_endog={x[2]:.4f} | Elapsed: {elapsed_str} | ETA: ~{eta_str} ... ", end='', flush=True)
        val = loss_fn(
            jnp.array(x), tuple(t_eval), t_span, tuple(constants.items()), fore_t, fore_y, tuple(iDuo_g), BWeight_kgb, Kp_PVG_min
        )
        print(f"Loss: {val:12.4f}  <-- (Target: 0.0000)", flush=True)
        return float(val)
        
    # Standard initial guess (industry standard fallback)
    x0 = np.array([63.7867, 0.0878, 0.0142])
    bounds = [(0.001, 100.0), (0.001, 0.1), (0.001, 0.1)]
    
    res = scipy_minimize(scipy_objective, x0, method='Powell', bounds=bounds, options={'disp': True})
    
    print("\nAutodiff Optimization Complete!")
    print(f"best K_absp_opt: {res.x[0]}")
    print(f"best K_digestrate_opt: {res.x[1]}")
    print(f"best K_endog_opt: {res.x[2]}")
    
    print("\nRefining best result...")
    try:
        duodenum_2, jejunum_2, ileum_2, df_fore = run_simulation(
            ingr_name, constants, res.x[0], res.x[1], res.x[2], t_eval, t_span, export_dataframes=True
        )
        
        opt_results = [[res.x[0], res.x[1], res.x[2]]]
        column_names = [f"k_absp_{ingr_name}", f"k_digestrate_{ingr_name}", f"k_endog_{ingr_name}"]
        df_opt_results = pd.DataFrame(opt_results, columns=column_names)
        
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "autodiff_results.xlsx")
        df_opt_results.to_excel(results_file, index=False)
        print(f"All results saved to '{results_file}'")
        
        return (res.x[0], res.x[1], res.x[2], res.x, duodenum_2, jejunum_2, ileum_2)
    except Exception as e:
        print(f"Refinement simulation failed: {e}")
        return None
