import os
import pickle
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from chickgut.simulation import run_simulation

def optimize_params(t_eval, t_span, ingr_name, constants, output_dir="."):
    """
    Fits the model parameters (k_absp, k_digestrate, Kp_endog_min) using differential evolution.
    Saves optimization results and the best solution pickle to the specified output directory.
    """
    protein_data = constants['target_v_sdis']
    best_current = {'x': None, 'fun': np.inf}
    evaluation_count = 0
    eval_times = []
    best_solution_path = os.path.join(output_dir, "best_solution.pkl")

    def SSE_function(opt_params):
        nonlocal evaluation_count
        evaluation_count += 1
        eval_start = time.time()
        print(f"\n## Evaluation {evaluation_count}")
        print(f"Parameters: k_absp={opt_params[0]:.6f}, k_digestrate={opt_params[1]:.6f}, Kp_endog_min={opt_params[2]:.6f}")

        k_absp = opt_params[0]
        k_digestrate = opt_params[1]
        Kp_endog_min = opt_params[2]

        try:
            # Solve ODEs (bypassing Pandas for speed)
            duodenum_instance, jejunum_instance, ileum_instance, df_fore = run_simulation(
                ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span, export_dataframes=False
            )
            
            sum_UI_flux, sum_SlI_flux, sum_RI_flux = ileum_instance.get_flux_sums(limit=2001)
            
            sum_flux_total = sum_RI_flux + sum_UI_flux + sum_SlI_flux 

            pred_protein = (1 - (sum_flux_total / constants['FI_24h_gbird'])) * 100

            SSE_protein = np.sum((pred_protein - protein_data) ** 2)
            my_callback.last_sse = SSE_protein
            
            eval_duration = time.time() - eval_start
            eval_times.append(eval_duration)
            avg_duration = sum(eval_times) / len(eval_times)
            
            # Estimate remaining time assuming ~30 evaluations
            est_remaining_evals = max(0, 30 - evaluation_count)
            est_remaining_time = est_remaining_evals * avg_duration
            
            print(f"Prediction: {pred_protein:.4f}% | Target: {protein_data:.4f}% | SSE: {SSE_protein:.4f}")
            print(f"Evaluation took {eval_duration:.1f}s (Avg: {avg_duration:.1f}s, Est. remaining: {est_remaining_time/60:.1f} min)")
            
            return SSE_protein
        
        except Exception as e:
            eval_duration = time.time() - eval_start
            eval_times.append(eval_duration)
            print(f"Solver failed at evaluation {evaluation_count}: {e}")
            print(f"Parameters that caused failure: k_absp={k_absp}, k_digestrate={k_digestrate}, Kp_endog_min={Kp_endog_min}")
            return 1e10
        
    if os.path.exists(best_solution_path):
        with open(best_solution_path, "rb") as f:
            best_current = pickle.load(f)
    else:
        best_current = {'x': None, 'fun': np.inf}
    
    bounds_params = [(0, 100), (0, 0.1), (0, 0.1)]    
    opt_results = []

    def my_callback(xk, convergence=0):
        last_sse = getattr(my_callback, 'last_sse', None)
        if last_sse is not None and last_sse < best_current['fun']:
            best_current['x'] = xk.copy()
            best_current['fun'] = last_sse
            
        with open(best_solution_path, 'wb') as f:
            pickle.dump(best_current, f)
        if last_sse is not None:
            print(f"saved better solution: SSE={last_sse}")
        else:
            print("saved solution status (no valid evaluation in this generation)")


    Optimize_params_result = differential_evolution(
        SSE_function, 
        bounds_params, 
        popsize=1, 
        maxiter=1, 
        callback=my_callback
    )
        
    Optimized_params = Optimize_params_result.x.copy()
    k_absp_opt = Optimized_params[0]
    k_digestrate_opt = Optimized_params[1]
    Kp_endog_min_opt = Optimized_params[2]
    
    print("best K_absp_opt:", k_absp_opt)
    print("best K_digestrate_opt:", k_digestrate_opt)
    print("best K_endog_opt:", Kp_endog_min_opt)
    SSE_value = Optimize_params_result.fun

    if SSE_value < best_current['fun']:
        best_current['fun'] = SSE_value
        best_current['x'] = Optimized_params.copy()
    
    result_row = [k_absp_opt, k_digestrate_opt, Kp_endog_min_opt]
    opt_results.append(result_row)

    if best_current['x'] is not None:
        print("Refining best result")
        k_absp_opt = best_current['x'][0]
        k_digestrate_opt = best_current['x'][1]
        Kp_endog_min_opt = best_current['x'][2]

        try:
            duodenum_2, jejunum_2, ileum_2, df_fore = run_simulation(
                ingr_name, constants, k_absp_opt, k_digestrate_opt, Kp_endog_min_opt, t_eval, t_span
            )
            
            result_row = [k_absp_opt] + [k_digestrate_opt] + [Kp_endog_min_opt]
            opt_results.append(result_row)
            
            column_names = [f"k_absp_{ingr_name}", f"k_digestrate_{ingr_name}", f"k_endog_{ingr_name}"]
            df_opt_results = pd.DataFrame(opt_results, columns=column_names)
           
            os.makedirs(output_dir, exist_ok=True)
            df_opt_results.to_excel(os.path.join(output_dir, "optimization_results.xlsx"), index=False)
            print(f"All results saved to '{os.path.join(output_dir, 'optimization_results.xlsx')}'")

            return (k_absp_opt, k_digestrate_opt, Kp_endog_min_opt, best_current['x'], duodenum_2, jejunum_2, ileum_2)
        except Exception as e:
            print(f"Refinement simulation failed: {e}")
            return None

