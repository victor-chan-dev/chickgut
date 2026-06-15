import sys
from chickgut.io import get_ing_const, plot_git_results, export_duojejil_results
from chickgut.simulation import (
    run_simulation, t_eval, t_span, k_absp, k_digestrate, Kp_endog_min
)
from chickgut.optimization import optimize_params

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the Chickgut Digestion Simulation & Optimization")
    parser.add_argument("ingredients", nargs="*", help="List of ingredients to process (e.g., sbm wheat corn)")
    parser.add_argument("--pop-size", type=int, default=40, help="Population size for the optimizer (default: 40)")
    parser.add_argument("--gen-size", type=int, default=150, help="Max iterations/generations for the optimizer (default: 150)")
    
    args = parser.parse_args()
    
    # Allow passing ingredients via command line, else prompt interactively
    if args.ingredients:
        ingredients = [ing.lower().strip() for ing in args.ingredients]
    else:
        ingr_input = input("enter ingredient names (space separated): ").lower().strip()
        ingredients = ingr_input.split()
        
    for ingr_name in ingredients:
        print(f"\n=========================================")
        print(f" Processing Ingredient: {ingr_name.upper()}")
        print(f"=========================================\n")
    
        # Retrieve the characteristics constants for this ingredient
        constants = get_ing_const(ingr_name)
        if not constants:
            print(f"Error: Ingredient '{ingr_name}' not found in diet characteristics CSV.", file=sys.stderr)
            continue

        print("-> [STEP 1] Running Baseline Pre-Optimization Simulation...")
        print("   (This single run will take ~45s because it generates the initial Excel spreadsheets")
        print("   and JAX is analyzing the math for the very first time)\n")

        # 1. Run the baseline simulation using current model parameters
        duodenum_instance, jejunum_instance, ileum_instance, df_fore = run_simulation(
            ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span
        )

        # 2. Plot results (saves charts to 'images/')
        plot_git_results(df_fore, image_dir="images")

        # 3. Export baseline results to spreadsheets (saves files to 'Python_VSCode_Files/')
        export_duojejil_results(duodenum_instance, jejunum_instance, ileum_instance, output_dir="Python_VSCode_Files")

        print("\n-> [STEP 2] Launching PyMoo JAX Optimizer...")
        print("   (The first evaluation takes ~14s while JAX compiles the math into raw machine code.")
        print("   Once compiled, all remaining simulations will fly at cruising speed of ~6.5s!)\n")

        # 4. Perform parameter optimization fitting (saves output to root or config directory)
        # Note: By default this uses pop_size=40 and gen_size=150, which is the mathematically optimal scale.
        optimize_params(
            t_eval, t_span, ingr_name, constants, output_dir=".",
            pop_size=args.pop_size, n_gen=args.gen_size
        )
        
    print("the end")

if __name__ == "__main__":
    main()
