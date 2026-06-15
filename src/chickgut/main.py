import sys
from chickgut.io import get_ing_const, plot_git_results, export_duojejil_results
from chickgut.simulation import (
    run_simulation, t_eval, t_span, k_absp, k_digestrate, Kp_endog_min
)
from chickgut.optimization import optimize_params

def main():
    # Prompt the user for an ingredient name
    ingr_name = input("enter ingredient name: ").lower().strip()
    
    # Retrieve the characteristics constants for this ingredient
    constants = get_ing_const(ingr_name)
    if not constants:
        print(f"Error: Ingredient '{ingr_name}' not found in diet characteristics CSV.", file=sys.stderr)
        sys.exit(1)

    # 1. Run the baseline simulation using current model parameters
    duodenum_instance, jejunum_instance, ileum_instance, df_fore = run_simulation(
        ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span
    )

    # 2. Plot results (saves charts to 'images/')
    plot_git_results(df_fore, image_dir="images")

    # 3. Export baseline results to spreadsheets (saves files to 'Python_VSCode_Files/')
    export_duojejil_results(duodenum_instance, jejunum_instance, ileum_instance, output_dir="Python_VSCode_Files")

    # 4. Perform parameter optimization fitting (saves output to root or config directory)
    optimize_params(t_eval, t_span, ingr_name, constants, output_dir=".")
    
    print("the end")

if __name__ == "__main__":
    main()
