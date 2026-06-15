import gzip
import pickle
import sys
from pathlib import Path

# Add the src directory to the Python path
# This is necessary to import the chickgut modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
print(sys.path)

from chickgut.simulation import run_simulation, t_eval, t_span, k_absp, k_digestrate, Kp_endog_min
from chickgut.io import get_ing_const

def generate_golden_data():
    """
    Runs the simulation with a fixed input and saves the results to a pickle file.
    This file will serve as the "golden standard" for our integration tests.
    """
    ingr_name = "sbm"
    constants = get_ing_const(ingr_name)

    duodenum_instance, jejunum_instance, ileum_instance, df_fore = run_simulation(
        ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span
    )

    golden_data = {
        "duodenum": {
            "length_cm": duodenum_instance.length_cm,
            "volume_cm3": duodenum_instance.volume_cm3,
            "df_UP_d": duodenum_instance.df_UP_d,
            "df_SlP_d": duodenum_instance.df_SlP_d,
            "df_RP_d": duodenum_instance.df_RP_d,
        },
        "jejunum": {
            "df_UP_j": jejunum_instance.df_UP_j,
            "df_SlP_j": jejunum_instance.df_SlP_j,
            "df_RP_j": jejunum_instance.df_RP_j,
        },
        "ileum": {
            "df_UP_i": ileum_instance.df_UP_i,
            "df_SlP_i": ileum_instance.df_SlP_i,
            "df_RP_i": ileum_instance.df_RP_i,
        }
    }

    # Ensure tests/data directory exists
    Path("tests/data").mkdir(parents=True, exist_ok=True)

    # Save as compressed pickle
    with gzip.open("tests/data/golden_data.pkl.gz", "wb") as f:
        pickle.dump(golden_data, f)

    print("Golden data generated successfully.")

if __name__ == "__main__":
    generate_golden_data()

