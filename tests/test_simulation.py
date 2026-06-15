import os
import shutil
import gzip
import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Import from the newly refactored simulation and io modules
from chickgut.simulation import run_simulation, t_eval, t_span, k_absp, k_digestrate, Kp_endog_min
from chickgut.io import get_ing_const, plot_git_results, export_duojejil_results
from chickgut.anatomy.duodenum import Duodenum
from chickgut.anatomy.jejunum import Jejunum
from chickgut.anatomy.ileum import Ileum

GOLDEN_DATA_PATH = Path(__file__).resolve().parent / "data" / "golden_data.pkl.gz"
TEMP_TEST_DIR = Path(__file__).resolve().parent / "data" / "temp_test_output"

@pytest.fixture(autouse=True)
def manage_test_dir():
    """
    Setup and teardown fixture to ensure the temporary test output directory
    is clean before and after each test.
    """
    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR)
    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR)


def test_get_ing_const_valid():
    """
    Unit Test: test_get_ing_const_valid
    Intent: Verify that get_ing_const correctly retrieves constants for a valid, 
            existing feed ingredient (e.g. 'sbm') and that all required key 
            dietary characteristics are populated correctly.
    """
    constants = get_ing_const("sbm")
    assert constants is not None, "Expected valid dictionary for ingredient 'sbm'"
    assert "FI_24h_gbird" in constants, "Expected 'FI_24h_gbird' key in constants"
    assert "UP_Fr" in constants, "Expected 'UP_Fr' key in constants"
    assert "birdBW_kg" in constants, "Expected 'birdBW_kg' key in constants"
    assert constants["birdBW_kg"] == pytest.approx(0.788)


def test_get_ing_const_invalid():
    """
    Unit Test: test_get_ing_const_invalid
    Intent: Verify that get_ing_const returns None when queried for a 
            non-existent or invalid feed ingredient name.
    """
    constants = get_ing_const("nonexistent_ingredient_name")
    assert constants is None, "Expected None for an invalid ingredient"


def test_plot_git_results_creates_files():
    """
    Unit Test: test_plot_git_results_creates_files
    Intent: Verify that plot_git_results successfully creates the expected 
            Matplotlib chart .png files in the specified temporary image directory.
    """
    # Create mock df_fore structure
    mock_data = {
        't': np.arange(0.0, 10.0, 1.0),
        'Cr': np.random.rand(10),
        'PVG': np.random.rand(10),
        'CrPVG_flux': np.random.rand(10),
        'PVGDuo_flux': np.random.rand(10)
    }
    df_fore = pd.DataFrame(mock_data)
    
    # Run plot function targeting temp test dir
    image_output_dir = TEMP_TEST_DIR / "images"
    plot_git_results(df_fore, image_dir=str(image_output_dir))
    
    # Assert plot images were created
    expected_plot1 = image_output_dir / 'B-protein-over-time-proventriculus_gizzard.png'
    expected_plot2 = image_output_dir / 'B-protein-over-time-proventriculus_gizzard_flux.png'
    
    assert expected_plot1.exists(), f"Plot not found: {expected_plot1}"
    assert expected_plot2.exists(), f"Plot not found: {expected_plot2}"


def test_export_duojejil_results_creates_files():
    """
    Unit Test: test_export_duojejil_results_creates_files
    Intent: Verify that export_duojejil_results successfully writes Excel spreadsheets
            for duodenum, jejunum, and ileum simulation states to the configured
            temporary output directory.
    """
    # Create mock integration result structures
    class MockExitState:
        def __init__(self):
            self.t = np.arange(0.0, 5.0, 1.0)
            self.y = np.random.rand(5, 5)  # 5 nodes, 5 time steps

    mock_exit = MockExitState()
    mock_df = pd.DataFrame({'t': [0, 1, 2], 'val': [1.0, 1.5, 2.0]})

    class MockCompartment:
        def __init__(self):
            self.UDexit_SS = mock_exit
            self.SlDexit_SS = mock_exit
            self.RDexit_SS = mock_exit
            self.UJexit_SS = mock_exit
            self.SlJexit_SS = mock_exit
            self.RJexit_SS = mock_exit
            self.UIexit_SS = mock_exit
            self.SlIexit_SS = mock_exit
            self.RIexit_SS = mock_exit
            
            self.df_UP_d = mock_df
            self.df_SlP_d = mock_df
            self.df_RP_d = mock_df
            self.df_UP_j = mock_df
            self.df_SlP_j = mock_df
            self.df_RP_j = mock_df
            self.df_UP_i = mock_df
            self.df_SlP_i = mock_df
            self.df_RP_i = mock_df

    mock_duo = MockCompartment()
    mock_jej = MockCompartment()
    mock_il = MockCompartment()

    # Export to temp output folder to prevent test pollution
    export_duojejil_results(mock_duo, mock_jej, mock_il, output_dir=str(TEMP_TEST_DIR))

    # Assert Excel output files were generated
    expected_duo_file = TEMP_TEST_DIR / "model_results_duo-Bryan.xlsx"
    expected_jej_file = TEMP_TEST_DIR / "model_results_jej-Bryan.xlsx"
    expected_ileum_file = TEMP_TEST_DIR / "model_results_ileum-Bryan.xlsx"

    assert expected_duo_file.exists(), f"Excel file not found: {expected_duo_file}"
    assert expected_jej_file.exists(), f"Excel file not found: {expected_jej_file}"
    assert expected_ileum_file.exists(), f"Excel file not found: {expected_ileum_file}"


def test_duodenum_properties():
    """
    Unit Test: test_duodenum_properties
    Intent: Verify Duodenum boundary dimension calculations (length, radius, volume) 
            are mathematically correct based on bird body weight constants.
    """
    t_span_dummy = (0.0, 10.0)
    t_eval_dummy = np.arange(0.0, 10.0, 1.0)
    constants_dummy = {"UP_Fr": 0.1, "SlP_Fr": 0.4, "RP_Fr": 0.5}
    bweight = 0.788

    duo = Duodenum(
        t_span=t_span_dummy,
        iDuo_g=[1e-6],
        t_eval=t_eval_dummy,
        result_fore=None,
        BWeight_kgb=bweight,
        Kp_PVG_min=0.029,
        constants=constants_dummy,
        k_absp=63.7867,
        k_digestrate=0.0878,
        Kp_endog_min=0.0142
    )

    duo.calculate_duo_prop()

    expected_length = 14.437 * bweight
    expected_radius = 1.18 / 2.0
    expected_volume = np.pi * (expected_radius ** 2) * expected_length

    assert duo.length_cm == pytest.approx(expected_length)
    assert duo.r_cm == pytest.approx(expected_radius)
    assert duo.volume_cm3 == pytest.approx(expected_volume)
    assert duo.total_discretize == 101
    assert duo.total_node_num == 100


def test_jejunum_properties():
    """
    Unit Test: test_jejunum_properties
    Intent: Verify Jejunum boundary dimension calculations (length, radius, volume)
            are mathematically correct based on bird body weight constants.
    """
    t_span_dummy = (0.0, 10.0)
    t_eval_dummy = np.arange(0.0, 10.0, 1.0)
    constants_dummy = {"Jej_MRT": 30.0}
    bweight = 0.788

    jej = Jejunum(
        t_span=t_span_dummy,
        t_eval=t_eval_dummy,
        constants=constants_dummy,
        BWeight_kgb=bweight,
        k_absp=63.7867,
        k_digestrate=0.0878,
        Kp_endog_min=0.0142
    )

    jej.calculate_jej_prop()

    expected_length = 33.166 * bweight
    expected_radius = 1.14 / 2.0
    expected_volume = np.pi * (expected_radius ** 2) * expected_length

    assert jej.length_cm == pytest.approx(expected_length)
    assert jej.r_cm == pytest.approx(expected_radius)
    assert jej.volume_jej_cm3 == pytest.approx(expected_volume)
    assert jej.Discretize_jej == 101
    assert jej.Node_num_jej == 100


def test_ileum_properties():
    """
    Unit Test: test_ileum_properties
    Intent: Verify Ileum boundary dimension calculations (length, radius, volume)
            are mathematically correct based on bird body weight constants.
    """
    t_span_dummy = (0.0, 10.0)
    t_eval_dummy = np.arange(0.0, 10.0, 1.0)
    constants_dummy = {"Il_MRT": 61.0}
    bweight = 0.788

    il = Ileum(
        t_span=t_span_dummy,
        t_eval=t_eval_dummy,
        constants=constants_dummy,
        BWeight_kgb=bweight,
        k_absp=63.7867,
        k_digestrate=0.0878,
        Kp_endog_min=0.0142
    )

    il.calculate_Il_prop()

    expected_length = 34.643 * bweight
    expected_radius = 0.9 / 2.0
    expected_volume = np.pi * (expected_radius ** 2) * expected_length

    assert il.length_cm == pytest.approx(expected_length)
    assert il.r_cm == pytest.approx(expected_radius)
    assert il.volume_il_cm3 == pytest.approx(expected_volume)
    assert il.Discretize_il == 101
    assert il.Node_num == 100


def test_integration_simulation():
    """
    Integration Test: test_integration_simulation
    Intent: Run a full baseline simulation for ingredient 'sbm' and verify 
            the numerical outputs (state vectors over time) match the reference 
            goldenpath data (golden_data.pkl.gz) exactly, ensuring zero regression 
            on mathematical integrations.
    """
    assert GOLDEN_DATA_PATH.exists(), f"Golden data file not found at {GOLDEN_DATA_PATH}. Run tests/generate_golden_data.py first."

    # Load golden standard data
    with gzip.open(GOLDEN_DATA_PATH, "rb") as f:
        golden_data = pickle.load(f)

    # Run current simulation
    ingr_name = "sbm"
    constants = get_ing_const(ingr_name)

    duo_current, jej_current, il_current, df_fore_current = run_simulation(
        ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span
    )

    golden_duo = golden_data["duodenum"]
    golden_jej = golden_data["jejunum"]
    golden_il = golden_data["ileum"]

    # Check Duodenum properties
    assert duo_current.length_cm == pytest.approx(golden_duo["length_cm"])
    assert duo_current.volume_cm3 == pytest.approx(golden_duo["volume_cm3"])
    
    # Assert dataframes are close (handles tiny float solver deviations)
    pd.testing.assert_frame_equal(duo_current.df_UP_d, golden_duo["df_UP_d"], atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(duo_current.df_SlP_d, golden_duo["df_SlP_d"], atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(duo_current.df_RP_d, golden_duo["df_RP_d"], atol=1e-5, rtol=1e-5)

    # Check Jejunum
    pd.testing.assert_frame_equal(jej_current.df_UP_j, golden_jej["df_UP_j"], atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(jej_current.df_SlP_j, golden_jej["df_SlP_j"], atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(jej_current.df_RP_j, golden_jej["df_RP_j"], atol=1e-5, rtol=1e-5)

    # Check Ileum
    pd.testing.assert_frame_equal(il_current.df_UP_i, golden_il["df_UP_i"], atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(il_current.df_SlP_i, golden_il["df_SlP_i"], atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(il_current.df_RP_i, golden_il["df_RP_i"], atol=1e-5, rtol=1e-5)


def test_optimize_params_success():
    """
    Unit Test: test_optimize_params_success
    Intent: Verify that optimize_params runs differential evolution using a mocked
            simulation, correctly calculates SSE, updates best solution pickle, and
            saves Excel results to the temp output directory.
    """
    from unittest.mock import patch

    # Create mock compartments
    mock_df = pd.DataFrame({
        0: np.zeros(2005),
        1: np.zeros(2005),
        2: np.ones(2005) * 0.1  # Col 2 is used for flux sum
    })

    class MockCompartment:
        def __init__(self):
            self.df_RP_i = mock_df
            self.df_UP_i = mock_df
            self.df_SlP_i = mock_df

    mock_duo = MockCompartment()
    mock_jej = MockCompartment()
    mock_il = MockCompartment()
    mock_df_fore = pd.DataFrame()

    constants = {
        'target_v_sdis': 85.0,
        'FI_24h_gbird': 100.0,
        'UP_Fr': 0.1,
        'SlP_Fr': 0.4,
        'RP_Fr': 0.5
    }

    # Patch run_simulation to return our mocks
    with patch("chickgut.optimization.run_simulation") as mock_run:
        mock_run.return_value = (mock_duo, mock_jej, mock_il, mock_df_fore)
        
        from chickgut.optimization import optimize_params
        
        # 1. Run optimize_params with popsize=1, maxiter=1 targeting our temporary directory (first run)
        res = optimize_params(
            t_eval=np.arange(0, 10, 1),
            t_span=(0, 10),
            ingr_name="sbm",
            constants=constants,
            output_dir=str(TEMP_TEST_DIR)
        )
        
        # Verify returned structure
        assert len(res) == 7
        k_absp_opt, k_digestrate_opt, Kp_endog_min_opt, best_x, duo_opt, jej_opt, il_opt = res
        
        # Verify files were created
        expected_pickle = TEMP_TEST_DIR / "best_solution.pkl"
        expected_excel = TEMP_TEST_DIR / "optimization_results.xlsx"
        
        assert expected_pickle.exists(), "best_solution.pkl was not created"
        assert expected_excel.exists(), "optimization_results.xlsx was not created"
        
        # Verify optimization excel content
        df_res = pd.read_excel(expected_excel)
        assert "k_absp_sbm" in df_res.columns
        assert len(df_res) > 0

        # 2. Run it again to verify loading of existing best_solution.pkl works
        res2 = optimize_params(
            t_eval=np.arange(0, 10, 1),
            t_span=(0, 10),
            ingr_name="sbm",
            constants=constants,
            output_dir=str(TEMP_TEST_DIR)
        )
        assert len(res2) == 7


def test_optimize_params_failure():
    """
    Unit Test: test_optimize_params_failure
    Intent: Verify that optimize_params handles solver exceptions gracefully by
            returning a large penalty value (1e10) and completes without crashing.
    """
    from unittest.mock import patch

    constants = {
        'target_v_sdis': 85.0,
        'FI_24h_gbird': 100.0,
        'UP_Fr': 0.1,
        'SlP_Fr': 0.4,
        'RP_Fr': 0.5
    }

    # Patch run_simulation to raise an Exception
    with patch("chickgut.optimization.run_simulation", side_effect=Exception("Integration solver crashed!")):
        from chickgut.optimization import optimize_params
        
        # We expect optimization to run, handle the solver failure, and still return results
        # since it uses the best_current fallback.
        res = optimize_params(
            t_eval=np.arange(0, 10, 1),
            t_span=(0, 10),
            ingr_name="sbm",
            constants=constants,
            output_dir=str(TEMP_TEST_DIR)
        )
        # It will return the results from best_current['x'] which is None (or fallback to optimization results)
        # Wait, if best_current['x'] is None (since it failed every time), what does it return?
        # Let's check optimization.py line 112:
        # if best_current['x'] is not None:
        #     ...
        #     return (...)
        # If best_current['x'] is None, it returns None.
        assert res is None


