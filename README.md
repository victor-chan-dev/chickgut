# Summary
This package models the intestinal tract of a chicken

It's main purpose is to determine which feeds are most optimal by determining how much protein
is left after a chicken poops.

# How to run
* Run the following in bash/terminal to run the interactive simulation:
```bash
PYTHONPATH=src venv/bin/python -m chickgut.main
```

# Testing

The project has a comprehensive test suite covering both unit and integration tests. All outputs written during testing are directed to sandbox folders to avoid workspace pollution.

### Run All Tests
To run the full test suite (unit and integration tests) along with a code coverage report:
```bash
PYTHONPATH=src venv/bin/pytest --cov=src/chickgut tests/
```

### Run Unit Tests Only
Unit tests verify individual components (anatomy physical property calculations, spreadsheet exports, diet characteristics loading, and mock-based parameter optimization runs) without running slow ODE integrations. To run only unit tests:
```bash
PYTHONPATH=src venv/bin/pytest -k "not integration" tests/
```

### Run Integration Tests Only
Integration tests run a complete simulation for a baseline ingredient and verify numerical output vectors against a stored golden dataset to ensure no mathematical regression. To run only integration tests:
```bash
PYTHONPATH=src venv/bin/pytest -k "integration" tests/
```

### Regenerating Golden Reference Data
If you modify the underlying ODE physics, anatomy formulas, or base parameters, the regression integration tests will fail. You can regenerate the golden reference pickles by running:
```bash
PYTHONPATH=src venv/bin/python tests/generate_golden_data.py
```

# Todo List:
* Move critical code out of main.py and into its own separate area 
* !Bottlenecks are caused by the `solve_ivp` function. Ideally move to JAX to use GPU instead
* !Replace existing differential evolution solution with scipy to use pymoo
    * Add checkpointing via pymoo + pickles

## Done
* Restructure the python code ✅
* Write a "How do you run this" guide in the ReadMe ✅
* Time each part to understand bottle necks ✅
* Write tests to verify existing functionality (pytest suite) ✅
* Add a code coverage report (pytest-cov) ✅
* Fix NumPy deprecation warnings during solver runs ✅
* Add detailed execution logging and time remaining estimates during parameter optimization ✅

## Maybe
* *Bonus* - Find out how to implement a Digital Twin-like system where we can have multiple computers computing at once 
    * V: This is not necessary since this would require 2 computers. It's a nice-to-have for something more complex

# Personal Notes & Insights

* **Performance & Scale Challenge**:
  * With a population size of 100 and 1,000 maximum iterations, the optimization requires **100,000 evaluations**.
  * At 37 seconds per evaluation, this takes **1,027 hours (~42 days)** to complete. Even if stripped to pure ODE speed (~9 seconds), it still takes **250 hours (~10 days)**.
  * **Solution Strategy**: Run feed ingredient optimizations sequentially (one at a time) with persistent checkpointing (saving progress) and parallelize execution across CPU cores.
* **Apparent Digestibility of Protein**:
  * Feed efficiency is measured by how much protein is absorbed. The lower the remaining protein flux at the end of the ileum, the more digestible the feed ingredient.

# Performance Profiling

### Previous SciPy Baseline
A single HindGIT simulation run took approximately **34 to 37 seconds**. The major execution time was spent in the differential equation solvers (`solve_ivp`):

| Anatomy Section | Duration (avg) | Key Bottleneck |
| :--- | :--- | :--- |
| **Duodenum** | ~20 - 21s | Solving slowly-digested protein equations (~18s) |
| **Jejunum** | ~7s | Solving rapidly-digested protein equations (~6.7s) |
| **Ileum** | ~5 - 6s | Solving rapidly-digested protein equations (~5.4s) |

### New JAX Implementation
A single HindGIT simulation run now takes approximately **37 seconds** using our JAX hybrid implementation. The primary bottleneck is **no longer ODE solving**, but memory allocation during Pandas DataFrame creation.

| Operations Phase | Duration (avg) | Key Bottleneck |
| :--- | :--- | :--- |
| **JAX ODE Solvers** | ~9s | `diffrax.diffeqsolve` evaluates mathematical physics models |
| **DataFrame Construction** | ~28s | Extracting large JAX matrices into `pd.DataFrame` and `jax.vmap` calculation |

*Note: For large evolutionary batches using PyMoo, stripping the Pandas DataFrames directly out of the objective loop reduces execution time to roughly ~9 seconds per evaluation.*