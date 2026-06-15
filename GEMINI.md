# Project Overview

This project is a Python-based simulation that models the digestive tract of a broiler chicken. Its primary goal is to analyze the digestion of crude protein (CP) from different feed ingredients to determine their nutritional value. The simulation is divided into different anatomical sections of the chicken's gut: the crop, proventriculus/gizzard, duodenum, jejunum, and ileum. It uses a plug-flow reactor model to simulate the movement and digestion of feed through the intestinal tract. The model calculates the amount of protein absorbed and the amount remaining at the end of the digestive process, which is used to assess the digestibility of the feed. The project uses `scipy` for numerical integration and optimization, `pandas` for data manipulation, and `matplotlib` for plotting results.

# Building and Running

To run the simulation, execute the following command in your terminal:

```bash
PYTHONPATH=src venv/bin/python -m chickgut.main
```

To run the test suite and verify code coverage:

```bash
PYTHONPATH=src venv/bin/pytest --cov=src/chickgut tests/
```

The script will prompt you to enter the name of an ingredient to simulate. The available ingredients are defined in `src/chickgut/resources/Plug_Flow_Diet_Charac.csv`.

# Development Conventions

## Project Structure

The project is structured as follows:

*   `src/chickgut/`: The main source code directory.
    *   `main.py`: The entry point of the application. It handles user input, initializes the simulation, and calls the optimization functions.
    *   `anatomy/`: Contains modules for each section of the intestinal tract (`duodenum.py`, `jejunum.py`, `ileum.py`). Each module defines a class that encapsulates the properties and calculations for that specific section.
    *   `utils/`: Contains utility functions, such as performance timers.
    *   `resources/`: Contains data files, such as the characteristics of different diets.
*   `tests/`: Contains tests for the project.
*   `images/`: Contains output plots from the simulation.
*   `Python_VSCode_Files/`: Contains exported data from the simulation in CSV and Excel formats.

## Coding Style

The code is written in an object-oriented style. Each section of the intestine is represented by a class. The main simulation logic is orchestrated in the `main.py` file. The code makes extensive use of scientific libraries like `numpy`, `pandas`, and `scipy`.

## Key Files

*   `src/chickgut/main.py`: The main entry point of the simulation. It contains the main `GIT` (Gastrointestinal Tract) function that runs the simulation, as well as the `optimize_params` function that uses differential evolution to find the optimal parameters for the model.
*   `src/chickgut/anatomy/duodenum.py`, `src/chickgut/anatomy/jejunum.py`, `src/chickgut/anatomy/ileum.py`: These files define the classes for each section of the small intestine. They contain the methods for calculating the digestion and passage of protein through each section.
*   `src/chickgut/resources/Plug_Flow_Diet_Charac.csv`: This CSV file contains the nutritional characteristics of the different feed ingredients that can be simulated.
