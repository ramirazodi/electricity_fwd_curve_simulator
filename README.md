# Electricity Forward Curve Monte Carlo Simulation

This project simulates forward curves for commodity prices using a Monte Carlo simulation. It takes a set of input parameters, such as price, volatility, and correlations, from an Excel file, simulates price paths, and then visualizes the results using Plotly.

## Overview

The script performs the following tasks:

1. Imports necessary libraries: numpy, pandas, math, and plotly.
2. Defines parameters and file paths.
3. Loads input parameters from an Excel file into a pandas DataFrame.
4. Converts daily volatility to a longer time horizon.
5. Calculates the covariance matrix.
6. Runs the Monte Carlo simulation, generating a set of simulated forward curves.
7. Processes and summarizes the results by calculating mean, median, and various percentiles.
8. Visualizes the simulated forward curves and summary statistics using Plotly.

## How to Use

1. Install the required libraries by running: `pip install -r requirements.txt`.
2. Make sure the input Excel file (`portfolio_parameters.xlsx`) is in the same folder as the script.
3. Run the script: `python forward_curve_simulation.py`.
4. The script will generate several output files:
   - `sim_results_quantiles.csv`: Contains the mean, median, and various percentiles for each delivery month.
   - `sim_results_details.csv`: Contains the simulated forward curves for each simulation run.
   - `sim_results_plot.pdf`: A plot of the simulated forward curves and summary statistics.

## Input File Format

The input Excel file (`portfolio_parameters.xlsx`) should contain the following columns:

- Month: Delivery month tags.
- FWD_Mark: Risk-neutral/no-arbitrage mean prices.
- Daily_STDEV: Daily volatility.
- Jump_Prob: Probability of jump diffusion.
- Jump_Size: Size of the jump.
- Min_Px: Minimum possible price (floor).
- Max_Px: Maximum possible price (ceiling).
- Columns 9-33: Correlation matrix.

The Excel file should have a header row with the column names.

## Customization

You can customize the simulation by adjusting the following parameters in the script:

- `hold`: Holding period in number of trading days.
- `num_sims`: Number of simulation runs.
