# Structural Estimation – Financial Econometrics II

This repository contains the Python implementation of the structural dynamic model estimated in the accompanying course report. The project implements a firm-level dynamic programming model with investment, cash holdings, productivity shocks, convex capital adjustment costs, and costly external finance, 
and estimates the structural parameters using simulated method of moments (SMM).

## Replication Notes on Estimation Settings

The structural SMM estimation is computationally intensive. The final version of the code provided in this repository runs the full-sample and both subsample estimations using the same optimizer limits specified in `settings.json`:

- `maxiter = 75`
- `maxfev = 150`

These limits apply to all three analyses when the code is executed from the repository.

In the results reported in the paper, the full-sample estimation was run using the same limits (75 iterations and 150 function evaluations). However, due to computational constraints, the subsample estimations (small firms and large firms) were run with tighter limits:

- `maxiter = 35`
- `maxfev = 70`

This difference was necessary to complete the full set of estimation runs within available compute time.

Running the full-sample and both subsample estimations with the current repository settings (75/150 for all runs) takes approximately **36 hours** on a standard laptop environment.

## Main script

### `runSMM.py`
This is the **main script for running the estimation used in the report**. It:
- loads the project configuration from `settings.json`
- loads or rebuilds the cleaned Compustat data
- computes summary statistics
- solves the dynamic program
- runs SMM estimation
- simulates the model at the estimated parameters
- computes standard errors and Jacobian-based identification diagnostics
- saves tables, logs, and figures to the `results/` directory

For the report’s full estimation, this is the script that should be used.

## Key run settings used by `runSMM.py`

The behavior of `runSMM.py` is controlled through `settings.json`.

### Sample selection
- `run_full_sample`: if `true`, runs the full-sample estimation
- `run_small_firms`: if `true`, runs the small-firm subsample estimation
- `run_large_firms`: if `true`, runs the large-firm subsample estimation

These three switches determine which estimation blocks are executed.

### Debug / reduced-scale mode
- `debug_mode`: if `true`, applies reduced-size runtime overrides from `debug_overrides`
- `validation_mode`: selects which debug override profile to use; currently `"fast"` or `"medium"`

When `debug_mode = true`, the script uses smaller simulation sizes and coarser dynamic-programming grids for quicker testing.

### Data handling
- `rebuild_data`: if `true`, rebuilds the cleaned Compustat dataset from the raw file
- `raw_data_path`: path to the raw Compustat data
- `clean_data_path`: path to the cleaned dataset used in estimation

### Starting values and bounds
- `theta0`: starting values for the four estimated parameters  
  `[psi, lambda_external_finance, rho, sigma]`
- `bounds`: lower and upper bounds for each estimated parameter

### Optimizer controls
- `smm_optimizer_method`: optimization routine (currently Nelder–Mead)
- `smm_optimizer_options.maxiter`: maximum number of optimizer iterations
- `smm_optimizer_options.maxfev`: maximum number of objective evaluations
- `smm_optimizer_options.xatol`: parameter tolerance for convergence
- `smm_optimizer_options.fatol`: objective-function tolerance for convergence
- `smm_optimizer_options.adaptive`: whether to use the adaptive Nelder–Mead variant

### Simulation controls
These are set in `settings.json` under `"simulation"` and determine the size of the simulated panel:
- `n_firms`
- `t_periods`
- `burn_in`
- `seed`

### Dynamic-programming grid controls
These are set in `settings.json` under `"dp"` and determine the discretized state and control grids:
- `k_grid_size`, `k_min`, `k_max`: capital grid size and bounds
- `p_grid_size`, `p_min`, `p_max`: cash grid size and bounds
- `z_grid_size`, `tauchen_m`: productivity grid size and Tauchen discretization width
- `control_grid_size`: grid size for investment controls
- `cash_control_grid_size`: grid size for next-period cash choices
- `investment_min`, `investment_max`: admissible investment-rate bounds
- `max_iter`, `tol`: value-function iteration controls

## Diagnostic script

### `run_diagnostics.py`
This script runs the project’s main diagnostic blocks. It is intended for checking numerical stability, optimizer sensitivity, and grid/bound choices outside the main estimation workflow.

The script uses the same `settings.json` file and reads three diagnostic blocks:

### 1. Optimizer diagnostics
Controlled by `optimizer_diagnostics`:
- `enabled`: turns the optimizer diagnostics block on or off
- `diagnostics_only`: runs diagnostics without treating the run as a full estimation workflow
- `run_baseline_check`: solves the model once at the starting parameter vector before diagnostics
- `compute_standard_errors`: computes Jacobian-based standard errors in the diagnostic run
- `make_plots`: saves policy/distribution plots in the diagnostic run
- `run_budget_sweep`: tests different optimizer budgets
- `budget_sweep`: list of alternative `{maxiter, maxfev}` settings to test
- `run_multi_start`: tests multiple starting values
- `multi_start_thetas`: list of labeled alternative starting parameter vectors

### 2. Layer 3 grid / bound diagnostics
Controlled by `layer3_diagnostics`:
- `enabled`: turns the grid/bound diagnostics block on or off
- `diagnostics_only`: diagnostic-only mode
- `make_plots`: saves plots for the diagnostic scenarios
- `fixed_theta`: parameter vector held fixed during the grid experiments
- `scenarios`: list of grid/bound changes to test

Current scenario overrides include:
- `investment_max`
- `k_max`
- `p_max`

These runs are used to evaluate whether the policy functions and simulated moments are sensitive to upper grid bounds and control limits.

### 3. Layer 4 structural diagnostics
Controlled by `layer4_diagnostics`:
- `enabled`: turns the structural diagnostics block on or off
- `diagnostics_only`: diagnostic-only mode
- `make_plots`: saves diagnostic plots
- `save_lambda_policy_plot`: optionally saves a policy-comparison plot
- `fixed_theta`: parameter vector held fixed during the experiments
- `scenarios`: list of model, simulation, or parameter overrides to test

Current scenario overrides can include:
- `theta_overrides`
- `model_overrides`
- `simulation_overrides`
- `dp_overrides`

These runs are used to study how simulated policies and moments respond to changes in cash-related and other structural settings.

## Main project files

### Root files
- `runSMM.py` — main estimation script for the full SMM workflow
- `run_diagnostics.py` — diagnostic runner for optimizer, grid, and structural checks
- `settings.json` — central project configuration file
- `requirements.txt` — Python package requirements

### Source files in `src/`
- `config.py` — loads, validates, and applies runtime configuration overrides
- `data.py` — rebuilds and loads the cleaned Compustat dataset
- `model.py` — defines fixed parameters and model primitives
- `dp_solver.py` — solves the firm’s dynamic program by value-function iteration
- `simulate.py` — simulates firm panels from the solved policy functions
- `moments.py` — computes empirical and simulated moment vectors
- `smm.py` — constructs the SMM objective and runs estimation
- `jacobian.py` — computes Jacobian-based standard errors and identification diagnostics
- `reporting.py` — exports tables and formatted estimation output
- `diagnostics.py` — helper routines for diagnostic analysis
- `diagnostics_plots.py` — policy and simulation diagnostic plots
- `solver_legacy.py` — older solver version retained for comparison / legacy reference
- `__init__.py` — package marker for the `src` module

## Output folders
- `results/tables/` — estimation tables and diagnostic tables
- `results/figures/` — policy plots and simulation figures
- `results/logs/` — saved terminal output and run logs

  
- Large raw and cleaned data files may be omitted from the repository depending on size constraints.
- The main report results are produced from `runSMM.py`, with supplementary numerical checks coming from `run_diagnostics.py`.
