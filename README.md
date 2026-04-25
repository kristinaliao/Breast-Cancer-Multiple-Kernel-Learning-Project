# Project: Predict the Proliferation Score of Invasive Lobular Carcinoma samples using Multiple Kernel Learning

## Project Structure

- `main.py`: Entry point for running the pipeline and experiments.
- `src/`: 
    - `data_processing.py`: Data loading, alignment, and multi-omic preprocessing.
    - `kernels.py`: Kernel computation and normalization functions.
    - `mkl.py`: MKL meta-learner using SLSQP optimization for kernel weighting.
    - `baselines.py`: Random Forest and SVR baseline models with hyperparameter tuning.
    - `biological_lists.py`: Centralized definitions for mRNA pathways and RPPA modules.
- `data/`: Directory for input CSV and GMT files.

## Installation

Required dependencies:
```bash
pip install pandas numpy scikit-learn scipy
```

## Usage

### Standard Mode
Run both baseline models and the MKL model with a consistent 80/20 train/test split:
```bash
python main.py --mode both
```
*Options for `--mode`: `baseline`, `mkl`, `both` (default).*

### Experiments
Run experiments:

1.  **Kernel Pruning**: Compares the full model to one using a subset of biologically relevant kernels.
    ```bash
    python main.py --experiment pruning
    ```
2.  **Ablation Study**: Evaluates the predictive power of mRNA vs. RPPA pathway kernels.
    ```bash
    python main.py --experiment ablation
    ```
3.  **Bootstrapping**: Perform 30 iterations of sampling with replacement to assess stability of kernel weights.
    ```bash
    python main.py --experiment bootstrapping
    ```

## Outputs

- `evaluation_results.txt`: Summary of test metrics and top drivers for standard runs.
- `experiment_results.txt`: Summary results for the selected experiment.
- `bootstrapping_results.txt`: Detailed per-iteration logs and stability analysis for the bootstrap experiment.
- `residual_analysis.txt`: Clinical profile of top outliers (highest prediction errors) from the MKL model.
