# Jit-Opt-Quant: A High-Performance Walk-Forward Optimization Framework

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

Jit-Opt-Quant is a high-performance quantitative research framework for discovering, backtesting, and validating ensemble trading strategies. It is built to combat overfitting by combining a high-speed, **Numba-JIT compiled event-driven backtester** with a robust **Walk-Forward Optimization (WFO)** engine.

This framework is not just a simple backtester; it's a "strategy factory" designed to run a "tournament" that pits hundreds of strategy combinations against each other to find the most robust performers on out-of-sample data.

---

## 🚀 Core Features

* **High-Speed Core:** The entire event-driven backtesting loop and all core technical indicators (RSI, MA, EMA, Stochastic, etc.) are **`@jit` compiled with Numba**. This provides C-like execution speeds, making thousands of optimization calls feasible.
* **Robust Walk-Forward Analysis:** Implements a professional-grade sliding window (Training / Testing / Sliding) to validate strategy performance on unseen data, which is critical for assessing real-world viability.
* **Hybrid Optimization "Tournament":**
    1.  **Early Rounds:** Uses a fast, `vectorized` backtest on *sub-samples* of data for a "cheap" and rapid elimination of weak candidates.
    2.  **Final Round:** Promoted "champions" are subjected to a full, accurate, `event_driven` backtest using **Bayesian Optimization** (`scikit-optimize`) on the complete training set.
* **Parallel Processing:** Leverages `joblib` to run optimization tasks across all available CPU cores, dramatically speeding up the research process.
* **Ensemble Strategies:** The framework is built to find the best *combination* of strategies (e.g., `RSIStrategy` + `BollingerBandsStrategy`), allowing signals to confirm each other.
* **Robust & Safe Indicators:** Indicator calculations are written from scratch and include critical robustness fixes, such as preventing division-by-zero errors in RSI and MFI, and correctly handling zero-range markets in the Stochastic Oscillator.

---

## ⚙️ How It Works: The Optimization Workflow

This framework is designed to be run from the `opt_o_jit.py` script, which automates the entire research process:

1.  **Data Pre-calculation:** Loads the full dataset and pre-computes static, common indicators (like ADX and ATR) just once to save time.
2.  **Walk-Forward Slicing:** The script divides the data into rolling `Training` and `Testing` periods based on the proportions you set (e.g., 50% train, 25% test).
3.  **Candidate Generation:** For each fold, it generates `NUM_ENSEMBLE_CANDIDATES` (e.g., 200) random combinations of strategies.
4.  **The "Tournament" (Hybrid Optimization):**
    * **Round 1 (Elimination):** All candidates are run in `vectorized` mode using `dummy_minimize` (random search) on a *small portion* of the training data. The bottom 50% are eliminated.
    * **Subsequent Rounds:** The surviving candidates are tested on progressively *larger* data portions.
    * **Final Round:** The `TOP_N_CHAMPIONS` (e.g., top 5) are run in the highly accurate `event_driven` mode, using `gp_minimize` (Bayesian optimization) on the *entire* training set.
5.  **Out-of-Sample Validation:** The top 5 champions from the training period are then run *once* on the unseen "Testing" data. Their performance here is the true measure of their quality.
6.  **Logging:** All in-sample training runs and the final out-of-sample champion results are saved to `.csv` files for analysis.
7.  **Slide:** The window slides forward, and the entire process repeats for the next fold.

---

## 🛠️ Technology Stack

* **Python 3.x**
* **Numba:** For high-speed JIT compilation of Python and NumPy code.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **scikit-optimize (`skopt`):** For Bayesian hyperparameter optimization (`gp_minimize`) and random search (`dummy_minimize`).
* **Joblib:** For simple, efficient parallel processing.

---

## 🏁 Getting Started

### 1. Prerequisites

You will need a Python environment and the libraries listed in `requirements.txt`.

### 2. Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/Jit-Opt-Quant.git](https://github.com/YourUsername/Jit-Opt-Quant.git)
    cd Jit-Opt-Quant
    ```

2.  Install the required packages. (It's highly recommended to use a virtual environment).
    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt`:**
    ```
    pandas
    numpy
    numba
    scikit-optimize
    joblib
    ```

### 3. Usage

1.  **Prepare Your Data:**
    * Ensure you have a CSV file with your financial data.
    * The code expects a format where columns are named `[asset_name]_[ohlcv]`, e.g., `BTC_close`, `BTC_high`, `ETH_volume`, etc.
    * Update the `DATA_FILE` constant in `opt_o_jit.py` to point to your file.

2.  **Configure the Optimizer:**
    * Open `opt_o_jit.py` and adjust the configuration constants at the top of the file to match your needs (e.g., `NUM_ENSEMBLE_CANDIDATES`, window proportions, optimization calls).

3.  **Run the Optimization:**
    ```bash
    python opt_o_jit.py
    ```

4.  **Analyze Results:**
    * The script will print its progress for each fold and round.
    * When finished, check the generated CSV files:
        * `hybrid_optimization_champions.csv`: Contains the detailed in-sample and out-of-sample results for the best strategies from each fold.
        * `all_in_sample_training_runs.csv`: A complete log of every single backtest run during the training process, useful for deeper analysis.

---

## 📂 Project Structure
