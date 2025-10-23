import pandas as pd
import random
import numpy as np
import os
import math
from skopt import gp_minimize, dummy_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from joblib import Parallel, delayed
import warnings

# --- IMPORTANT: Import from the NEW optimized backtesting file ---
from pure_jit_bcktst import (
    Backtester, EnsembleStrategy, RSIStrategy, MACrossoverStrategy, MACDStrategy, 
    BollingerBandsStrategy, StochasticOscillatorStrategy, RateOfChangeStrategy, 
    OBVStrategy, MFIStrategy, TMOStrategy,
    calculate_atr, calculate_adx  # Import Pandas versions for pre-calculation
)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DATA_FILE = 'crypto_test_w_volume.csv' # <-- UPDATED Data File
AVAILABLE_STRATEGIES = [
    RSIStrategy, MACrossoverStrategy, MACDStrategy, BollingerBandsStrategy, 
    StochasticOscillatorStrategy, RateOfChangeStrategy, OBVStrategy, 
    MFIStrategy, TMOStrategy
]

# --- Optimization Configuration ---
NUM_ENSEMBLE_CANDIDATES = 200
MIN_OPT_CALLS = 50  # Calls for early rounds (random search)
MAX_OPT_CALLS = 150 # Calls for final round (Bayesian)
MIN_DATA_PROPORTION = 0.25
TOP_N_CHAMPIONS = 5 # <-- NEW: Test Top 5 OOS

# --- Walk-Forward Analysis & Objective ---
TRAINING_WINDOW_PROPORTION = 0.50
TESTING_WINDOW_PROPORTION = 0.25
SLIDING_STEP_PROPORTION = 0.25

MAX_DRAWDOWN_LIMIT = 0.40
MIN_PROFIT_TARGET = 0.01
MIN_CALMAR_TARGET = 0.5

# ==============================================================================
# 2. OBJECTIVE FUNCTION (Worker for Parallelization)
# ==============================================================================
def run_optimization_for_ensemble(
    ensemble_candidate, 
    training_data, 
    n_calls, 
    data_proportion, 
    backtest_mode,
    is_final_round, # <-- NEW Flag
    fold_number
):
    """
    MODIFIED:
    - Uses random block sub-sampling.
    - Uses dummy_minimize (random search) for early rounds.
    - Uses gp_minimize (Bayesian) for final round.
    """
    all_optimization_runs = [] # This list will capture all runs for THIS candidate
    
    # --- NEW: Robust Random-Block Sub-sampling ---
    full_training_len = len(training_data)
    budget_len = int(full_training_len * data_proportion)
    
    if budget_len < full_training_len:
        # Select a random starting point for the block
        max_start = full_training_len - budget_len
        start_idx = random.randint(0, max_start)
        budgeted_training_data = training_data.iloc[start_idx : start_idx + budget_len]
    else:
        budgeted_training_data = training_data
    # --- END NEW ---

    param_space = []
    for i, strategy_class in enumerate(ensemble_candidate):
        for dim in strategy_class.get_hyperparameter_space():
            new_dim = type(dim)(dim.low, dim.high, name=f's{i}_{dim.name}')
            param_space.append(new_dim)

    # --- MODIFIED: Removed SL/TP, changed risk parameter ---
    param_space.extend([
        Real(low=0.01, high=0.1, name='position_size_pct') # 1% to 10% of portfolio per trade
    ])
    # --- END MODIFICATION ---

    @use_named_args(dimensions=param_space)
    def objective_function(**params):
        sub_strategies = []
        for i, strategy_class in enumerate(ensemble_candidate):
            original_params = {
                key.split('_', 1)[1]: value 
                for key, value in params.items() 
                if key.startswith(f's{i}_')
            }
            sub_strategies.append(strategy_class(data=None, assets=[], **original_params))
        
        ensemble_strategy = EnsembleStrategy(data=None, assets=[], sub_strategies=sub_strategies)
        portfolio_params = {'initial_cash': 1000}
        
        # --- MODIFIED: Use new risk parameter ---
        risk_params = {k: v for k, v in params.items() if k in ['position_size_pct']}
        
        try:
            # Pass the (potentially smaller) budgeted data
            backtester = Backtester(budgeted_training_data, ensemble_strategy, {}, risk_params, portfolio_params)
            results = backtester.run(mode=backtest_mode)
            
            if backtest_mode == 'vectorized':
                objective_score = -results.get('returns', -1)
            else:
                total_return, max_dd = results.get('returns', -1), results.get('max_drawdown', -1.0)
                calmar = results.get('calmar_ratio', -10)
                
                # Penalize failures or poor performance heavily
                if results.get('returns', -1) == -1: objective_score = 9999
                elif total_return < MIN_PROFIT_TARGET: objective_score = 1000
                elif abs(max_dd) > MAX_DRAWDOWN_LIMIT: objective_score = 500
                elif calmar < MIN_CALMAR_TARGET: objective_score = 200
                else: objective_score = -calmar # Minimize negative Calmar
            
            # --- NEW: Handle NaN scores BEFORE returning ---
            if np.isnan(objective_score):
                objective_score = 9999
            # --- END NEW ---
            
            run_details = {
                **params, 
                **results, 
                'objective_score': objective_score,
                'fold': fold_number,
                'ensemble_str': str([s.__name__ for s in ensemble_candidate])
            }
            all_optimization_runs.append(run_details)
            return objective_score
        except Exception as e:
            # --- MODIFIED: Exception printing is NOW ENABLED ---
            print(f"Error in objective func (Fold {fold_number}, {ensemble_candidate[0].__name__}, {ensemble_candidate[1].__name__}): {e}")
            return 9999
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # --- NEW: Select optimizer based on round ---
        if is_final_round:
            # Use Bayesian optimization for the final, expensive round
            gp_minimize(
                func=objective_function, 
                dimensions=param_space, 
                n_calls=int(n_calls), 
                n_random_starts=10, 
                random_state=random.randint(1, 1000)
            )
        else:
            # Use fast random search for early, cheap rounds
            dummy_minimize(
                func=objective_function, 
                dimensions=param_space, 
                n_calls=int(n_calls), 
                random_state=random.randint(1, 1000)
            )
        # --- END NEW ---

    if not all_optimization_runs:
        return None, [] # Return None for best_run, empty list for all_runs
        
    best_run = min(all_optimization_runs, key=lambda x: x['objective_score'])
    best_run['ensemble_candidate'] = ensemble_candidate
    
    return best_run, all_optimization_runs

# ==============================================================================
# 3. MAIN WALK-FORWARD LOOP
# ==============================================================================
if __name__ == '__main__':
    total_cores = os.cpu_count()
    n_parallel_jobs = max(1, total_cores - 2 if total_cores is not None else 1)
    print(f"--- Using {n_parallel_jobs} of {total_cores} CPU cores ---\n")

    full_data = pd.read_csv(DATA_FILE, index_col='timestamp', parse_dates=True)
    
    # --- NEW: PRE-COMPUTE STATIC INDICATORS FOR ALL ASSETS ---
    all_parts = [c.rsplit('_', 1) for c in full_data.columns]
    assets = sorted(list(set(parts[0] for parts in all_parts if len(parts) == 2 and 'vwap' not in parts[1])))
    print(f"--- Detected {len(assets)} assets. Pre-calculating static indicators (ATR, ADX)... ---")
    
    for asset in assets:
        try:
            high = full_data[f'{asset}_high']
            low = full_data[f'{asset}_low']
            close = full_data[f'{asset}_close']
            
            # ATR is still needed for ADX calculation
            full_data[f'{asset}_atr_static'] = calculate_atr(high, low, close)
            full_data[f'{asset}_adx_static'] = calculate_adx(high, low, close)
        except KeyError:
            print(f"Warning: Could not find full OHLC data for asset {asset}. Skipping pre-calculation.")
            
    print("--- Static indicators cached. Starting Walk-Forward Analysis... ---")
    # --- END NEW ---
    
    
    out_of_sample_champions = [] # Stores OOS results for the Top-N of each fold
    all_in_sample_training_runs = [] # Stores EVERY run from EVERY fold
    
    fold = 1
    current_start_index = 0
    total_rows = len(full_data)
    training_rows = int(total_rows * TRAINING_WINDOW_PROPORTION)
    testing_rows = int(total_rows * TESTING_WINDOW_PROPORTION)
    sliding_rows = int(total_rows * SLIDING_STEP_PROPORTION)


    while current_start_index + training_rows + testing_rows <= total_rows:
        train_end_index = current_start_index + training_rows
        test_end_index = train_end_index + testing_rows
        
        # Ensure data is copied to avoid SettingWithCopyWarning
        training_data = full_data.iloc[current_start_index:train_end_index].copy()
        testing_data = full_data.iloc[train_end_index:test_end_index].copy()

        if training_data.empty or testing_data.empty or len(training_data) < 250:
            print(f"--- Fold {fold}: Skipping, not enough data (training: {len(training_data)}, testing: {len(testing_data)}) ---")
            current_start_index += sliding_rows
            fold += 1
            break

        print(f"\n--- Fold {fold}: Training on {len(training_data)} rows ({training_data.index.min().date()} to {training_data.index.max().date()}) ---")
        
        promoted_candidates = [random.sample(AVAILABLE_STRATEGIES, 2) for _ in range(NUM_ENSEMBLE_CANDIDATES)]
        num_rounds = int(math.log2(len(promoted_candidates))) if len(promoted_candidates) > 0 else 0
                
        for r in range(num_rounds + 1):
            if not promoted_candidates: break
            n_candidates = len(promoted_candidates)
            
            # --- MODIFIED: Adjust logic for final round ---
            is_final_round = (r == num_rounds) or (n_candidates <= TOP_N_CHAMPIONS)
            mode = 'event_driven' if is_final_round else 'vectorized'
            data_prop = 1.0 if is_final_round else MIN_DATA_PROPORTION + (1 - MIN_DATA_PROPORTION) * (r / num_rounds)
            opt_calls = MAX_OPT_CALLS if is_final_round else MIN_OPT_CALLS
            # --- END MODIFICATION ---
            
            print(f"  -> Round {r+1} ({mode} mode, data_prop={data_prop:.2f}, calls={opt_calls}): Testing {n_candidates} candidates...")

            parallel_results_tuples = Parallel(n_jobs=n_parallel_jobs)(
                delayed(run_optimization_for_ensemble)(
                    c, training_data, opt_calls, data_prop, mode, is_final_round, fold
                ) for c in promoted_candidates
            )
            
            valid_best_runs = sorted(
                [res[0] for res in parallel_results_tuples if res is not None and res[0] is not None], 
                key=lambda x: x['objective_score']
            )
            
            all_runs_from_round = [
                run for res in parallel_results_tuples if res is not None and res[1] is not None for run in res[1]
            ]
            all_in_sample_training_runs.extend(all_runs_from_round)

            if not valid_best_runs: 
                print("  -> No valid results this round.")
                break
            
            # --- MODIFIED: Test Top-N Champions OOS ---
            if is_final_round:
                top_n_champions = valid_best_runs[:TOP_N_CHAMPIONS]
                print(f"  -> Fold {fold} Final Round Complete. Testing Top {len(top_n_champions)} Champions Out-of-Sample...")
                print(f"     (OOS Period: {testing_data.index.min().date()} to {testing_data.index.max().date()})")

                for i, champion_result in enumerate(top_n_champions):
                    ensemble_name = [s.__name__ for s in champion_result['ensemble_candidate']]
                    print(f"    -> Testing Champion {i+1}/{len(top_n_champions)}: {ensemble_name}")
                    print(f"       In-Sample Score: {champion_result.get('objective_score', 0):.2f}, Calmar: {champion_result.get('calmar_ratio', 0):.2f}")

                    champion_ensemble = champion_result['ensemble_candidate']
                    sub_strategies = []
                    for j, strat_class in enumerate(champion_ensemble):
                        strat_params = {key.split('_',1)[1]: value for key, value in champion_result.items() if key.startswith(f's{j}_')}
                        sub_strategies.append(strat_class(data=None, assets=[], **strat_params))

                    ensemble_instance = EnsembleStrategy(data=None, assets=[], sub_strategies=sub_strategies)
                    
                    # --- MODIFIED: Use new risk parameter ---
                    risk_params = {k:v for k,v in champion_result.items() if k in ['position_size_pct']}
                    portfolio_params = {'initial_cash': 1000}

                    oos_backtester = Backtester(testing_data, ensemble_instance, {}, risk_params, portfolio_params)
                    oos_results = oos_backtester.run(mode='event_driven')

                    print(f"       Out-of-Sample Test: Value=${oos_results.get('final_portfolio_value', 0):,.2f}, DD={oos_results.get('max_drawdown', 0):.2%}, Calmar={oos_results.get('calmar_ratio', 0):.2f}")

                    # Store OOS results with rank and fold info
                    oos_results_renamed = {f"oos_{k}": v for k, v in oos_results.items()}
                    champion_result['in_sample_rank'] = i + 1
                    champion_result['fold'] = fold
                    out_of_sample_champions.append({**champion_result, **oos_results_renamed})
                
                break # Exit round loop
            
            # --- End Modified Block ---

            # If not final round, promote top half
            promoted_candidates = [res['ensemble_candidate'] for res in valid_best_runs[:math.ceil(n_candidates / 2)]]
        
        
        current_start_index += sliding_rows
        fold += 1

    # --- SAVE ALL RESULTS ---
    if out_of_sample_champions:
        champions_df = pd.DataFrame(out_of_sample_champions)
        champions_df.to_csv("hybrid_optimization_champions.csv", index=False)
        print("\n--- Top-N Champion results saved to hybrid_optimization_champions.csv ---")
        
    if all_in_sample_training_runs:
        all_runs_df = pd.DataFrame(all_in_sample_training_runs)
        all_runs_df.to_csv("all_in_sample_training_runs.csv", index=False)
        print("--- All training runs saved to all_in_sample_training_runs.csv ---")
    else:
        print("\n--- No optimization runs were recorded. ---")

