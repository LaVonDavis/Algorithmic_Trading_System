import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from skopt.space import Real, Integer
from numba import jit

# ==============================================================================
# 0. STANDALONE, JIT-COMPILED INDICATOR FUNCTIONS
# ==============================================================================

# --- Original JIT Functions ---

@jit(nopython=True)
def jit_calculate_ma(data, period):
    n = len(data)
    if n < period: return np.full(n, np.nan)
    result = np.full(n, np.nan)
    # Calculate initial sum for the first window
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    # Use sliding window for subsequent values
    for i in range(period, n):
        current_sum += data[i] - data[i - period]
        result[i] = current_sum / period
    return result

@jit(nopython=True)
def jit_calculate_ema(data, period):
    n = len(data)
    if n < period: return np.full(n, np.nan)
    result = np.full(n, np.nan)
    alpha = 2.0 / (period + 1.0)
    
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
            
    if first_valid_idx == -1: return result
    
    result[first_valid_idx] = data[first_valid_idx]
    
    for i in range(first_valid_idx + 1, n):
        if not np.isnan(data[i]):
            if np.isnan(result[i-1]): # Handle case where previous was NaN
                 result[i] = data[i]
            else:
                result[i] = alpha * data[i] + (1.0 - alpha) * result[i-1]
        else:
            result[i] = result[i-1]
    return result

# --- NEW: Wilder's MA (RMA) for RSI ---
@jit(nopython=True)
def jit_calculate_rma(data, period):
    n = len(data)
    if n < period: return np.full(n, np.nan)
    result = np.full(n, np.nan)
    alpha = 1.0 / period
    
    # Calculate first value as simple MA
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    
    # Calculate subsequent values using RMA formula
    for i in range(period, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i-1]
    return result

# --- MODIFIED: Correct RSI (uses RMA) ---
@jit(nopython=True)
def jit_calculate_rsi(close, period=14):
    n = len(close)
    if n < period: return np.full(n, np.nan)
    
    gains = np.zeros(n)
    losses = np.zeros(n)
    result = np.full(n, np.nan)
    
    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = -delta
            
    avg_gain = jit_calculate_rma(gains, period)
    avg_loss = jit_calculate_rma(losses, period)
    
    for i in range(period - 1, n):
        # --- ROBUSTNESS FIX ---
        # Add epsilon to denominator to prevent division by zero
        avg_loss_safe = avg_loss[i] + 1e-12 
        rs = avg_gain[i] / avg_loss_safe
        
        # The (1.0 + rs) is already safe because rs >= 0
        result[i] = 100.0 - (100.0 / (1.0 + rs)) 
        # --- END FIX ---
            
    return result

@jit(nopython=True)
def jit_calculate_roc(close, period=12):
    n = len(close)
    if n < period: return np.full(n, np.nan)
    result = np.full(n, np.nan)
    for i in range(period, n):
        if close[i-period] != 0:
            result[i] = ((close[i] - close[i-period]) / close[i-period]) * 100.0
    return result

@jit(nopython=True)
def jit_calculate_obv(close, volume):
    n = len(close)
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv

@jit(nopython=True)
def jit_rolling_std(data, period):
    n = len(data)
    if n < period: return np.full(n, np.nan)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = np.nanstd(window)
    return result

@jit(nopython=True)
def jit_calculate_bollinger_bands(close_np, period, std_dev):
    mean_np = jit_calculate_ma(close_np, period)
    std_np = jit_rolling_std(close_np, period)
    upper_band_np = mean_np + (std_np * std_dev)
    lower_band_np = mean_np - (std_np * std_dev)
    return upper_band_np, lower_band_np

@jit(nopython=True)
def jit_rolling_min(data, period):
    n = len(data)
    if n < period: return np.full(n, np.nan)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = np.nanmin(window)
    return result

@jit(nopython=True)
def jit_rolling_max(data, period):
    n = len(data)
    if n < period: return np.full(n, np.nan)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = np.nanmax(window)
    return result

# --- CRITICAL FIX: Robust Stochastic calculation ---
@jit(nopython=True)
def jit_calculate_stochastic(high_np, low_np, close_np, period_k, period_d):
    low_k_np = jit_rolling_min(low_np, period_k)
    high_k_np = jit_rolling_max(high_np, period_k)
    
    n = len(high_np)
    k_line_np = np.full(n, 50.0) # Default to 50 (neutral)
    
    for i in range(n):
        range_k = high_k_np[i] - low_k_np[i]
        if range_k > 0: # Check for divide-by-zero
            k_line_np[i] = 100.0 * ((close_np[i] - low_k_np[i]) / range_k)
        # If range_k is 0, k_line_np[i] remains 50.0
            
    d_line_np = jit_calculate_ma(k_line_np, period_d)
    return k_line_np, d_line_np
# --- END CRITICAL FIX ---

@jit(nopython=True)
def jit_calculate_mfi(high_np, low_np, close_np, volume_np, mfi_period):
    n = len(high_np)
    if n < mfi_period: return np.full(n, np.nan)

    typical_price = (high_np + low_np + close_np) / 3.0
    money_flow = typical_price * volume_np
    
    pos_money_flow = np.zeros(n)
    neg_money_flow = np.zeros(n)
    
    for i in range(1, n):
        if typical_price[i] > typical_price[i-1]:
            pos_money_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            neg_money_flow[i] = money_flow[i]

    # Use rolling sum via MA * period
    pos_mf_sum = jit_calculate_ma(pos_money_flow, mfi_period) * mfi_period
    neg_mf_sum = jit_calculate_ma(neg_money_flow, mfi_period) * mfi_period

    mfi_np = np.full(n, 50.0) # Default to 50
    for i in range(mfi_period - 1, n):
        # --- ROBUSTNESS FIX ---
        # Add epsilon to denominator to prevent division by zero
        neg_mf_sum_safe = neg_mf_sum[i] + 1e-12
        money_ratio = pos_mf_sum[i] / neg_mf_sum_safe
        
        # The (1.0 + money_ratio) is already safe because money_ratio >= 0
        mfi_np[i] = 100.0 - (100.0 / (1.0 + money_ratio))
        # --- END FIX ---
            
    return mfi_np

# ==============================================================================
# 1. TRADING STRATEGIES (Re-architected for efficiency)
# ==============================================================================
class TradingStrategy(ABC):
    def __init__(self, data, assets, **kwargs):
        self.data, self.assets, self.params = data, assets, kwargs
        self.indicators = {asset: {} for asset in assets}

    def calculate_indicators(self, common_indicators=None):
        if common_indicators is None: common_indicators = {}
        for asset in self.assets:
            if asset not in common_indicators: common_indicators[asset] = {}
            
            # --- MODIFIED: READ PRE-COMPUTED STATIC INDICATORS ---
            # ATR is still needed for ADX calculation
            if 'atr' not in common_indicators[asset]:
                static_col = f'{asset}_atr_static'
                if static_col in self.data.columns:
                    common_indicators[asset]['atr'] = self.data[static_col]
                else: # Fallback
                    high, low, close = self.data[f'{asset}_high'], self.data[f'{asset}_low'], self.data[f'{asset}_close']
                    common_indicators[asset]['atr'] = calculate_atr(high, low, close)
                    
            if 'adx' not in common_indicators[asset]:
                static_col = f'{asset}_adx_static'
                if static_col in self.data.columns:
                    common_indicators[asset]['adx'] = self.data[static_col]
                else: # Fallback
                    high, low, close = self.data[f'{asset}_high'], self.data[f'{asset}_low'], self.data[f'{asset}_close']
                    common_indicators[asset]['adx'] = calculate_adx(high, low, close)
            # --- END MODIFICATION ---
            
        self.indicators = common_indicators
        self._calculate_unique_indicators()

    def _calculate_unique_indicators(self): pass
    @abstractmethod
    def generate_signals(self, asset): pass
    @staticmethod
    @abstractmethod
    def get_hyperparameter_space(): pass

class RSIStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        for asset in self.assets:
            close_np = self.data[f"{asset}_close"].to_numpy()
            rsi_np = jit_calculate_rsi(close_np, period=int(self.params.get('look_back_length', 14)))
            self.indicators[asset]['rsi'] = pd.Series(rsi_np, index=self.data.index)

    def generate_signals(self, asset):
        rsi, adx = self.indicators[asset]['rsi'], self.indicators[asset]['adx']
        buy_thresh, sell_thresh = self.params.get('buy_threshold', 30), self.params.get('sell_threshold', 70)
        adx_thresh = self.params.get('adx_regime_threshold', 23)
        entries = (rsi < buy_thresh) & (adx < adx_thresh)
        exits = (rsi > sell_thresh)
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=7, high=50, name='look_back_length'), Integer(low=10, high=40, name='buy_threshold'), Integer(low=60, high=90, name='sell_threshold'), Integer(low=20, high=35, name='adx_regime_threshold')]

class MACrossoverStrategy(TradingStrategy):
    def generate_signals(self, asset):
        close_np = self.data[f"{asset}_close"].to_numpy()
        short_period, long_period = int(self.params.get('short_window', 50)), int(self.params.get('long_window', 200))
        short_mavg = pd.Series(jit_calculate_ma(close_np, short_period), index=self.data.index)
        long_mavg = pd.Series(jit_calculate_ma(close_np, long_period), index=self.data.index)
        entries = (short_mavg > long_mavg) & (short_mavg.shift(1) <= long_mavg.shift(1))
        exits = (short_mavg < long_mavg) & (short_mavg.shift(1) >= long_mavg.shift(1))
        adx = self.indicators[asset]['adx']
        adx_thresh = self.params.get('adx_regime_threshold', 25)
        entries &= (adx > adx_thresh)
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=10, high=80, name='short_window'), Integer(low=90, high=250, name='long_window'), Integer(low=20, high=35, name='adx_regime_threshold')]

class MACDStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        fast_p, slow_p, signal_p = int(self.params.get('fast_period', 12)), int(self.params.get('slow_period', 26)), int(self.params.get('signal_period', 9))
        for asset in self.assets:
            close_np = self.data[f"{asset}_close"].to_numpy()
            fast_ema = jit_calculate_ema(close_np, fast_p)
            slow_ema = jit_calculate_ema(close_np, slow_p)
            macd_line_np = fast_ema - slow_ema
            signal_line_np = jit_calculate_ema(macd_line_np, signal_p)
            self.indicators[asset]['macd_line'] = pd.Series(macd_line_np, index=self.data.index)
            self.indicators[asset]['signal_line'] = pd.Series(signal_line_np, index=self.data.index)

    def generate_signals(self, asset):
        macd_line = self.indicators[asset]['macd_line']
        signal_line = self.indicators[asset]['signal_line']
        adx = self.indicators[asset]['adx']
        adx_thresh = self.params.get('adx_regime_threshold', 25)
        entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)) & (adx > adx_thresh)
        exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=5, high=50, name='fast_period'), Integer(low=20, high=100, name='slow_period'), Integer(low=5, high=50, name='signal_period'), Integer(low=20, high=35, name='adx_regime_threshold')]

class BollingerBandsStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        look_back, std_dev = int(self.params.get('look_back_length', 20)), float(self.params.get('num_std_dev', 2.0))
        for asset in self.assets:
            close_np = self.data[f"{asset}_close"].to_numpy()
            upper_np, lower_np = jit_calculate_bollinger_bands(close_np, look_back, std_dev)
            self.indicators[asset]['upper_band'] = pd.Series(upper_np, index=self.data.index)
            self.indicators[asset]['lower_band'] = pd.Series(lower_np, index=self.data.index)
    
    def generate_signals(self, asset):
        close = self.data[f"{asset}_close"]
        upper = self.indicators[asset]['upper_band']
        lower = self.indicators[asset]['lower_band']
        adx = self.indicators[asset]['adx']
        adx_thresh = self.params.get('adx_regime_threshold', 23)
        entries = (close < lower) & (adx < adx_thresh)
        exits = (close > upper)
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=10, high=50, name='look_back_length'), Real(low=1.5, high=3.5, name='num_std_dev'), Integer(low=20, high=30, name='adx_regime_threshold')]

class StochasticOscillatorStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        period_k, period_d = int(self.params.get('period_k', 14)), int(self.params.get('period_d', 3))
        for asset in self.assets:
            high_np = self.data[f'{asset}_high'].to_numpy()
            low_np = self.data[f'{asset}_low'].to_numpy()
            close_np = self.data[f'{asset}_close'].to_numpy()
            
            k_line_np, d_line_np = jit_calculate_stochastic(high_np, low_np, close_np, period_k, period_d)

            self.indicators[asset]['%K'] = pd.Series(k_line_np, index=self.data.index)
            self.indicators[asset]['%D'] = pd.Series(d_line_np, index=self.data.index)

    def generate_signals(self, asset):
        k_line, d_line = self.indicators[asset]['%K'], self.indicators[asset]['%D']
        oversold, overbought = self.params.get('oversold_threshold', 20), self.params.get('overbought_threshold', 80)
        adx, adx_thresh = self.indicators[asset]['adx'], self.params.get('adx_regime_threshold', 23)
        entries = (k_line > d_line) & (k_line.shift(1) <= d_line.shift(1)) & (k_line < oversold) & (adx < adx_thresh)
        exits = (k_line < d_line) & (k_line.shift(1) >= d_line.shift(1)) & (k_line > overbought)
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=5, high=50, name='period_k'), Integer(low=2, high=20, name='period_d'), Integer(low=10, high=40, name='oversold_threshold'), Integer(low=60, high=90, name='overbought_threshold'), Integer(low=20, high=30, name='adx_regime_threshold')]

class RateOfChangeStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        for asset in self.assets:
            close_np = self.data[f"{asset}_close"].to_numpy()
            roc_np = jit_calculate_roc(close_np, period=int(self.params.get('look_back_length', 12)))
            self.indicators[asset]['roc'] = pd.Series(roc_np, index=self.data.index)

    def generate_signals(self, asset):
        roc = self.indicators[asset]['roc']
        up_thresh, down_thresh = self.params.get('upward_roc_threshold', 5), self.params.get('downward_roc_threshold', -5)
        adx, adx_thresh = self.indicators[asset]['adx'], self.params.get('adx_regime_threshold', 25)
        entries = (roc > up_thresh) & (adx > adx_thresh)
        exits = (roc < down_thresh)
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=5, high=50, name='look_back_length'), Real(low=1, high=20, name='upward_roc_threshold'), Real(low=-20, high=-1, name='downward_roc_threshold'), Integer(low=20, high=35, name='adx_regime_threshold')]

class OBVStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        for asset in self.assets:
            close_np = self.data[f"{asset}_close"].to_numpy()
            volume_np = self.data[f"{asset}_volume"].fillna(0).to_numpy()
            obv_np = jit_calculate_obv(close_np, volume_np)
            obv_ma_np = jit_calculate_ma(obv_np, period=int(self.params.get('obv_sma_period', 20)))
            self.indicators[asset]['obv'] = pd.Series(obv_np, index=self.data.index)
            self.indicators[asset]['obv_sma'] = pd.Series(obv_ma_np, index=self.data.index)

    def generate_signals(self, asset):
        obv, obv_sma = self.indicators[asset]['obv'], self.indicators[asset]['obv_sma']
        adx, adx_thresh = self.indicators[asset]['adx'], self.params.get('adx_regime_threshold', 25)
        entries = (obv > obv_sma) & (obv.shift(1) <= obv_sma.shift(1)) & (adx > adx_thresh)
        exits = (obv < obv_sma) & (obv.shift(1) >= obv_sma.shift(1))
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=10, high=100, name='obv_sma_period'), Integer(low=20, high=35, name='adx_regime_threshold')]

class MFIStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        mfi_period = int(self.params.get('mfi_period', 14))
        for asset in self.assets:
            high_np = self.data[f'{asset}_high'].to_numpy()
            low_np = self.data[f'{asset}_low'].to_numpy()
            close_np = self.data[f'{asset}_close'].to_numpy()
            volume_np = self.data[f'{asset}_volume'].fillna(0).to_numpy()
            
            mfi_np = jit_calculate_mfi(high_np, low_np, close_np, volume_np, mfi_period)
            self.indicators[asset]['mfi'] = pd.Series(mfi_np, index=self.data.index)

    def generate_signals(self, asset):
        mfi = self.indicators[asset]['mfi']
        oversold, overbought = self.params.get('oversold_threshold', 20), self.params.get('overbought_threshold', 80)
        adx, adx_thresh = self.indicators[asset]['adx'], self.params.get('adx_regime_threshold', 23)
        entries = (mfi < oversold) & (adx < adx_thresh)
        exits = (mfi > overbought)
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=7, high=30, name='mfi_period'), Integer(low=10, high=40, name='oversold_threshold'), Integer(low=60, high=90, name='overbought_threshold'), Integer(low=20, high=30, name='adx_regime_threshold')]

class TMOStrategy(TradingStrategy):
    def _calculate_unique_indicators(self):
        short_p, long_p, signal_p = int(self.params.get('tmo_short_period', 19)), int(self.params.get('tmo_long_period', 39)), int(self.params.get('tmo_signal_period', 9))
        for asset in self.assets:
            momentum_np = self.data[f"{asset}_close"].diff().fillna(0).to_numpy()
            ema_short_np = jit_calculate_ema(momentum_np, short_p)
            ema_long_np = jit_calculate_ema(momentum_np, long_p)
            tmo_line_np = ema_short_np - ema_long_np
            tmo_signal_np = jit_calculate_ema(tmo_line_np, signal_p)
            
            self.indicators[asset]['tmo_line'] = pd.Series(tmo_line_np, index=self.data.index)
            self.indicators[asset]['tmo_signal'] = pd.Series(tmo_signal_np, index=self.data.index)

    def generate_signals(self, asset):
        tmo_line, tmo_signal = self.indicators[asset]['tmo_line'], self.indicators[asset]['tmo_signal']
        adx, adx_thresh = self.indicators[asset]['adx'], self.params.get('adx_regime_threshold', 25)
        entries = (tmo_line > tmo_signal) & (tmo_line.shift(1) <= tmo_signal.shift(1)) & (adx > adx_thresh)
        exits = (tmo_line < tmo_signal) & (tmo_line.shift(1) >= tmo_signal.shift(1))
        return entries, exits

    @staticmethod
    def get_hyperparameter_space(): return [Integer(low=10, high=50, name='tmo_short_period'), Integer(low=30, high=100, name='tmo_long_period'), Integer(low=5, high=30, name='tmo_signal_period'), Integer(low=20, high=35, name='adx_regime_threshold')]

class EnsembleStrategy(TradingStrategy):
    def __init__(self, data, assets, sub_strategies, **kwargs):
        super().__init__(data, assets, **kwargs)
        self.sub_strategies = sub_strategies

    def calculate_indicators(self):
        super().calculate_indicators()
        for strategy in self.sub_strategies:
            strategy.data, strategy.assets = self.data, self.assets
            strategy.calculate_indicators(common_indicators=self.indicators)
    
    def generate_signals(self, asset):
        entry_signals = pd.Series(False, index=self.data.index)
        exit_signals = pd.Series(False, index=self.data.index)
        for strategy in self.sub_strategies:
            entries, exits = strategy.generate_signals(asset)
            entry_signals |= entries
            exit_signals |= exits
        return entry_signals, exits

    @staticmethod
    def get_hyperparameter_space(): return []

# ==============================================================================
# 2. HYBRID BACKTESTING ENGINE (With JIT Event Loop)
# ==============================================================================

# --- MODIFIED: Removed SL/TP and ATR logic, changed risk param ---
@jit(nopython=True)
def jit_event_loop(
    close_prices_np,  # 2D Array (n_steps, n_assets)
    entry_signals_np, # 2D Array (n_steps, n_assets)
    exit_signals_np,  # 2D Array (n_steps, n_assets)
    initial_cash,
    position_size_pct # NEW: Replaces risk_per_trade_pct
):
    """
    Numba-compiled event-driven backtesting loop.
    - REMOVED: Stop-loss and Take-profit logic.
    - CHANGED: Position sizing is now fixed fractional.
    """
    n_steps, n_assets = close_prices_np.shape
    
    cash = initial_cash
    portfolio_history = np.zeros(n_steps)
    
    # [0] = quantity, [1] = entry_price
    positions_np = np.zeros((n_assets, 2)) 

    for i in range(n_steps):
        # --- 1. Calculate current portfolio value ---
        current_value = cash
        for asset_idx in range(n_assets):
            current_value += positions_np[asset_idx, 0] * close_prices_np[i, asset_idx]
        portfolio_history[i] = current_value
        
        # --- 2. Check for exits (Signal-Based Only) ---
        for asset_idx in range(n_assets):
            quantity = positions_np[asset_idx, 0]
            
            if quantity > 0:
                price = close_prices_np[i, asset_idx]
                
                # Check for invalid price data
                if np.isnan(price) or price <= 0:
                    continue
                
                # Exit condition (ONLY from signal)
                if exit_signals_np[i, asset_idx]:
                    cash += quantity * price * 0.999 # 0.1% commission
                    positions_np[asset_idx, 0] = 0.0
                    positions_np[asset_idx, 1] = 0.0
        
        # --- 3. Check for new entries ---
        for asset_idx in range(n_assets):
            # If no position and entry signal is true
            if positions_np[asset_idx, 0] == 0 and entry_signals_np[i, asset_idx]:
                price = close_prices_np[i, asset_idx]
                
                # Check for valid data to place a trade
                if np.isnan(price) or price <= 0:
                    continue
                
                # --- NEW: Fixed Fractional Position Sizing ---
                notional = current_value * position_size_pct
                
                if 0 < notional < cash:
                    quantity = (notional / price) * 0.999 # 0.1% commission
                    cash -= notional
                    positions_np[asset_idx, 0] = quantity
                    positions_np[asset_idx, 1] = price
                            
    return portfolio_history


class Backtester:
    def __init__(self, data, strategy_class, strategy_params, risk_params, portfolio_params):
        self.data = data.copy()
        # This asset detection logic is robust for the new data format
        all_parts = [c.rsplit('_', 1) for c in self.data.columns]
        # --- CRITICAL FIX: Make asset detection stricter ---
        valid_suffixes = {'close', 'high', 'low', 'open', 'volume'}
        self.assets = sorted(list(set(parts[0] for parts in all_parts if len(parts) == 2 and parts[1] in valid_suffixes)))
        # --- END FIX ---
        
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.risk_params = risk_params
        self.portfolio_params = portfolio_params
        self.min_data_len = 250

    def _analyze_performance(self, portfolio_history):
        if isinstance(portfolio_history, pd.Series):
            history_df = pd.DataFrame({'value': portfolio_history})
        else:
            if len(portfolio_history) == 0:
                return {'final_portfolio_value': 0, 'returns': -1, 'sharpe_ratio': -100, 'max_drawdown': -1, 'calmar_ratio': -100}
            history_df = pd.DataFrame({'value': portfolio_history}, index=self.data.index)

        initial_cash = self.portfolio_params.get('initial_cash', 1000)
        
        if history_df['value'].sum() == 0:
             return {'final_portfolio_value': 0, 'returns': -1, 'sharpe_ratio': -100, 'max_drawdown': -1, 'calmar_ratio': -100}
             
        final_value = history_df['value'].iloc[-1]
        total_return = (final_value / initial_cash) - 1
        returns = history_df['value'].pct_change().fillna(0)
        
        std_dev = np.std(returns)
        sharpe = np.mean(returns) / std_dev * np.sqrt(252 * 24) if std_dev != 0 else 0
        
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_drawdown = drawdown.min()
        
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 and total_return > 0 else 0
        
        return {'final_portfolio_value': final_value, 'returns': total_return, 'sharpe_ratio': sharpe, 'max_drawdown': max_drawdown, 'calmar_ratio': calmar}

    def run(self, mode='event_driven'):
        if len(self.data) < self.min_data_len:
            return {'final_portfolio_value': self.portfolio_params.get('initial_cash', 1000), 'returns': -1, 'sharpe_ratio': -100, 'max_drawdown': -1, 'calmar_ratio': -100}
        
        if isinstance(self.strategy_class, type):
            strategy = self.strategy_class(self.data, self.assets, **self.strategy_params)
        else:
            strategy = self.strategy_class
            strategy.data, strategy.assets = self.data, self.assets
            
        strategy.calculate_indicators()
        
        entry_signals = {asset: strategy.generate_signals(asset)[0].shift(1).fillna(False) for asset in self.assets}
        exit_signals = {asset: strategy.generate_signals(asset)[1].shift(1).fillna(False) for asset in self.assets}
        
        if mode == 'vectorized':
            return self._run_vectorized(entry_signals, exit_signals)
        else:
            return self._run_event_driven(strategy, entry_signals, exit_signals)

    def _run_vectorized(self, entry_signals, exit_signals):
        initial_cash = self.portfolio_params.get('initial_cash', 1000)
        returns_df = pd.DataFrame(index=self.data.index)
        
        for asset in self.assets:
            price = self.data[f'{asset}_close']
            positions = pd.Series(np.nan, index=self.data.index)
            positions[entry_signals[asset]] = 1
            positions[exit_signals[asset]] = 0
            positions = positions.ffill().fillna(0)
            
            strategy_returns = positions.shift(1) * price.pct_change()
            returns_df[asset] = strategy_returns
            
        portfolio_returns = returns_df.mean(axis=1).fillna(0)
        portfolio_history = initial_cash * (1 + portfolio_returns).cumprod()
        return self._analyze_performance(portfolio_history)

    def _run_event_driven(self, strategy, entry_signals, exit_signals):
        asset_list = self.assets
        
        close_arrays = [self.data[f'{asset}_close'].to_numpy() for asset in asset_list]
        entry_arrays = [entry_signals[asset].to_numpy() for asset in asset_list]
        exit_arrays = [exit_signals[asset].to_numpy() for asset in asset_list]
        
        # --- REMOVED: ATR arrays are no longer needed ---
        # atr_arrays = [
        #     strategy.indicators.get(asset, {}).get('atr', pd.Series(0)).fillna(0).to_numpy() 
        #     for asset in asset_list
        # ]

        close_prices_np = np.stack(close_arrays, axis=1)
        entry_signals_np = np.stack(entry_arrays, axis=1)
        exit_signals_np = np.stack(exit_arrays, axis=1)
        # atr_np = np.stack(atr_arrays, axis=1) # REMOVED

        initial_cash = self.portfolio_params.get('initial_cash', 1000)
        
        # --- CHANGED: Use new position_size_pct parameter ---
        position_pct = self.risk_params.get('position_size_pct', 0.05) # Default to 5% of portfolio

        portfolio_history_np = jit_event_loop(
            close_prices_np, entry_signals_np, exit_signals_np,
            initial_cash, position_pct
        )
        
        return self._analyze_performance(portfolio_history_np)

# ==============================================================================
# 3. PANDAS-BASED INDICATORS (Used for pre-calculation in opt script)
# ==============================================================================

def calculate_atr(high, low, close, period=14):
    """
    Pandas-based ATR calculation. (Still needed for ADX)
    """
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean() # Using SMA for ATR

def calculate_adx(high, low, close, period=14):
    """
    Pandas-based ADX calculation.
    """
    atr = calculate_atr(high, low, close, period=period)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[(plus_dm < 0) | (plus_dm <= abs(minus_dm))] = 0
    minus_dm[(minus_dm > 0) | (abs(minus_dm) <= plus_dm)] = 0
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-6))
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/period, adjust=False).mean()) / atr.replace(0, 1e-6))
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-6))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(25)

