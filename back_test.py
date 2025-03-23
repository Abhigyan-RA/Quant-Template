import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from untrade.client import Client

# -------------------------------
# Dynamic Adaptive Trader Class
# -------------------------------
class DynamicAdaptiveTrader:
    def __init__(self, data_path, test_percentage=0.4, initial_risk_tolerance=0.5):
        """
        Initialize the adaptive trading algorithm.
        
        Parameters:
        - data_path (str): Path to the price data CSV file
        - test_percentage (float): Percentage of data to use for testing (0.0-1.0)
        - initial_risk_tolerance (float): Initial risk tolerance factor (0.0-1.0)
        """
        self.data_path = data_path
        self.test_percentage = test_percentage
        self.risk_tolerance = initial_risk_tolerance
        self.memory_window = 20  # Number of past decisions to remember
        self.penalty_weights = {}  # Will store penalties for different indicators
        self.opportunity_memory = []  # Store missed opportunities
        self.decision_memory = []  # Store past decisions and outcomes
        self.df = None
        
    def load_data(self):
        """Load and prepare price data."""
        df = pd.read_csv(self.data_path)
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        self._calculate_technical_indicators(df)
        self.df = df
        return df
    
    def _calculate_technical_indicators(self, df):
        """Calculate technical indicators with dynamic lookback periods."""
        periods = {
            'rsi': 14,
            'so': 14,
            'ma_short': 10,
            'ma_long': 30,
            'volatility': 20
        }
        df['rsi'] = self._calculate_rsi(df, periods['rsi'])
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, periods['so'])
        df['ma_short'] = df['close'].rolling(window=periods['ma_short']).mean()
        df['ma_long'] = df['close'].rolling(window=periods['ma_long']).mean()
        df['bb_middle'] = df['close'].rolling(window=periods['volatility']).mean()
        df['bb_std'] = df['close'].rolling(window=periods['volatility']).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['daily_range'] = (df['high'] - df['low']) / df['low']
        df['volatility'] = df['daily_range'].rolling(window=periods['volatility']).std()
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['momentum'] = df['close'].pct_change(periods=periods['ma_short'])
        for col in ['rsi', 'stoch_k', 'stoch_d', 'ma_short', 'ma_long', 'momentum', 'volume_ratio']:
            self.penalty_weights[col] = 1.0
        return df
    
    def _calculate_rsi(self, df, period=14, epsilon=1e-10):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
        rs = gain / (loss + epsilon)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df, period=14, d_period=3, epsilon=1e-10):
        high_roll = df['high'].rolling(window=period).max()
        low_roll = df['low'].rolling(window=period).min()
        denom = (high_roll - low_roll).replace(0, epsilon)
        k = 100 * (df['close'] - low_roll) / denom
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def _adapt_parameters(self):
        if len(self.decision_memory) < 10:
            return
        recent_decisions = self.decision_memory[-10:]
        correct_decisions = sum(1 for decision in recent_decisions if decision['correct'])
        success_rate = correct_decisions / len(recent_decisions)
        if success_rate > 0.6:
            self.risk_tolerance = min(0.8, self.risk_tolerance * 1.1)
        elif success_rate < 0.4:
            self.risk_tolerance = max(0.2, self.risk_tolerance * 0.9)
        indicator_success = {}
        for indicator in self.penalty_weights.keys():
            correct_with_indicator = 0
            total_with_indicator = 0
            for decision in recent_decisions:
                if decision['indicators'][indicator]:
                    total_with_indicator += 1
                    if decision['correct']:
                        correct_with_indicator += 1
            if total_with_indicator > 0:
                indicator_success[indicator] = correct_with_indicator / total_with_indicator
            else:
                indicator_success[indicator] = 0.5
        for indicator, success in indicator_success.items():
            adjustment = (success - 0.5) * 0.2
            self.penalty_weights[indicator] = max(0.1, min(2.0, self.penalty_weights[indicator] + adjustment))
    
    def _assess_missed_opportunities(self, current_idx, window=10):
        if current_idx < window:
            return
        for i in range(current_idx - window, current_idx):
            price_change = (self.df.iloc[i+1]['close'] - self.df.iloc[i]['close']) / self.df.iloc[i]['close']
            if abs(price_change) > 0.01 and not any(decision['index'] == i for decision in self.decision_memory):
                opportunity = {
                    'index': i,
                    'date': self.df.index[i],
                    'price_change': price_change,
                    'indicators': {k: self.df.iloc[i][k] for k in self.penalty_weights.keys() if k in self.df.columns}
                }
                self.opportunity_memory.append(opportunity)
                if len(self.opportunity_memory) > 50:
                    self.opportunity_memory.pop(0)
    
    def _trade_decision(self, idx):
        if idx < 30:
            return 0, {}
        row = self.df.iloc[idx]
        prev_row = self.df.iloc[idx-1]
        signal_strength = 0
        used_indicators = {}
        # RSI signal
        rsi_signal = 0
        if row['rsi'] < 30:
            rsi_signal = 1
        elif row['rsi'] > 70:
            rsi_signal = -1
        signal_strength += rsi_signal * self.penalty_weights['rsi']
        used_indicators['rsi'] = (rsi_signal != 0)
        # Stochastic signal
        stoch_signal = 0
        if row['stoch_k'] < 20 and row['stoch_d'] < 20:
            stoch_signal = 1
        elif row['stoch_k'] > 80 and row['stoch_d'] > 80:
            stoch_signal = -1
        signal_strength += stoch_signal * self.penalty_weights['stoch_k']
        used_indicators['stoch_k'] = (stoch_signal != 0)
        used_indicators['stoch_d'] = (stoch_signal != 0)
        # Moving averages signal
        ma_signal = 0
        if row['ma_short'] > row['ma_long'] and prev_row['ma_short'] <= prev_row['ma_long']:
            ma_signal = 1
        elif row['ma_short'] < row['ma_long'] and prev_row['ma_short'] >= prev_row['ma_long']:
            ma_signal = -1
        signal_strength += ma_signal * ((self.penalty_weights['ma_short'] + self.penalty_weights['ma_long']) / 2)
        used_indicators['ma_short'] = (ma_signal != 0)
        used_indicators['ma_long'] = (ma_signal != 0)
        # Momentum signal
        momentum_signal = 0
        if row['momentum'] > 0.02:
            momentum_signal = 1
        elif row['momentum'] < -0.02:
            momentum_signal = -1
        signal_strength += momentum_signal * self.penalty_weights['momentum']
        used_indicators['momentum'] = (momentum_signal != 0)
        # Volume signal
        volume_signal = 0
        if row['volume_ratio'] > 1.5 and row['close'] > prev_row['close']:
            volume_signal = 1
        elif row['volume_ratio'] > 1.5 and row['close'] < prev_row['close']:
            volume_signal = -1
        signal_strength += volume_signal * self.penalty_weights['volume_ratio']
        used_indicators['volume_ratio'] = (volume_signal != 0)
        if signal_strength > self.risk_tolerance:
            return 1, used_indicators
        elif signal_strength < -self.risk_tolerance:
            return -1, used_indicators
        else:
            return 0, used_indicators
    
    def _evaluate_decision(self, idx, signal, used_indicators):
        if idx >= len(self.df) - 1:
            return
        future_price = self.df.iloc[idx + 1]['close']
        current_price = self.df.iloc[idx]['close']
        price_change = (future_price - current_price) / current_price
        correct = False
        if (signal == 1 and price_change > 0) or (signal == -1 and price_change < 0):
            correct = True
        elif signal == 0 and abs(price_change) < 0.005:
            correct = True
        decision = {
            'index': idx,
            'date': self.df.index[idx],
            'signal': signal,
            'price_change': price_change,
            'correct': correct,
            'indicators': used_indicators
        }
        self.decision_memory.append(decision)
        if len(self.decision_memory) > self.memory_window:
            self.decision_memory.pop(0)
        if not correct and signal != 0:
            for indicator, used in used_indicators.items():
                if used and indicator in self.penalty_weights:
                    self.penalty_weights[indicator] = max(0.1, self.penalty_weights[indicator] * 0.95)
    
    def generate_signals(self):
        if self.df is None:
            self.load_data()
        clean_df = self.df.dropna()
        split_idx = int(len(clean_df) * (1 - self.test_percentage))
        test_df = clean_df.iloc[split_idx:].copy()
        signals = []
        test_indices = list(range(len(test_df)))
        for i in test_indices:
            actual_idx = i + split_idx
            signal, used_indicators = self._trade_decision(actual_idx)
            signals.append(signal)
            self._evaluate_decision(actual_idx, signal, used_indicators)
            self._assess_missed_opportunities(actual_idx)
            if i % 10 == 0 and i > 0:
                self._adapt_parameters()
        test_df['Signal'] = signals
        # Convert 0 signals to NaN for clarity in backtest output
        test_df.loc[test_df['Signal'] == 0, 'Signal'] = np.nan
        return test_df
    
    def save_predictions_to_csv(self, output_path="adaptive_signals.csv"):
        test_df = self.generate_signals()
        signal = []
        prev = None
        for value in test_df['Signal']:
            if pd.isna(value):
                signal.append(0)
            elif value == prev:
                signal.append(0)
            else:
                signal.append(value)
            prev = value
        output_df = test_df.copy()
        output_df['signals'] = signal
        output_df['Short_MA'] = output_df['ma_short']
        output_df['Long_MA'] = output_df['ma_long']
        output_df = output_df.reset_index()
        output_df.to_csv(output_path, index=False)
        return output_df
    
    def run_untrade_backtest(self, output_csv_path="adaptive_signals.csv"):
        client = Client()
        result = client.backtest(
            file_path=output_csv_path,
            leverage=1
        )
        return result
    
    def run_full_pipeline(self, output_csv_path="adaptive_signals.csv"):
        print("Loading and preparing data...")
        self.load_data()
        print("Generating adaptive trading signals...")
        self.save_predictions_to_csv(output_csv_path)
        print(f"Signals saved to {output_csv_path}")
        print("Running backtest using untrade...")
        backtest_result = self.run_untrade_backtest(output_csv_path)
        last_value = None
        for value in backtest_result:
            last_value = value
        print("Backtest complete.")
        print("Backtest results:", last_value)
        print("\nAlgorithm Adaptation Insights:")
        print(f"Final Risk Tolerance: {self.risk_tolerance:.2f}")
        print("Final Indicator Weights:")
        for indicator, weight in self.penalty_weights.items():
            print(f"  - {indicator}: {weight:.2f}")
        if len(self.decision_memory) > 0:
            correct_decisions = sum(1 for d in self.decision_memory if d['correct'])
            accuracy = correct_decisions / len(self.decision_memory)
            print(f"Recent Decision Accuracy: {accuracy:.2f}")
        return last_value

# -------------------------------
# Blended Crypto Trader Class
# -------------------------------
class BlendedCryptoTrader:
    def __init__(self, data_path, test_percentage=0.4):
        """
        Advanced trading algorithm blending multiple proven BTC/USD strategies.
        Parameters:
        - data_path (str): Path to the price data CSV file.
        - test_percentage (float): Percentage of data to use for testing (0.0-1.0).
        """
        self.data_path = data_path
        self.test_percentage = test_percentage
        self.df = None
        self.strategy_weights = {
            'trend_following': 1.0,
            'mean_reversion': 1.0,
            'volume_based': 1.0,
            'volatility_based': 1.0,
            'whale_activity': 1.0
        }
        # Separate weights for long and short signals
        self.long_weights = {
            'macd': 1.2,
            'ema_cross': 1.3,
            'bull_div': 1.5,
            'support_bounce': 1.4,
            'volume_surge': 1.2
        }
        self.short_weights = {
            'bearish_div': 1.5,
            'resistance_reject': 1.4,
            'volatility_expansion': 1.3,
            'volume_climax': 1.2,
            'supertrend': 1.4
        }
        
    def load_data(self):
        """Load and prepare price data."""
        df = pd.read_csv(self.data_path)
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        self._calculate_indicators(df)
        self.df = df
        return df
    
    def _calculate_indicators(self, df):
        """Calculate comprehensive set of technical indicators."""
        # Trend Following Indicators
        for period in [8, 13, 21, 34, 55, 89, 144, 200]:
            df[f'ema_{period}'] = self._calculate_ema(df['close'], period)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        df['adx'] = self._calculate_adx(df, 14)
        df['supertrend'], df['supertrend_direction'] = self._calculate_supertrend(df, 10, 3)
        self._calculate_ichimoku(df)
        
        # Mean Reversion Indicators
        for period in [6, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        for period in [20, 50]:
            middle, upper, lower = self._calculate_bollinger_bands(df['close'], period)
            df[f'bb_middle_{period}'] = middle
            df[f'bb_upper_{period}'] = upper
            df[f'bb_lower_{period}'] = lower
            df[f'bb_pct_{period}'] = (df['close'] - lower) / (upper - lower)
        
        # Volume-Based Indicators
        df['obv'] = self._calculate_obv(df)
        for period in [10, 20, 50]:
            df[f'volume_ema_{period}'] = self._calculate_ema(df['volume'], period)
        df['volume_rsi'] = self._calculate_rsi(df['volume'], 14)
        df['cmf'] = self._calculate_cmf(df, 20)
        df['vzo'] = self._calculate_vzo(df, 14, 28)
        
        # Volatility-Based Indicators
        for period in [5, 14, 21]:
            df[f'atr_{period}'] = self._calculate_atr(df, period)
        self._calculate_keltner_channels(df, 20, 2)
        df['historical_vol'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)
        
        # Pattern Recognition
        self._identify_support_resistance(df, window=10)
        self._calculate_divergences(df)
        
        # Combined Signals
        df['hma_50'] = self._calculate_hull_ma(df['close'], 50)
        self._calculate_squeeze_momentum(df)
        self._calculate_elder_ray(df)
        self._calculate_wavetrend(df)
        
        return df
    
    # --- Technical indicator methods (same as before) ---
    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        fast_ema = self._calculate_ema(series, fast)
        slow_ema = self._calculate_ema(series, slow)
        macd = fast_ema - slow_ema
        macd_signal = self._calculate_ema(macd, signal)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_adx(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_dm = high - high.shift()
        minus_dm = low.shift() - low
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(period).mean()
        return adx
    
    def _calculate_supertrend(self, df, period=10, multiplier=3):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        upperband = (high + low) / 2 + multiplier * atr
        lowerband = (high + low) / 2 - multiplier * atr
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if close.iloc[i] > upperband.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lowerband.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                    lowerband.iloc[i] = lowerband.iloc[i-1]
                if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                    upperband.iloc[i] = upperband.iloc[i-1]
            supertrend.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]
        return supertrend, direction
    
    def _calculate_ichimoku(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        nine_period_high = high.rolling(window=9).max()
        nine_period_low = low.rolling(window=9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        df['chikou_span'] = close.shift(-26)
    
    def _calculate_rsi(self, series, period=14, epsilon=1e-10):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=period).mean()
        avg_loss = down.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + epsilon)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df, k_period=14, d_period=3, epsilon=1e-10):
        high_roll = df['high'].rolling(window=k_period).max()
        low_roll = df['low'].rolling(window=k_period).min()
        denom = (high_roll - low_roll).replace(0, epsilon)
        k = 100 * (df['close'] - low_roll) / denom
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def _calculate_bollinger_bands(self, series, period=20, num_std=2):
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return middle, upper, lower
    
    def _calculate_obv(self, df):
        obv = pd.Series(index=df.index)
        obv.iloc[0] = df['volume'].iloc[0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def _calculate_cmf(self, df, period=20):
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 1)
        money_flow_volume = money_flow_multiplier * df['volume']
        cmf = money_flow_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
        return cmf
    
    def _calculate_vzo(self, df, short_period=14, long_period=28):
        short_vol_ema = self._calculate_ema(df['volume'], short_period)
        long_vol_ema = self._calculate_ema(df['volume'], long_period)
        vzo = 100 * (short_vol_ema - long_vol_ema) / long_vol_ema
        return vzo
    
    def _calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_keltner_channels(self, df, period=20, multiplier=2):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        ema = self._calculate_ema(typical_price, period)
        atr = self._calculate_atr(df, period)
        df['keltner_middle'] = ema
        df['keltner_upper'] = ema + (multiplier * atr)
        df['keltner_lower'] = ema - (multiplier * atr)
    
    def _identify_support_resistance(self, df, window=10):
        df['potential_resistance'] = df['high'].rolling(window=window, center=True).max()
        df['potential_support'] = df['low'].rolling(window=window, center=True).min()
        resistance_threshold = 0.005
        support_threshold = 0.005
        df['near_resistance'] = (df['high'] >= df['potential_resistance'] * (1 - resistance_threshold)) & \
                                (df['high'] <= df['potential_resistance'] * (1 + resistance_threshold))
        df['near_support'] = (df['low'] <= df['potential_support'] * (1 + support_threshold)) & \
                             (df['low'] >= df['potential_support'] * (1 - support_threshold))
        df['support_bounce'] = (df['near_support']) & (df['close'] > df['open']) & (df['low'] < df['open'])
        df['resistance_reject'] = (df['near_resistance']) & (df['close'] < df['open']) & (df['high'] > df['open'])
    
    def _calculate_divergences(self, df):
        rsi = df['rsi_14']
        close = df['close']
        def find_local_extrema(series, window=5):
            maxima = (series.rolling(window=window, center=True).max() == series) & (series != series.shift(1))
            minima = (series.rolling(window=window, center=True).min() == series) & (series != series.shift(1))
            return maxima, minima
        price_maxima, price_minima = find_local_extrema(close)
        rsi_maxima, rsi_minima = find_local_extrema(rsi)
        df['bull_div'] = False
        df['bear_div'] = False
        for i in range(2, len(df)):
            if price_minima.iloc[i]:
                for j in range(i-1, max(0, i-10), -1):
                    if price_minima.iloc[j]:
                        if (close.iloc[i] < close.iloc[j]) and (rsi.iloc[i] > rsi.iloc[j]):
                            df['bull_div'].iloc[i] = True
                        break
        for i in range(2, len(df)):
            if price_maxima.iloc[i]:
                for j in range(i-1, max(0, i-10), -1):
                    if price_maxima.iloc[j]:
                        if (close.iloc[i] > close.iloc[j]) and (rsi.iloc[i] < rsi.iloc[j]):
                            df['bear_div'].iloc[i] = True
                        break
    
    def _calculate_hull_ma(self, series, period):
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        wma1 = series.rolling(window=half_period).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)))
        wma2 = series.rolling(window=period).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)))
        hull = 2 * wma1 - wma2
        hma = hull.rolling(window=sqrt_period).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)))
        return hma
    
    def _calculate_squeeze_momentum(self, df):
        basis = df['close'].rolling(window=20).mean()
        dev = 2 * df['close'].rolling(window=20).std()
        df['upper_bb'] = basis + dev
        df['lower_bb'] = basis - dev
        atr = self._calculate_atr(df, 20)
        df['upper_kc'] = basis + (1.5 * atr)
        df['lower_kc'] = basis - (1.5 * atr)
        df['squeeze_on'] = (df['lower_bb'] > df['lower_kc']) & (df['upper_bb'] < df['upper_kc'])
        highest = df['high'].rolling(window=20).max()
        lowest = df['low'].rolling(window=20).min()
        mid = (highest + lowest) / 2
        df['squeeze_mom'] = (df['close'] - mid) / df['close']
    
    def _calculate_elder_ray(self, df):
        ema13 = self._calculate_ema(df['close'], 13)
        df['bull_power'] = df['high'] - ema13
        df['bear_power'] = df['low'] - ema13
    
    def _calculate_wavetrend(self, df, channel_length=10, avg_length=21):
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        esa = self._calculate_ema(hlc3, channel_length)
        d = self._calculate_ema(abs(hlc3 - esa), channel_length)
        d = d.replace(0, 0.0001)
        ci = (hlc3 - esa) / (0.015 * d)
        df['wt1'] = self._calculate_ema(ci, avg_length)
        df['wt2'] = self._calculate_ema(df['wt1'], 4)
    
    def generate_signals(self):
        if self.df is None:
            self.load_data()
        clean_df = self.df.dropna()
        split_idx = int(len(clean_df) * (1 - self.test_percentage))
        test_df = clean_df.iloc[split_idx:].copy()
        test_df['long_signal'] = self._calculate_long_signals(test_df)
        test_df['short_signal'] = self._calculate_short_signals(test_df)
        test_df['Signal'] = np.where(
            test_df['long_signal'] > abs(test_df['short_signal']), 
            1, 
            np.where(abs(test_df['short_signal']) > test_df['long_signal'], -1, 0)
        )
        weak_signal_threshold = 0.3
        test_df.loc[(test_df['long_signal'] < weak_signal_threshold) & 
                   (abs(test_df['short_signal']) < weak_signal_threshold), 'Signal'] = 0
        test_df = self._filter_signals(test_df)
        return test_df
    
    def _calculate_long_signals(self, df):
        long_signals = pd.Series(0.0, index=df.index)
        macd_signal = ((df['macd'] > df['macd_signal']) & 
                       (df['macd'].shift(1) <= df['macd_signal'].shift(1))) * self.long_weights['macd']
        ema_cross = ((df['ema_8'] > df['ema_21']) & 
                     (df['ema_8'].shift(1) <= df['ema_21'].shift(1))) * self.long_weights['ema_cross']
        bull_div = df['bull_div'] * self.long_weights['bull_div']
        support_bounce = df['support_bounce'] * self.long_weights['support_bounce']
        volume_surge = ((df['volume'] > df['volume_ema_20'] * 1.5) & 
                        (df['close'] > df['open'])) * self.long_weights['volume_surge']
        oversold_bounce = ((df['rsi_14'] < 30) & (df['rsi_14'] > df['rsi_14'].shift(1))) * 1.0
        trend_confirm = ((df['close'] > df['ema_55']) & 
                         (df['ema_8'] > df['ema_8'].shift(5))) * 0.8
        supertrend_buy = (df['supertrend_direction'] == 1) & (df['supertrend_direction'].shift(1) == -1) * 1.2
        ichimoku_bull = ((df['close'] > df['senkou_span_a']) & 
                         (df['close'] > df['senkou_span_b']) &
                         (df['tenkan_sen'] > df['kijun_sen'])) * 0.9
        squeeze_bull = ((df['squeeze_on'].shift(1) == True) & 
                        (df['squeeze_on'] == False) & 
                        (df['squeeze_mom'] > 0)) * 1.1
        wavetrend_bull = ((df['wt1'] > df['wt2']) & 
                          (df['wt1'].shift(1) <= df['wt2'].shift(1)) & 
                          (df['wt1'] < -60)) * 1.2
        long_signals = (macd_signal + ema_cross + bull_div + support_bounce + volume_surge + 
                        oversold_bounce + trend_confirm + supertrend_buy + ichimoku_bull + 
                        squeeze_bull + wavetrend_bull)
        max_possible = sum([w for w in self.long_weights.values()]) + 5.2
        long_signals = long_signals / max_possible
        return long_signals
    
    def _calculate_short_signals(self, df):
        short_signals = pd.Series(0.0, index=df.index)
        bearish_div = df['bear_div'] * self.short_weights['bearish_div']
        resistance_reject = df['resistance_reject'] * self.short_weights['resistance_reject']
        volatility_expansion = ((df['atr_14'] > df['atr_14'].rolling(10).mean() * 1.5) & 
                               (df['close'] < df['open'])) * self.short_weights['volatility_expansion']
        volume_climax = ((df['volume'] > df['volume_ema_20'] * 2.0) & 
                         (df['close'] < df['open']) & 
                         df['near_resistance']) * self.short_weights['volume_climax']
        supertrend_sell = ((df['supertrend_direction'] == -1) & 
                           (df['supertrend_direction'].shift(1) == 1)) * self.short_weights['supertrend']
        death_cross = ((df['ema_55'] < df['ema_200']) & 
                       (df['ema_55'].shift(1) >= df['ema_200'].shift(1))) * 1.3
        overbought_reversal = ((df['rsi_14'] > 70) & 
                               (df['rsi_14'] < df['rsi_14'].shift(1))) * 1.0
        support_break = ((df['close'] < df['ema_200']) & 
                         (df['close'].shift(1) >= df['ema_200'].shift(1))) * 1.1
        ichimoku_bear = ((df['close'] < df['senkou_span_a']) & 
                         (df['close'] < df['senkou_span_b']) &
                         (df['tenkan_sen'] < df['kijun_sen'])) * 0.9
        squeeze_bear = ((df['squeeze_on'].shift(1) == True) & 
                        (df['squeeze_on'] == False) & 
                        (df['squeeze_mom'] < 0)) * 1.1
        wavetrend_bear = ((df['wt1'] < df['wt2']) & 
                          (df['wt1'].shift(1) >= df['wt2'].shift(1)) & 
                          (df['wt1'] > 60)) * 1.2
        short_signals = -1 * (bearish_div + resistance_reject + volatility_expansion + 
                              volume_climax + supertrend_sell + death_cross + 
                              overbought_reversal + support_break + ichimoku_bear + 
                              squeeze_bear + wavetrend_bear)
        max_possible = sum([w for w in self.short_weights.values()]) + 5.5
        short_signals = short_signals / max_possible
        return short_signals
    
    def _filter_signals(self, df):
        filtered_df = df.copy()
        min_hold = 3
        last_signal = 0
        periods_held = 0
        for i in range(len(filtered_df)):
            current_signal = filtered_df['Signal'].iloc[i]
            if current_signal != last_signal and periods_held < min_hold and last_signal != 0:
                filtered_df['Signal'].iloc[i] = last_signal
            else:
                if current_signal != last_signal:
                    periods_held = 0
                    last_signal = current_signal
                else:
                    periods_held += 1
        filtered_df['long_strength'] = filtered_df['long_signal']
        filtered_df['short_strength'] = filtered_df['short_signal']
        return filtered_df
    
    def backtest(self, initial_capital=10000, position_size=0.95, disable_shorts=False):
        signals_df = self.generate_signals()
        signals_df['Position'] = 0
        signals_df['BTC_Holdings'] = 0.0
        signals_df['USD_Holdings'] = initial_capital
        signals_df['Equity'] = initial_capital
        signals_df['Returns'] = 0.0
        signals_df['Cumulative_Returns'] = 1.0
        signals_df['Drawdown'] = 0.0
        trades = []
        current_position = 0
        entry_price = 0
        for i in range(1, len(signals_df)):
            date = signals_df.index[i]
            price = signals_df['close'].iloc[i]
            signal = signals_df['Signal'].iloc[i]
            prev_equity = signals_df['Equity'].iloc[i-1]
            prev_btc = signals_df['BTC_Holdings'].iloc[i-1]
            prev_usd = signals_df['USD_Holdings'].iloc[i-1]
            signals_df['BTC_Holdings'].iloc[i] = prev_btc
            signals_df['USD_Holdings'].iloc[i] = prev_usd
            if signal == 1 and current_position <= 0:
                if current_position < 0:
                    profit_loss = entry_price - price
                    signals_df['USD_Holdings'].iloc[i] = prev_usd + (abs(prev_btc) * price) + (abs(prev_btc) * profit_loss)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': 'short',
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss / entry_price
                    })
                entry_price = price
                entry_date = date
                available_capital = signals_df['USD_Holdings'].iloc[i]
                btc_to_buy = (available_capital * position_size) / price
                signals_df['BTC_Holdings'].iloc[i] = btc_to_buy
                signals_df['USD_Holdings'].iloc[i] = available_capital - (btc_to_buy * price)
                current_position = 1
            elif signal == -1 and current_position >= 0 and not disable_shorts:
                if current_position > 0:
                    profit_loss = price - entry_price
                    signals_df['USD_Holdings'].iloc[i] = prev_usd + (prev_btc * price)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': 'long',
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss / entry_price
                    })
                entry_price = price
                entry_date = date
                available_capital = signals_df['USD_Holdings'].iloc[i]
                btc_to_short = (available_capital * position_size) / price
                signals_df['BTC_Holdings'].iloc[i] = -btc_to_short
                signals_df['USD_Holdings'].iloc[i] = available_capital + (btc_to_short * price)
                current_position = -1
            elif signal == 0 and current_position != 0:
                if current_position > 0:
                    profit_loss = price - entry_price
                    signals_df['USD_Holdings'].iloc[i] = prev_usd + (prev_btc * price)
                    signals_df['BTC_Holdings'].iloc[i] = 0
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': 'long',
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss / entry_price
                    })
                elif current_position < 0:
                    profit_loss = entry_price - price
                    signals_df['USD_Holdings'].iloc[i] = prev_usd + (abs(prev_btc) * price) + (abs(prev_btc) * profit_loss)
                    signals_df['BTC_Holdings'].iloc[i] = 0
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'type': 'short',
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss / entry_price
                    })
                current_position = 0
            btc_value = signals_df['BTC_Holdings'].iloc[i] * price
            signals_df['Equity'].iloc[i] = signals_df['USD_Holdings'].iloc[i] + abs(btc_value)
            signals_df['Returns'].iloc[i] = signals_df['Equity'].iloc[i] / prev_equity - 1
            signals_df['Cumulative_Returns'].iloc[i] = signals_df['Cumulative_Returns'].iloc[i-1] * (1 + signals_df['Returns'].iloc[i])
            peak = signals_df['Equity'].iloc[:i+1].max()
            signals_df['Drawdown'].iloc[i] = (signals_df['Equity'].iloc[i] - peak) / peak
        performance = self._calculate_performance_metrics(signals_df, trades)
        return signals_df, trades, performance
    
    def _calculate_performance_metrics(self, df, trades):
        total_return = df['Equity'].iloc[-1] / df['Equity'].iloc[0] - 1
        days = (df.index[-1] - df.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        daily_returns = df['Returns']
        volatility = daily_returns.std() * np.sqrt(252)
        max_drawdown = df['Drawdown'].min()
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / max(volatility, 0.0001)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / max(downside_deviation, 0.0001)
        num_trades = len(trades)
        if num_trades > 0:
            wins = sum(1 for t in trades if t['profit_loss'] > 0)
            losses = num_trades - wins
            win_rate = wins / num_trades
            avg_profit = np.mean([t['profit_loss_pct'] for t in trades if t['profit_loss'] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t['profit_loss_pct'] for t in trades if t['profit_loss'] <= 0]) if losses > 0 else 0
            total_profit = sum([t['profit_loss'] for t in trades if t['profit_loss'] > 0])
            total_loss = abs(sum([t['profit_loss'] for t in trades if t['profit_loss'] <= 0]))
            profit_factor = total_profit / max(total_loss, 0.0001)
            results = [1 if t['profit_loss'] > 0 else 0 for t in trades]
            max_consecutive_wins = max(sum(1 for _ in group) for key, group in itertools.groupby(results) if key == 1) if wins > 0 else 0
            max_consecutive_losses = max(sum(1 for _ in group) for key, group in itertools.groupby(results) if key == 0) if losses > 0 else 0
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    # --- New Methods for Untrade Backtest Pipeline ---
    def save_predictions_to_csv(self, output_path="blended_signals.csv"):
        """Save the generated signals to CSV for the untrade backtester."""
        test_df = self.generate_signals()
        # Ensure no consecutive duplicate signals
        signal = []
        prev = None
        for value in test_df['Signal']:
            if pd.isna(value):
                signal.append(0)
            elif value == prev:
                signal.append(0)
            else:
                signal.append(value)
            prev = value
        output_df = test_df.copy()
        output_df['signals'] = signal
        # Add any extra columns required by untrade
        output_df['Short_MA'] = output_df['ema_8']  # Example; adjust as needed
        output_df['Long_MA'] = output_df['ema_21']    # Example; adjust as needed
        output_df = output_df.reset_index()
        output_df.to_csv(output_path, index=False)
        return output_df
    
    def run_untrade_backtest(self, output_csv_path="blended_signals.csv"):
        """Run backtest using the untrade client."""
        client = Client()
        result = client.backtest(
            file_path=output_csv_path,
            leverage=1
        )
        return result
    
    def run_full_pipeline(self, output_csv_path="blended_signals.csv"):
        """Execute full pipeline: load data, generate signals, save to CSV, and run untrade backtest."""
        print("Loading data and calculating indicators...")
        self.load_data()
        print("Generating blended trading signals...")
        self.save_predictions_to_csv(output_csv_path)
        print(f"Signals saved to {output_csv_path}")
        print("Running backtest via untrade client...")
        backtest_result = self.run_untrade_backtest(output_csv_path)
        last_value = None
        for value in backtest_result:
            last_value = value
        print("Backtest complete.")
        print("Backtest results:", last_value)
        return last_value

# -------------------------------
# Main Execution Section
# -------------------------------
if __name__ == "__main__":
    # Set the data path to your CSV file containing columns: datetime, open, high, low, close, volume.
    data_path = r"data\2018-22\btc_18_22_1d.csv"  # Change this as needed

    # # To run the dynamic adaptive trader backtest:
    # print("=== Running Dynamic Adaptive Trader Backtest ===")
    # adaptive_trader = DynamicAdaptiveTrader(data_path, test_percentage=0.4, initial_risk_tolerance=0.5)
    # adaptive_result = adaptive_trader.run_full_pipeline(output_csv_path="adaptive_signals.csv")
    # print("\nDynamic Adaptive Trader Backtest Result:")
    # print(adaptive_result)

    # # To run the blended crypto trader backtest:
    # print("\n=== Running Blended Crypto Trader Backtest ===")
    # blended_trader = BlendedCryptoTrader(data_path, test_percentage=0.4)
    # blended_trader.load_data()
    # signals_df, trades, performance = blended_trader.backtest(initial_capital=10000, position_size=0.95, disable_shorts=False)
    # print("\nBlended Crypto Trader Performance Metrics:")
    # for key, value in performance.items():
    #     print(f"{key}: {value}")
    # print("\nTrades Executed:")
    # for trade in trades:
    #     print(trade)
    blended_trader = BlendedCryptoTrader(data_path, test_percentage=0.4)
    result = blended_trader.run_full_pipeline(output_csv_path="blended_signals.csv")
    print("\nBlended Crypto Trader Backtest Result:")
    print(result)