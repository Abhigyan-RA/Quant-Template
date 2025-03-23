import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# 1. Technical Indicators 
# --------------------------------------------------------------------------------
def calculate_stochastic_oscillator(df, n=14, d=3, epsilon=1e-10):
    high_roll = df['High'].rolling(window=n).max()
    low_roll = df['Low'].rolling(window=n).min()

    denom = (high_roll - low_roll).replace(0, epsilon)
    
    # Fast %K
    k = 100 * (df['Close'] - low_roll) / denom
    # Slow %K (Fast %D)
    d_val = k.rolling(window=d).mean()
    
    return k, d_val

def calculate_obv(df):
    obv = pd.Series(index=df.index, dtype='float64')
    obv.iloc[0] = df['No. of contracts'].iloc[0]
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['No. of contracts'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['No. of contracts'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_aroon_up(df, period=14):
    high_list = df['High'].rolling(period).apply(lambda x: x.argmax())
    aroon_up = ((period - high_list) / period) * 100
    return aroon_up

def calculate_williams_r(df, period=14):
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    wr = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    return wr

def calculate_rsi(df, period=14, epsilon=1e-10):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
    
    rs = gain / (loss + epsilon)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --------------------------------------------------------------------------------
# 2. Data Preprocessing and Feature Engineering 
# --------------------------------------------------------------------------------
def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['SO_k'], df['SO_d'] = calculate_stochastic_oscillator(df)
    df['OBV'] = calculate_obv(df)
    df['Aroon_Up'] = calculate_aroon_up(df)
    df['Williams_R'] = calculate_williams_r(df)
    df['RSI'] = calculate_rsi(df)
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df

# --------------------------------------------------------------------------------
# 3. Model Training 
# --------------------------------------------------------------------------------
def train_model(df):
    features = [
        'SO_k', 'SO_d', 'OBV', 'Aroon_Up',
        'Williams_R', 'RSI'
    ]
    
    df = df.dropna()
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test, y_test

# --------------------------------------------------------------------------------
# 4. Trading Strategy Implementation 
# --------------------------------------------------------------------------------
class TradingStrategy:
    def __init__(self, initial_capital=10000000, max_lot_size=100):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.long_position = 0    # Track long positions
        self.short_position = 0   # Track short positions
        self.max_lot_size = max_lot_size
        self.df = None  

        self.trades = []
        self.trade_history = []
        self.monthly_trades_count = {}

    def _calculate_commission(self, trade_value):
        return min(20, trade_value * 0.0003)

    def execute_trades(self, df, predictions, confidences):
        df = df.copy()
        # Modify signals: 1 for long, -1 for short, 0 for exit
        df['Signal'] = predictions
        df['Signal'] = df['Signal'].map({1: 1, 0: -1})  # Map 0 to short (-1)
        df['Confidence'] = confidences
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            confidence = df['Confidence'].iloc[i]
            date = df['Date'].iloc[i]
            month_key = date.strftime('%Y-%m')
            
            if month_key not in self.monthly_trades_count:
                self.monthly_trades_count[month_key] = 0
            
            lot_size = int(self.max_lot_size * confidence)
            
            # Long Entry
            if signal == 1 and self.long_position == 0 and self.short_position == 0:
                self._open_long_position(current_price, lot_size, date)
                self.monthly_trades_count[month_key] += 1
            
            # Short Entry
            elif signal == -1 and self.short_position == 0 and self.long_position == 0:
                self._open_short_position(current_price, lot_size, date)
                self.monthly_trades_count[month_key] += 1
            
            # Exit Long Position
            elif signal == -1 and self.long_position > 0:
                self._close_long_position(current_price, date)
                self.monthly_trades_count[month_key] += 1
            
            # Exit Short Position
            elif signal == 1 and self.short_position > 0:
                self._close_short_position(current_price, date)
                self.monthly_trades_count[month_key] += 1

        final_report = self.calculate_performance_metrics()
        
        return final_report

    def _open_long_position(self, current_price, lot_size, date):
        buy_value = current_price * lot_size
        buy_commission = self._calculate_commission(buy_value)
        total_cost = buy_value + buy_commission
        
        amount_from_capital = total_cost * 0.10
        leverage_amount = total_cost * 0.90
        
        if self.capital >= amount_from_capital:
            self.long_position = lot_size
            self.capital -= amount_from_capital
            
            self.trades.append({
                'Type': 'Long',
                'Entry Date': date,
                'Entry Price': current_price,
                'Commission': buy_commission,
                'Lot Size': lot_size,
                'Leverage Amount': leverage_amount,
                'Capital Used': amount_from_capital
            })
        else:
            print(f"Insufficient capital to open long position on {date}. Required: {amount_from_capital}, Available: {self.capital}")

    def _open_short_position(self, current_price, lot_size, date):
        sell_value = current_price * lot_size
        sell_commission = self._calculate_commission(sell_value)
        total_cost = sell_value + sell_commission
        
        amount_from_capital = total_cost * 0.10
        leverage_amount = total_cost * 0.90
        
        if self.capital >= amount_from_capital:
            self.short_position = lot_size
            self.capital -= amount_from_capital
            
            self.trades.append({
                'Type': 'Short',
                'Entry Date': date,
                'Entry Price': current_price,
                'Commission': sell_commission,
                'Lot Size': lot_size,
                'Leverage Amount': leverage_amount,
                'Capital Used': amount_from_capital
            })
        else:
            print(f"Insufficient capital to open short position on {date}. Required: {amount_from_capital}, Available: {self.capital}")

    def _close_long_position(self, current_price, exit_date):
        last_trade = self.trades[-1]
        entry_price = last_trade['Entry Price']
        initial_commission = last_trade['Commission']
        
        exit_value = current_price * self.long_position
        exit_commission = self._calculate_commission(exit_value)
        
        raw_profit = (current_price - entry_price) * self.long_position
        total_commission = initial_commission + exit_commission
        net_profit = raw_profit - total_commission
        
        self.capital += last_trade['Capital Used'] + net_profit
        
        self._record_closed_trade(last_trade, exit_date, current_price, 
                                net_profit, initial_commission, exit_commission)
        
        self.long_position = 0

    def _close_short_position(self, current_price, exit_date):
        last_trade = self.trades[-1]
        entry_price = last_trade['Entry Price']
        initial_commission = last_trade['Commission']
        
        exit_value = current_price * self.short_position
        exit_commission = self._calculate_commission(exit_value)
        
        raw_profit = (entry_price - current_price) * self.short_position
        total_commission = initial_commission + exit_commission
        net_profit = raw_profit - total_commission
        
        self.capital += last_trade['Capital Used'] + net_profit
        
        self._record_closed_trade(last_trade, exit_date, current_price, 
                                net_profit, initial_commission, exit_commission)
        
        self.short_position = 0

    def _record_closed_trade(self, entry_trade, exit_date, exit_price, 
                           net_profit, entry_commission, exit_commission):
        trade_record = {
            'Type': entry_trade['Type'],
            'Entry Date': entry_trade['Entry Date'],
            'Exit Date': exit_date,
            'Entry Price': entry_trade['Entry Price'],
            'Exit Price': exit_price,
            'Profit': net_profit,
            'Entry Commission': entry_commission,
            'Exit Commission': exit_commission,
            'Total Commission': entry_commission + exit_commission,
            'Duration': (exit_date - entry_trade['Entry Date']).days,
            'Lot Size': entry_trade['Lot Size'],
            'Capital Used': entry_trade['Capital Used']
        }
        self.trade_history.append(trade_record)

    def calculate_performance_metrics(self):
        if not self.trade_history:
            return "No trades executed."
        
        profits = [trade['Profit'] for trade in self.trade_history]
        durations = [trade['Duration'] for trade in self.trade_history]
        capital_used = [trade['Capital Used'] for trade in self.trade_history]
        
        long_trades = [t for t in self.trade_history if t['Type'] == 'Long']
        short_trades = [t for t in self.trade_history if t['Type'] == 'Short']
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_trades = len(self.trade_history)
        winning_trades_count = len(winning_trades)
        win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
        
        long_profits = [t['Profit'] for t in long_trades]
        short_profits = [t['Profit'] for t in short_trades]
        
        # Calculate new metrics
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = sum(losing_trades) if losing_trades else 0
        avg_winning_trade = np.mean(winning_trades) if winning_trades else 0
        avg_losing_trade = np.mean(losing_trades) if losing_trades else 0
        largest_winning_trade = max(profits) if profits else 0
        largest_losing_trade = min(profits) if profits else 0
        net_profit = sum(profits)

        total_return_pct = (net_profit / self.initial_capital) * 100

        
        # Calculate Sortino Ratio
        sortino_ratio = self._calculate_sortino_ratio(profits, capital_used)
        
        # Calculate Buy-and-Hold Return
        buy_hold_return = self._calculate_buy_and_hold_return()
        
        metrics = {
            'Initial Capital': self.initial_capital,  # Added initial capital
            'Final Capital': self.initial_capital + net_profit,  # Added final capital
            'Total Return (%)': total_return_pct,
            'Total Trades': total_trades,
            'Long Trades': len(long_trades),
            'Short Trades': len(short_trades),
            'Win Rate (%)': win_rate,
            'Long Positions P&L': sum(long_profits) if long_profits else 0,
            'Short Positions P&L': sum(short_profits) if short_profits else 0,
            'Net Profit': net_profit,
            'Gross Profit': gross_profit,
            'Gross Loss': gross_loss,
            'Average Profit per Trade': np.mean(profits),
            'Average Winning Trade': avg_winning_trade,
            'Average Losing Trade': avg_losing_trade,
            'Largest Winning Trade': largest_winning_trade,
            'Largest Losing Trade': largest_losing_trade,
            'Average Duration (days)': np.mean(durations),
            'Sharpe Ratio': self._calculate_sharpe_ratio(profits, capital_used),
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': self._calculate_max_drawdown(profits) * 100,
            'Long Win Rate (%)': self._calculate_win_rate(long_profits),
            'Short Win Rate (%)': self._calculate_win_rate(short_profits),
            'Buy-and-Hold Return (%)': buy_hold_return
        }
        
        return metrics

    def _calculate_sortino_ratio(self, profits, capital_used, risk_free_rate=0.05):
        """
        Calculate Sortino Ratio using only negative returns for denominator
        """
        returns = pd.Series(profits) / sum(capital_used)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        # Calculate downside returns (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf')
        
        return np.sqrt(252) * (excess_returns.mean() / downside_returns.std())

    def _calculate_buy_and_hold_return(self):
        """
        Calculate buy-and-hold return for the trading period
        """
        if not self.trade_history or not hasattr(self, 'df'):
            return 0
        
        # Get the first and last dates from trade history
        start_date = min(trade['Entry Date'] for trade in self.trade_history)
        end_date = max(trade['Exit Date'] for trade in self.trade_history)
        
        # Filter the dataframe for the trading period
        mask = (self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)
        period_df = self.df[mask]
        
        if len(period_df) < 2:
            return 0
        
        start_price = period_df.iloc[0]['Close']
        end_price = period_df.iloc[-1]['Close']
        
        buy_hold_return = ((end_price - start_price) / start_price) * 100
        return buy_hold_return

    def _calculate_sharpe_ratio(self, profits, capital_used):
        returns = pd.Series(profits) / sum(capital_used)
        risk_free_rate = 0.05 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        
        if excess_returns.std() == 0:
            return float('inf')
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    def _calculate_max_drawdown(self, profits):
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(min(drawdown, default=0))

    def _calculate_win_rate(self, profits):
        if not profits:
            return 0
        winning_trades = sum(1 for p in profits if p > 0)
        return (winning_trades / len(profits)) * 100

# --------------------------------------------------------------------------------
# 5. Main Execution: Training + Test Performance 
# --------------------------------------------------------------------------------
def run_trading_system(df):
    processed_df = prepare_data(df)
    
    model, scaler, X_test, y_test = train_model(processed_df)
    
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    confidences = model.predict_proba(X_test_scaled)[:, 1]
    
    strategy = TradingStrategy(initial_capital=10000000, max_lot_size=100)
    test_df = processed_df.iloc[-len(X_test):]
    strategy.df = test_df  
    performance_metrics = strategy.execute_trades(test_df, predictions, confidences)
    
    return model, scaler, performance_metrics

# --------------------------------------------------------------------------------
# 6. Backtesting
def backtest_trading_system(df, model, scaler):
    prepared_df = prepare_data(df).dropna()
    
    features = [
        'SO_k', 'SO_d', 'OBV', 'Aroon_Up',
        'Williams_R', 'RSI'
    ]
    X_new = prepared_df[features]
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    confidences = model.predict_proba(X_new_scaled)[:, 1]
    
    strategy = TradingStrategy(initial_capital=1000000, max_lot_size=100)
    strategy.df = prepared_df  
    backtest_metrics = strategy.execute_trades(prepared_df, predictions, confidences)
    
    return backtest_metrics

if __name__ == "__main__":
    df = pd.read_csv('contract wise data.csv') #Training dataframe
    df.columns = df.columns.str.strip()  # Remove leading and trailing spaces

    model, scaler, performance_metrics = run_trading_system(df)
    
    contract_df = pd.read_csv('contract wise data.csv') #Backtesting dataframe
    contract_df.columns = contract_df.columns.str.strip()  # Remove leading and trailing spaces

    backtest_metrics = backtest_trading_system(contract_df, model, scaler)
    
    print("\nBacktest Performance Metrics:")
    for metric, value in backtest_metrics.items():
        print(f"{metric}: {value}")