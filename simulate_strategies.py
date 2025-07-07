import pandas as pd
import numpy as np
import yfinance as yf


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical OHLC data for the given ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data retrieved for {ticker}")
    return df


def buy_and_hold(data: pd.DataFrame) -> pd.Series:
    """Simple buy and hold strategy. Returns portfolio value series."""
    returns = data['Adj Close'].pct_change().fillna(0)
    return (1 + returns).cumprod()


def moving_average_crossover(data: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """Moving average crossover strategy."""
    short_ma = data['Adj Close'].rolling(window=short_window).mean()
    long_ma = data['Adj Close'].rolling(window=long_window).mean()
    signals = (short_ma > long_ma).astype(int)
    returns = data['Adj Close'].pct_change().fillna(0)
    strategy_returns = signals.shift(1) * returns
    strategy_returns.fillna(0, inplace=True)
    return (1 + strategy_returns).cumprod()


def rsi_strategy(data: pd.DataFrame, period: int = 14, lower: float = 30, upper: float = 70) -> pd.Series:
    """RSI based trading strategy."""
    delta = data['Adj Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    position = np.where(rsi < lower, 1, np.where(rsi > upper, 0, np.nan))
    position = pd.Series(position, index=data.index).fillna(method='ffill').fillna(0)
    returns = data['Adj Close'].pct_change().fillna(0)
    strategy_returns = position.shift(1) * returns
    strategy_returns.fillna(0, inplace=True)
    return (1 + strategy_returns).cumprod()


def momentum_strategy(data: pd.DataFrame, window: int = 252) -> pd.Series:
    """Momentum strategy based on past 'window' days performance."""
    momentum = data['Adj Close'].pct_change(window)
    position = (momentum > 0).astype(int)
    returns = data['Adj Close'].pct_change().fillna(0)
    strategy_returns = position.shift(1) * returns
    strategy_returns.fillna(0, inplace=True)
    return (1 + strategy_returns).cumprod()


def bollinger_band_strategy(data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.Series:
    """Mean reversion strategy using Bollinger Bands."""
    rolling_mean = data['Adj Close'].rolling(window=window).mean()
    rolling_std = data['Adj Close'].rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    position = np.where(data['Adj Close'] < lower_band, 1,
                        np.where(data['Adj Close'] > upper_band, 0, np.nan))
    position = pd.Series(position, index=data.index).fillna(method='ffill').fillna(0)
    returns = data['Adj Close'].pct_change().fillna(0)
    strategy_returns = position.shift(1) * returns
    strategy_returns.fillna(0, inplace=True)
    return (1 + strategy_returns).cumprod()


def run_strategies(ticker: str = "VOO", start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """Run all strategies and compare with buy and hold."""
    data = fetch_data(ticker, start, end or pd.Timestamp.today().strftime("%Y-%m-%d"))

    results = pd.DataFrame(index=data.index)
    results['Buy&Hold'] = buy_and_hold(data)
    results['MA_Crossover'] = moving_average_crossover(data)
    results['RSI'] = rsi_strategy(data)
    results['Momentum'] = momentum_strategy(data)
    results['Bollinger'] = bollinger_band_strategy(data)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate trading strategies for VOO")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date in YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="strategies.csv", help="CSV file to store results")
    args = parser.parse_args()

    results = run_strategies("VOO", start=args.start, end=args.end)
    results.to_csv(args.output)
    print(f"Results saved to {args.output}")

