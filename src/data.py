import os
import yfinance as yf
import pandas as pd


def fetch_ticker(ticker, start, end, cache=True, cache_dir='../data/raw'):
    """
    Fetch OHLCV data for a ticker from Yahoo Finance.
    Caches locally to avoid redundant API calls.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. '^GSPC', 'AAPL')
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format
    cache : bool
        If True, cache data locally and load from cache if available
    cache_dir : str
        Directory to store cached data

    Returns
    -------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker.replace('^', '')}_{start}_{end}.csv")

    if cache and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"Fetching {ticker} from Yahoo Finance...")
    df = yf.Ticker(ticker).history(start=start, end=end)
    
    if cache:
        df.to_csv(cache_path)
        print(f"Cached to {cache_path}")

    return df