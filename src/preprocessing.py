import numpy as np
import pandas as pd


def compute_log_returns(df):
    """
    Compute log returns from a price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a 'Close' column

    Returns
    -------
    returns : pd.Series
        Log return series
    """
    return np.log(df['Close'] / df['Close'].shift(1)).dropna()


def introduce_missing(returns, halt_dates, n_random=50, seed=42):
    """
    Artificially introduce missing data to simulate real-world gaps.

    Parameters
    ----------
    returns : pd.Series
    halt_dates : list of str
        Start dates of trading halts (5 consecutive days each)
    n_random : int
        Number of random missing values to introduce
    seed : int

    Returns
    -------
    returns_missing : pd.Series
    """
    np.random.seed(seed)
    returns_missing = returns.copy()

    for start in halt_dates:
        mask = (returns_missing.index >= start)
        idx = returns_missing.index[mask][:5]
        returns_missing[idx] = np.nan

    random_idx = np.random.choice(returns_missing.index, size=n_random, replace=False)
    returns_missing[random_idx] = np.nan

    return returns_missing


def clean_returns(returns):
    """
    Clean a return series:
    - Interpolate missing values
    - Remove infinite values
    - Remove duplicate dates

    Parameters
    ----------
    returns : pd.Series

    Returns
    -------
    returns : pd.Series
    """
    returns = returns.interpolate(method='linear')
    returns = returns[~np.isinf(returns)]
    returns = returns[~returns.index.duplicated()]
    return returns


def normalize(returns):
    """
    Normalize returns to zero mean and unit variance.

    Parameters
    ----------
    returns : pd.Series

    Returns
    -------
    normalized : pd.Series
    """
    return (returns - returns.mean()) / returns.std()