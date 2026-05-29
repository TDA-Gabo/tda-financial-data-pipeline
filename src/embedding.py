import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA


def false_nearest_neighbors(series, max_dim=10, rtol=15.0, threshold=1.0):
    """
    Compute the percentage of false nearest neighbors for each embedding dimension.
    Stops when FNN percentage drops below threshold.

    Parameters
    ----------
    series : np.ndarray
        1D time series
    max_dim : int
        Maximum embedding dimension to test
    rtol : float
        Threshold ratio for declaring a false neighbor (typically 10-15)
    threshold : float
        Stop when FNN percentage drops below this value (default 1.0%)

    Returns
    -------
    fnn_percentages : list
    optimal_d : int
    """
    fnn_percentages = []
    optimal_d = max_dim

    for d in range(1, max_dim + 1):
        n = len(series) - d
        X = np.array([series[i:i+d] for i in range(n)])

        tree = KDTree(X)
        distances, indices = tree.query(X, k=2)

        false_neighbors = 0
        total = 0

        for i in range(len(X) - 1):
            nn_idx = indices[i, 1]
            d_current = distances[i, 1]

            if d_current == 0:
                continue

            d_next = abs(series[i + d] - series[nn_idx + d])

            if d_next / d_current > rtol:
                false_neighbors += 1
            total += 1

        fnn_pct = (false_neighbors / total) * 100 if total > 0 else 0
        fnn_percentages.append(fnn_pct)

        if fnn_pct < threshold:
            optimal_d = d
            break

    return fnn_percentages, optimal_d


def takens_embedding(series, d):
    """
    Compute the Takens delay embedding of a time series.

    Parameters
    ----------
    series : np.ndarray
        1D time series
    d : int
        Embedding dimension

    Returns
    -------
    X : np.ndarray, sha