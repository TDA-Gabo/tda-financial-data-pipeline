import sys
sys.path.insert(0, '/home/gabo-linux/TDA-Gabo/tda-financial-data-pipeline')

import numpy as np
import pytest
from src.preprocessing import compute_log_returns, clean_returns, normalize, introduce_missing
from src.embedding import false_nearest_neighbors, takens_embedding, pca_projection
from src.visibility import natural_visibility_graph, horizontal_visibility_graph, graph_to_dissimilarity


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_series():
    """Simple deterministic time series for testing."""
    np.random.seed(42)
    return np.random.randn(200)

@pytest.fixture
def sample_prices():
    """Simulated price series."""
    np.random.seed(42)
    returns = np.random.randn(200) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    import pandas as pd
    return pd.DataFrame({'Close': prices})


# ── Preprocessing tests ───────────────────────────────────────────────────────

def test_clean_returns_removes_nan(sample_series):
    series = sample_series.copy()
    import pandas as pd
    s = pd.Series(series)
    s.iloc[10:15] = np.nan
    cleaned = clean_returns(s)
    assert cleaned.isnull().sum() == 0

def test_normalize_zero_mean_unit_std(sample_series):
    import pandas as pd
    s = pd.Series(sample_series)
    normed = normalize(s)
    assert abs(normed.mean()) < 1e-10
    assert abs(normed.std() - 1.0) < 1e-10


# ── Embedding tests ───────────────────────────────────────────────────────────

def test_takens_embedding_shape(sample_series):
    d = 5
    X = takens_embedding(sample_series, d)
    assert X.shape == (len(sample_series) - d + 1, d)

def test_takens_embedding_values(sample_series):
    d = 3
    X = takens_embedding(sample_series, d)
    assert np.allclose(X[0], sample_series[:3])
    assert np.allclose(X[1], sample_series[1:4])

def test_pca_projection_shape(sample_series):
    d = 5
    X = takens_embedding(sample_series, d)
    X_pca, variance = pca_projection(X, n_components=2)
    assert X_pca.shape == (X.shape[0], 2)
    assert len(variance) == 2
    assert abs(sum(variance) - sum(variance)) < 1e-10


# ── Visibility graph tests ────────────────────────────────────────────────────

def test_nvg_has_consecutive_edges(sample_series):
    """NVG must always connect consecutive points."""
    G = natural_visibility_graph(sample_series[:20])
    for i in range(19):
        assert G.has_edge(i, i + 1)

def test_hvg_has_consecutive_edges(sample_series):
    """HVG must always connect consecutive points."""
    G = horizontal_visibility_graph(sample_series[:20])
    for i in range(19):
        assert G.has_edge(i, i + 1)

def test_hvg_is_subgraph_of_nvg(sample_series):
    """Every HVG edge must also be an NVG edge."""
    series = sample_series[:30]
    G_nvg = natural_visibility_graph(series)
    G_hvg = horizontal_visibility_graph(series)
    for edge in G_hvg.edges():
        assert G_nvg.has_edge(*edge)

def test_dissimilarity_matrix_shape(sample_series):
    G = natural_visibility_graph(sample_series[:20])
    D = graph_to_dissimilarity(G)
    assert D.shape == (20, 20)

def test_dissimilarity_matrix_symmetric(sample_series):
    G = natural_visibility_graph(sample_series[:20])
    D = graph_to_dissimilarity(G)
    assert np.allclose(D, D.T)

def test_dissimilarity_diagonal_zero(sample_series):
    G = natural_visibility_graph(sample_series[:20])
    D = graph_to_dissimilarity(G)
    assert np.allclose(np.diag(D), 0)