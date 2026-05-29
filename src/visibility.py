import numpy as np
import networkx as nx


def natural_visibility_graph(series):
    """
    Construct the Natural Visibility Graph (NVG) from a time series.

    Parameters
    ----------
    series : np.ndarray
        1D time series

    Returns
    -------
    G : networkx.Graph
    """
    n = len(series)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                interpolated = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                if series[k] >= interpolated:
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)
            else:
                break
    return G


def horizontal_visibility_graph(series):
    """
    Construct the Horizontal Visibility Graph (HVG) from a time series.

    Parameters
    ----------
    series : np.ndarray
        1D time series

    Returns
    -------
    G : networkx.Graph
    """
    n = len(series)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            min_val = min(series[i], series[j])
            visible = all(series[k] < min_val for k in range(i + 1, j))
            if visible:
                G.add_edge(i, j)
            else:
                break
    return G


def graph_to_dissimilarity(G):
    """
    Convert a visibility graph to a dissimilarity matrix
    using shortest path distances.

    Parameters
    ----------
    G : networkx.Graph

    Returns
    -------
    D : np.ndarray, shape (n, n)
    """
    n = G.number_of_nodes()
    D = np.zeros((n, n))

    lengths = dict(nx.all_pairs_shortest_path_length(G))

    for i in range(n):
        for j in range(n):
            D[i, j] = lengths[i].get(j, np.inf)

    return D