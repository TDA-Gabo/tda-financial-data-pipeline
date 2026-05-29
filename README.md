# TDA Financial Data Pipeline
### From raw market data to TDA-ready point clouds

> Raw price data means nothing to TDA. This repo builds the bridge — cleaning, transforming, and embedding financial time series into the geometric structures that topology can analyze.

---

## What This Repo Does

Financial time series are 1D (1-dimensional) sequences — TDA needs point clouds in metric spaces. The pipeline has three stages:

1. **Data Collection** — pull raw OHLCV data from Yahoo Finance via `yfinance`
2. **Preprocessing** — compute returns, handle missing data, normalize
3. **Takens Embedding** — convert 1D return series into point clouds ready for TDA

The output feeds directly into the future projects in signal generation, risk monitoring, and regime detection.

---

## Repository Structure

```
tda-financial-data-pipeline/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_takens_embedding.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py          ← data collection & caching
│   ├── preprocessing.py ← returns, normalization, cleaning
│   └── embedding.py     ← Takens embedding
├── data/
│   └── raw/             ← cached market data
└── tests/
    └── test_pipeline.py
```

---

## Installation

```bash
git clone https://github.com/TDA-Gabo/tda-financial-data-pipeline.git
cd tda-financial-data-pipeline

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
yfinance        # Market data
ripser          # Persistent homology
persim          # Persistence diagrams
giotto-tda      # TDA pipeline
gudhi           # TDA library
scikit-learn    # ML utilities
jupyter
```

---

## Roadmap

- [x] Notebook 01: Data collection — pull and cache market data
- [x] Notebook 02: Preprocessing — returns, normalization, cleaning
- [x] Notebook 03: Takens embedding — FNN + point clouds
- [ ] Notebook 04: Visibility graphs — alternative pipeline
- [ ] src/ modules — clean reusable pipeline code
- [ ] tests/ — unit tests for pipeline components

---

## References

- Gidea & Katz (2018) — *Topological Data Analysis of Financial Time Series* — [arxiv.org/abs/1703.04385](https://arxiv.org/abs/1703.04385)
- Takens (1981) — *Detecting Strange Attractors in Turbulence*
- Güzel (2026) — *Persistent Homology of Time Series through Complex Networks* — [arxiv.org/abs/2605.01624](https://arxiv.org/abs/2605.01624)
- [yfinance documentation](https://pypi.org/project/yfinance/)

---

## Author

**TDA-Gabo**
*You may not see the data, but you can always describe its shape.*