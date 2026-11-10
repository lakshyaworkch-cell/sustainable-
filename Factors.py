"""Fama-French factor regression."""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def run_ff_regression(stock_returns: pd.Series, factors: pd.DataFrame):
    """Regress a stock's monthly excess returns on factor returns.

    stock_returns: monthly simple returns (decimals), month-end index.
    factors: decimal factor returns incl. an 'RF' column, month-end index.
    Returns (fitted OLS results with HAC standard errors, aligned DataFrame).
    """
    factor_cols = [c for c in factors.columns if c != "RF"]
    data = pd.concat([stock_returns.rename("ret"), factors], axis=1, join="inner")
    data = data.dropna()
    if len(data) < 24:
        raise ValueError(
            f"Only {len(data)} overlapping months; need at least 24 for a regression."
        )
    y = data["ret"] - data["RF"]
    X = sm.add_constant(data[factor_cols])
    # Newey-West (HAC) standard errors guard against autocorrelation/heteroskedasticity
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return res, data
