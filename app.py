"""Streamlit app: Fama-French factor exposures for European stocks."""
import pandas as pd
import streamlit as st

from src.data_loader import (
    DEFAULT_TICKERS,
    FF_DATASETS,
    load_french_factors,
    load_prices,
    to_monthly_returns,
)
from src.factors import run_ff_regression

st.set_page_config(page_title="European Factor Explorer", layout="wide")
st.title("European Factor Explorer")
st.caption(
    "Fama-French regressions on Dutch/European stocks. Data: Kenneth R. French "
    "Data Library and Yahoo Finance (via yfinance). Free price data has "
    "survivorship bias; treat results as research, not investment advice."
)


@st.cache_data(show_spinner="Loading factors...")
def get_factors(dataset: str) -> pd.DataFrame:
    return load_french_factors(dataset)


@st.cache_data(show_spinner="Loading prices...")
def get_prices(tickers: tuple, start: str) -> pd.DataFrame:
    return load_prices(list(tickers), start=start)


with st.sidebar:
    st.header("Settings")
    dataset = st.selectbox(
        "Factor model",
        [k for k in FF_DATASETS if k in ("europe_3", "europe_5", "developed_3", "developed_5")],
    )
    tickers = st.multiselect("Stocks", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:5])
    start = st.text_input("Start date (YYYY-MM-DD)", "2012-01-01")

if not tickers:
    st.info("Pick at least one stock in the sidebar.")
    st.stop()

try:
    factors = get_factors(dataset)
    prices = get_prices(tuple(tickers), start)
except Exception as e:  # network or naming problems
    st.error(f"Could not load data: {e}")
    st.stop()

prices = prices.dropna(axis=1, how="all")
if prices.empty:
    st.error("No price data returned. Check the tickers and date.")
    st.stop()

monthly = to_monthly_returns(prices)
monthly.index = monthly.index.tz_localize(None) if monthly.index.tz else monthly.index

tab1, tab2 = st.tabs(["Prices", "Factor regression"])

with tab1:
    st.subheader("Cumulative growth of 1 EUR")
    st.line_chart((1 + monthly.fillna(0)).cumprod())

with tab2:
    stock = st.selectbox("Stock to analyse", list(monthly.columns))
    try:
        res, data = run_ff_regression(monthly[stock].dropna(), factors)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Alpha (monthly)", f"{res.params['const']:.4f}")
    c2.metric("R-squared", f"{res.rsquared:.2f}")
    c3.metric("Months used", int(res.nobs))

    table = pd.DataFrame(
        {"coef": res.params, "t-stat": res.tvalues, "p-value": res.pvalues}
    ).round(4)
    st.dataframe(table)
    st.markdown(
        "**Notes:** standard errors are Newey-West (HAC). Alpha is the monthly "
        "intercept. Results depend on the factor region, sample period and "
        "data quality, and a stock's own country exposure is not isolated here."
    )
