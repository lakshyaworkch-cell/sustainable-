"""Streamlit app: Fama-French factor exposures for European stocks."""
from datetime import date

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

st.set_page_config(
    page_title="European Factor Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- light styling on top of the dark theme (.streamlit/config.toml) ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    h1 {font-weight: 700; letter-spacing: -0.5px;}
    div[data-testid="stMetric"] {
        background: #1A1F2B; border: 1px solid #2A3142;
        border-radius: 12px; padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] {opacity: 0.75;}
    button[data-baseweb="tab"] {font-size: 1rem;}
    section[data-testid="stSidebar"] {border-right: 1px solid #2A3142;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 European Factor Explorer")
st.caption(
    "Fama-French regressions on Dutch and European stocks. Data: Kenneth R. French "
    "Data Library and Yahoo Finance (via yfinance). Free price data has survivorship "
    "bias; treat results as research, not investment advice."
)

MODEL_LABELS = {
    "europe_3": "Europe 3-factor",
    "europe_5": "Europe 5-factor",
    "developed_3": "Developed markets 3-factor",
    "developed_5": "Developed markets 5-factor",
}


@st.cache_data(show_spinner="Loading factors...")
def get_factors(dataset: str) -> pd.DataFrame:
    return load_french_factors(dataset)


@st.cache_data(show_spinner="Loading prices...")
def get_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    return load_prices(list(tickers), start=start, end=end)


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    dataset = st.selectbox(
        "Factor model",
        list(MODEL_LABELS),
        format_func=lambda k: MODEL_LABELS[k],
    )
    tickers = st.multiselect("Stocks", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:5])

    st.markdown("**Date range**")
    st.caption("Type the date or pick it from the calendar (YYYY/MM/DD).")
    start_date = st.date_input(
        "Start date",
        value=date(2012, 1, 1),
        min_value=date(1990, 1, 1),   # allows much earlier start dates
        max_value=date.today(),
        format="YYYY/MM/DD",
    )
    end_date = st.date_input(
        "End date",
        value=date.today(),
        min_value=date(1990, 1, 1),
        max_value=date.today(),
        format="YYYY/MM/DD",
    )

if not tickers:
    st.info("Pick at least one stock in the sidebar.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before the end date.")
    st.stop()

try:
    factors = get_factors(dataset)
    prices = get_prices(tuple(tickers), start_date.isoformat(), end_date.isoformat())
except Exception as e:  # network or naming problems
    st.error(f"Could not load data: {e}")
    st.stop()

prices = prices.dropna(axis=1, how="all")
if prices.empty:
    st.error("No price data returned. Check the tickers and dates.")
    st.stop()

monthly = to_monthly_returns(prices)
if getattr(monthly.index, "tz", None) is not None:
    monthly.index = monthly.index.tz_localize(None)

# ---------- tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Prices", "🧮 Factor regression", "ℹ️ About"])

with tab1:
    st.subheader("Growth of 1 EUR invested")
    st.line_chart((1 + monthly.fillna(0)).cumprod())
    st.caption(
        f"Month-end returns from {monthly.index.min():%b %Y} to "
        f"{monthly.index.max():%b %Y}."
    )

with tab2:
    stock = st.selectbox("Stock to analyse", list(monthly.columns))
    try:
        res, data = run_ff_regression(monthly[stock].dropna(), factors)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha (monthly)", f"{res.params['const']:.4f}")
    c2.metric("R-squared", f"{res.rsquared:.2f}")
    c3.metric("Months used", int(res.nobs))
    c4.metric("Market beta", f"{res.params.get('Mkt-RF', float('nan')):.2f}")

    st.caption(
        f"Sample: {data.index.min():%b %Y} to {data.index.max():%b %Y}  |  "
        f"Model: {MODEL_LABELS[dataset]}  |  Monthly frequency"
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Coefficients")
        table = pd.DataFrame(
            {"coef": res.params, "t-stat": res.tvalues, "p-value": res.pvalues}
        ).round(4)
        table["significant (p<0.05)"] = table["p-value"] < 0.05
        st.dataframe(table, width="stretch")
    with right:
        st.subheader("Factor loadings")
        st.bar_chart(res.params.drop("const"))

    st.markdown(
        "**How to read this:** the market beta shows how strongly the stock moves "
        "with the market. SMB and HML show tilts toward small and value stocks "
        "(negative means large and growth). Alpha is the average monthly return left "
        "after the factors, and R-squared is the share of movement they explain. "
        "Standard errors are Newey-West (HAC)."
    )

with tab3:
    st.subheader("About this app")
    st.markdown(
        """
        - **Factors:** monthly Fama-French data from Kenneth R. French's Data Library.
        - **Prices:** adjusted closing prices from Yahoo Finance, converted to
          month-end returns.
        - **Method:** OLS regression of monthly excess returns on the factors, with
          Newey-West standard errors.
        - **Limitations:** survivorship bias in free price data, short samples for
          recently listed stocks, and results depend on the chosen factor region and
          period.
        """
    )
