"""Streamlit app: Fama-French factor exposures for European stocks (dark UI)."""
from datetime import date

import altair as alt
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

# ---------------------------------------------------------------------------
# Dark theme forced from inside the app, so it works even if
# .streamlit/config.toml is missing or not picked up.
# ---------------------------------------------------------------------------
BG, PANEL, BORDER, TEXT, MUTED, ACCENT = (
    "#0E1117", "#1A1F2B", "#2A3142", "#E6E9EF", "#9AA3B5", "#4F8BF9",
)

st.markdown(
    f"""
    <style>
    /* page and header */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stToolbar"] {{background-color: {BG} !important;}}
    .block-container {{padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}}

    /* sidebar */
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
        background-color: {PANEL} !important; border-right: 1px solid {BORDER};
    }}

    /* text */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stMarkdown, .stCaption,
    [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] {{
        color: {TEXT};
    }}
    [data-testid="stCaptionContainer"], .stCaption {{color: {MUTED} !important;}}
    h1 {{font-weight: 700; letter-spacing: -0.5px;}}

    /* inputs, selects, date boxes */
    div[data-baseweb="select"] > div, div[data-baseweb="input"],
    div[data-baseweb="input"] > div, div[data-baseweb="base-input"],
    input, textarea {{
        background-color: {BG} !important; color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}
    div[data-baseweb="select"] svg, div[data-baseweb="input"] svg {{fill: {MUTED};}}
    span[data-baseweb="tag"] {{background-color: {ACCENT} !important;}}
    span[data-baseweb="tag"] span {{color: #FFFFFF !important;}}

    /* dropdown menus and calendar popups (rendered outside the main page) */
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"], ul[role="listbox"], li[role="option"],
    div[data-baseweb="calendar"], div[data-baseweb="calendar"] * {{
        background-color: {PANEL} !important; color: {TEXT} !important;
    }}
    li[role="option"]:hover, li[aria-selected="true"] {{
        background-color: {BORDER} !important;
    }}
    div[data-baseweb="calendar"] [aria-selected="true"] {{
        background-color: {ACCENT} !important; color: #FFFFFF !important;
    }}

    /* tabs */
    button[data-baseweb="tab"] {{font-size: 1rem; color: {MUTED} !important;}}
    button[data-baseweb="tab"][aria-selected="true"] {{color: {TEXT} !important;}}
    div[data-baseweb="tab-highlight"] {{background-color: {ACCENT} !important;}}
    div[data-baseweb="tab-border"] {{background-color: {BORDER} !important;}}

    /* metric cards */
    div[data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 12px; padding: 14px 18px;
    }}
    div[data-testid="stMetricLabel"] p {{color: {MUTED} !important;}}

    /* static tables (st.table renders as HTML, so it follows this CSS) */
    [data-testid="stTable"] table {{border-collapse: collapse; width: 100%;}}
    [data-testid="stTable"] th {{
        background-color: {PANEL} !important; color: {MUTED} !important;
        border-bottom: 1px solid {BORDER} !important;
    }}
    [data-testid="stTable"] td {{
        background-color: {BG} !important; color: {TEXT} !important;
        border-bottom: 1px solid {BORDER} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Dark styling for Altair charts so they don't depend on Streamlit's theme."""
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=BORDER, domainColor=BORDER,
            tickColor=BORDER,
        )
        .configure_legend(labelColor=TEXT, titleColor=MUTED)
        .configure_title(color=TEXT)
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
        "Factor model", list(MODEL_LABELS), format_func=lambda k: MODEL_LABELS[k]
    )
    tickers = st.multiselect("Stocks", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:5])

    st.markdown("**Date range**")
    st.caption("Type the date or pick it from the calendar (YYYY/MM/DD).")
    start_date = st.date_input(
        "Start date",
        value=date(2012, 1, 1),
        min_value=date(1990, 1, 1),
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
    prices = get_prices(tickers=tuple(tickers), start=start_date.isoformat(), end=end_date.isoformat())
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
    growth = (1 + monthly.fillna(0)).cumprod()
    long = growth.rename_axis("date").reset_index().melt(
        "date", var_name="stock", value_name="growth"
    )
    line = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("growth:Q", title="Growth of 1 EUR"),
            color=alt.Color("stock:N", title=None),
            tooltip=["date:T", "stock:N", alt.Tooltip("growth:Q", format=".2f")],
        )
        .properties(height=380)
    )
    st.altair_chart(style_chart(line), theme=None, width="stretch")
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
        display = table.map(lambda v: f"{v:.4f}")
        display["significant (p<0.05)"] = table["p-value"].lt(0.05).map(
            {True: "Yes", False: "No"}
        )
        st.table(display)
    with right:
        st.subheader("Factor loadings")
        loadings = res.params.drop("const").rename_axis("factor").reset_index()
        loadings.columns = ["factor", "loading"]
        bars = (
            alt.Chart(loadings)
            .mark_bar(color=ACCENT, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("factor:N", sort=None, title=None),
                x=alt.X("loading:Q", title="Loading"),
                tooltip=["factor:N", alt.Tooltip("loading:Q", format=".3f")],
            )
            .properties(height=260)
        )
        st.altair_chart(style_chart(bars), theme=None, width="stretch")

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
