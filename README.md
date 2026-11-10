# European Factor Explorer

A Streamlit app that runs Fama-French factor regressions on Dutch and European
stocks, using free data.

## What it does
- Downloads Fama-French factors (Europe / developed markets) from the Kenneth R.
  French Data Library and converts them to decimal returns.
- Downloads adjusted prices for Amsterdam-listed stocks (tickers end in `.AS`)
  with yfinance.
- Regresses a stock's monthly excess returns on the factors, with Newey-West
  (HAC) standard errors, and shows alpha, factor loadings, t-stats and R-squared.

## Run locally
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Project structure
```
app.py               Streamlit app (entry point)
src/data_loader.py   Factor and price downloads, caching, return helpers
src/factors.py       Fama-French regression
tests/               Parser test on a sample file
requirements.txt
```

## Data sources and limitations
- Factors: Kenneth R. French Data Library (credit: Kenneth R. French).
- Prices: Yahoo Finance via yfinance (unofficial).
- Free price data lists current constituents only, so results carry
  survivorship bias. Adjusted prices can contain errors; spot-check series.
- Monthly regressions on a short sample give wide confidence intervals.
- This is research code, not investment advice.

## Possible extensions
- Add EGARCH-t volatility forecasts and a regime-switching model.
- Portfolio construction (MPT / Black-Litterman) and a walk-forward backtest
  with transaction costs.
- VaR / expected shortfall with backtests.
