"""Data loading for the European portfolio & risk project.

- Fama-French factors: downloaded directly from Kenneth French's data library.
- Prices: downloaded with yfinance (Amsterdam tickers end in .AS).

Everything is cached in data/cache/ (gitignored) so repeated runs are fast.

Data sources and caveats (put these in your README too):
- Kenneth R. French Data Library (credit the source).
- yfinance prices are unofficial and show current listings only, so results
  carry survivorship bias. Spot-check a few series against another source.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
FF_BASE_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
)

# Dataset names as published on the French library site. Verify against the
# website if a download fails, since names occasionally change.
FF_DATASETS = {
    "europe_3": "Europe_3_Factors",
    "europe_5": "Europe_5_Factors",
    "europe_mom": "Europe_MOM_Factor",
    "developed_3": "Developed_3_Factors",
    "developed_5": "Developed_5_Factors",
    "developed_mom": "Developed_Mom_Factor",
}

DEFAULT_TICKERS = [
    "ASML.AS", "SHELL.AS", "INGA.AS", "ADYEN.AS", "PHIA.AS",
    "UNA.AS", "HEIA.AS", "AD.AS", "NN.AS", "ABN.AS",
]

_DATA_LINE = re.compile(r"^\s*\d{6,8}\s*,")
_MISSING = (-99.99, -999.0, -99.0)


def parse_french_csv(text: str, daily: bool = False) -> pd.DataFrame:
    """Parse the text of a French-library CSV into a DataFrame of decimal returns.

    The files contain a description header, a column header, the data block and
    sometimes a second (annual) block. Only the first data block is returned.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _DATA_LINE.match(ln)), None)
    if start is None:
        raise ValueError("No data rows found in the file.")

    # The column header is the closest non-empty line above the first data row.
    j = start - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    header = [c.strip() for c in lines[j].split(",")]
    header[0] = "date"

    rows = []
    for ln in lines[start:]:
        if not _DATA_LINE.match(ln):
            break  # end of the first block
        rows.append([c.strip() for c in ln.split(",")])

    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    fmt = "%Y%m%d" if daily else "%Y%m"
    df["date"] = pd.to_datetime(df["date"], format=fmt)
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df.isin(_MISSING))
    if not daily:
        df.index = df.index + pd.offsets.MonthEnd(0)
    return df / 100.0  # percent -> decimal


def load_french_factors(
    dataset: str = "europe_3", daily: bool = False, refresh: bool = False
) -> pd.DataFrame:
    """Load a Fama-French dataset (decimals). Cached after the first download."""
    if dataset not in FF_DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Options: {list(FF_DATASETS)}")
    name = FF_DATASETS[dataset] + ("_Daily" if daily else "")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{name}.csv"
    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    resp = requests.get(FF_BASE_URL.format(name=name), timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(member).decode("latin-1")

    df = parse_french_csv(text, daily=daily)
    df.to_csv(cache_file)
    return df


def load_prices(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted close prices from yfinance, cached to CSV."""
    import yfinance as yf  # imported here so factor loading works without it

    tickers = tickers or DEFAULT_TICKERS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"prices_{abs(hash(tuple(sorted(tickers)))) % 10**8}_{start}_{end}.csv"
    cache_file = CACHE_DIR / key
    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all")
    prices.to_csv(cache_file)
    return prices


def to_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Daily (log) returns from a price DataFrame."""
    import numpy as np

    r = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return r.dropna(how="all")


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns, indexed at month end to match the factor data."""
    m = prices.resample("ME").last().pct_change().dropna(how="all")
    return m


if __name__ == "__main__":
    f = load_french_factors("europe_3")
    print(f.tail())
    p = load_prices(["ASML.AS", "INGA.AS"], start="2018-01-01")
    print(p.tail())
