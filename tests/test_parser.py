from src.data_loader import parse_french_csv

SAMPLE = """This file was created by CMPT_ME_BEME_RETS using the 202509 CRSP database.

,Mkt-RF,SMB,HML,RF
199001,  -1.50,   0.20,  -0.10,   0.60
199002,  -99.99,   0.30,   0.40,   0.55
199003,   2.00,  -0.50,   0.10,   0.57

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1990,   5.0,  1.0,  2.0,  7.0
"""


def test_parse_monthly():
    df = parse_french_csv(SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3                      # annual block excluded
    assert abs(df.iloc[0]["Mkt-RF"] + 0.015) < 1e-12  # percent -> decimal
    assert df.iloc[1].isna()["Mkt-RF"]       # -99.99 treated as missing
    assert df.index[0].day == 31             # month-end index
