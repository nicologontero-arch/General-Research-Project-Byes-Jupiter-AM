"""Coverage of every input series, as its loader returns it.

'rows' is what the file holds, 'used' what survives the loader's filtering; the
two differ where a file carries blank rows. Nothing is constructed here.
"""
import warnings

import pandas as pd

import cp_core as cc
import country_cp as ccp
import cp_intl as ci
import spread as sp
import strategy as st

warnings.filterwarnings("ignore")

cc.report("data_summary", "Input data coverage",
          "every series as its loader returns it, with the raw row count beside "
          "the count that survives filtering")


def span(s):
    """First and last index date, ISO."""
    return str(s.index[0].date()), str(s.index[-1].date())


def raw_rows(path, **kw):
    """Rows in the file, before loader filtering."""
    return len(pd.read_csv(path, **kw))


HDR = ["series", "source file", "freq", "units", "first", "last", "rows", "used"]
rows = []


# ---------------------------------------------------------------- US
y = cc.load_yields()
a, b = span(y)
rows.append(["GSW zero-coupon 1-5y", "gsw_yields_daily.csv", "daily", "percent p.a.",
             a, b, raw_rows(cc.DATA_DIR / "gsw_yields_daily.csv"), len(y)])

ff = cc.load_ff5_all()
a, b = span(ff)
rows.append(["Fama-French 5 factors", "F-F_Research_Data_5_Factors_2x3_daily.csv",
             "daily", "percent", a, b, len(ff), len(ff)])

legs = cc.load_ff_legs()
a, b = span(legs)
rows.append(["6 size-B/M portfolios", "6_Portfolios_2x3_Daily.csv", "daily", "percent",
             a, b, len(legs), len(legs)])

spr = sp.load_value_spread(6)
a, b = span(spr)
rows.append(["BE/ME characteristic", "Portfolios_Formed_on_BE-ME.csv", "annual",
             "log ratio", a, b, len(spr), len(spr)])


# ---------------------------------------------------------------- international
CURVES = [("UK gilt curve 1-5y", "yields_UK.csv", "daily", ccp.load_boe),
          ("Canada curve 1-5y", "yields_CA.csv", "daily", ccp.load_canada),
          ("Japan JGB curve 1-5y", "yields_JP.csv", "daily", ccp.load_japan),
          ("Germany curve 1-5y", "yields_DE.csv", "monthly", ccp.load_germany)]
for name, fname, freq, loader in CURVES:
    d = loader()
    a, b = span(d)
    rows.append([name, f"intl/{fname}", freq, "percent p.a.", a, b, len(d), len(d)])

for cty in ci.MARKETS:
    h = ci.load_aqr_hml(ci.HMLCODE[cty])
    a, b = span(h)
    rows.append([f"{cty} value factor", "intl/hml_daily_intl.csv", "daily", "percent",
                 a, b, raw_rows(cc.DATA_DIR / "intl" / "hml_daily_intl.csv"), len(h)])

cc.table(HDR, rows,
         note="'rows' counts what the file holds, 'used' what the loader returns. GSW carries\n"
              "blank rows on US market holidays; the four value factors share one wide file with\n"
              "NaNs preserved, so each market drops on its own date.")


# ---------------------------------------------------------------- sample end dates
cc.h2("Where each sample stops")
us = st.run_strategy(cc.build_cp_realtime(cc.month_end_yields(y), min_obs=120),
                     cc.load_ff5(), "QE", 4, z_minpts=12, nw_lags=4,
                     start="1975-01-01")[0]
cc.kv([("GSW yields end", span(y)[1]),
       ("US value factor ends", span(ff)[1]),
       ("foreign value factors end", span(ci.load_aqr_hml("GBR"))[1]),
       ("BE/ME characteristic ends", span(spr)[1]),
       ("US traded quarters", f"{us['n']} ({us['start']} to {us['end']})")])
