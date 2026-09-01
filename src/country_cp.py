"""Foreign CP factors. Each loader returns SVENY01..SVENY05 (1-5y, percent),
daily except Germany, so cp_core's construction applies unchanged. Curves come
from data/intl/yields_*.csv, written by prepare_data.py.

float_precision='round_trip' is required: the default parser drops the last bit
of a float64 and shifts every foreign result in its final decimal."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import cp_core as cc
warnings.filterwarnings("ignore")
INTL = cc.DATA_DIR / "intl"


def _yields(market):
    df = pd.read_csv(INTL / f"yields_{market}.csv", index_col=0, parse_dates=True,
                     float_precision="round_trip")
    df.index.name = None
    return df


def load_boe():
    """UK: Bank of England nominal gilt zero-coupon spot curve (1979+)."""
    return _yields("UK")


def load_canada():
    """Canada: Bank of Canada zero-coupon yields (1986+, decimals -> percent)."""
    return _yields("CA")


def load_japan():
    """Japan: MoF JGB constant-maturity yields (1974+, coupon-yield proxy)."""
    return _yields("JP")


def load_germany():
    """Germany: Bundesbank Svensson zero-coupon spot 1-5y (monthly, 1972+)."""
    return _yields("DE")


def diagnose(daily_pct, name, min_obs=120):
    """One row: yields span, full-sample CP R2, tent coefs, real-time CP span."""
    ym = cc.month_end_yields(daily_pct)
    f = cc.forwards(ym); rx = cc.avg_excess_return_12m(ym)
    d = pd.concat([f, rx.rename("rx")], axis=1).dropna()
    m = sm.OLS(d["rx"], sm.add_constant(d[["f1", "f2", "f3", "f4", "f5"]])).fit()
    cp = cc.build_cp_realtime(ym, min_obs=min_obs)
    coefs = " ".join(f"{m.params[c]:+.2f}" for c in ['f1','f2','f3','f4','f5'])
    return [name, f"{daily_pct.index[0].date()}..{daily_pct.index[-1].date()}", len(daily_pct),
            f"{m.rsquared:.3f}", int(m.nobs), coefs,
            f"{cp.index[0].date()}..{cp.index[-1].date()}", len(cp)]


if __name__ == "__main__":
    cc.report("country_cp", "Country CP factors: construction & validation",
              "full-sample CP R2, forward-rate coefficients and real-time CP span per market")
    cc.note("US anchor: full-sample CP R2 = 0.153")
    HDR = ["market", "yields span", "obs", "CP R2", "n", "coefs f1..f5", "real-time CP span", "CP obs"]
    rows = [diagnose(load_boe(),     "UK (BoE gilt zero-coupon, 1979+)"),
            diagnose(load_canada(),  "Canada (BoC zero-coupon, 1986+)"),
            diagnose(load_japan(),   "Japan (MoF JGB constant-maturity, 1974+; coupon proxy)"),
            diagnose(load_germany(), "Germany (Bundesbank Svensson zero-coupon, 1972+)")]
    cc.table(HDR, rows)
