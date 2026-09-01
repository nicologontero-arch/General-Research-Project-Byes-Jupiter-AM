"""Regression alpha beside the benchmark-difference alpha.

    alpha_diff  = E[r_timed] - E[r_HML]                    beta forced to 1
    r_timed(t)  = alpha_reg + beta*r_HML(t) + e(t)         beta estimated, HAC
    alpha_diff - alpha_reg = (beta - 1)*E[r_HML]

Covers the CP-only overlay, the seven CP x spread designs and the FF5 spec.
Return series come from the strategy runners unchanged; nothing feeds back into
a signal.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import cp_core as cc
import strategy as st
import spread as sp

cc.report("factor_alpha", "Factor-model alpha - strategy return regressed on HML",
          "r_strategy = alpha + beta * r_HML + e (Newey-West), against the "
          "benchmark-difference alpha reported everywhere else")

FREQS = [("ME", 12, "Monthly", 36, 12), ("QE", 4, "Quarterly", 12, 4), ("YE", 1, "Annual", 3, 2)]
SUBS = [("1975-1989", "1975-01-01", "1989-12-31"), ("1990-2004", "1990-01-01", "2004-12-31"),
        ("2005-2019", "2005-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", "2026-12-31")]
START = "1975-01-01"

y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
cp = cc.build_cp_realtime(ym, min_obs=120)


# ----------------------------------------------------------------- machinery
factor_reg = cc.factor_reg      # the regression lives in cp_core

HDR = ["label", "n", "alpha_reg %/yr", "NW t(a)", "beta", "NW t(b-1)", "R2",
       "alpha_diff %/yr", "diff - reg"]


def row(label, r):
    return [label, r["n"], f"{r['alpha_reg']:+.2f}", f"{r['t_alpha']:+.2f}",
            f"{r['beta']:+.3f}", f"{r['t_beta_1']:+.2f}", f"{r['r2']:.3f}",
            f"{r['alpha_diff']:+.2f}", f"{r['gap']:+.2f}"]


cc.note("alpha_reg is the regression intercept, beta estimated. alpha_diff is mean(timed) -")
cc.note("mean(HML), beta forced to 1. Both annualised percent. The last column is their gap,")
cc.note("which equals (beta - 1) * E[r_HML].")


# =========================================================== 1  CP-only
cc.h2("1) CP-only overlay  w = 1 + tanh(z_CP),  by frequency (1975+, net 10bps)")
rows, keep = [], {}
for freq, ppy, name, zmin, lags in FREQS:
    res, df = st.run_strategy(cp, ff, freq, ppy, z_minpts=zmin, nw_lags=lags, start=START)
    keep[name] = (df, ppy, lags, res)
    rows.append(row(name, factor_reg(df["timed"], df["static"], ppy, lags)))
cc.table(HDR, rows,
         note="beta is the beta of the timed portfolio on HML. NW t(b-1) tests beta = 1.")
cc.kv([(f"E[HML] on the {name} sample",
        f"{factor_reg(df['timed'], df['static'], ppy, lags)['mean_hml']:+.2f} %/yr")
       for name, (df, ppy, lags, _) in keep.items()],
      title="E[r_HML] on each sample, the multiplier in the reconciliation")


# =========================================================== 2  identity
cc.h2("2) Identity check: the active return regressed on HML")
cc.note("active = timed - hml, so regressing active on hml returns the same intercept with")
cc.note("beta lower by exactly 1. The check column reports both differences.")
rows = []
for name, (df, ppy, lags, _) in keep.items():
    rt = factor_reg(df["timed"], df["static"], ppy, lags)
    ra = factor_reg(df["active"], df["static"], ppy, lags)
    rows.append([name, f"{rt['alpha_reg']:+.4f}", f"{ra['alpha_reg']:+.4f}",
                 f"{abs(rt['alpha_reg'] - ra['alpha_reg']):.2e}",
                 f"{rt['beta']:+.4f}", f"{ra['beta']:+.4f}",
                 f"{abs((rt['beta'] - 1) - ra['beta']):.2e}",
                 "PASS" if abs(rt["alpha_reg"] - ra["alpha_reg"]) < 1e-10
                 and abs((rt["beta"] - 1) - ra["beta"]) < 1e-10 else "FAIL"])
cc.table(["period", "alpha (timed)", "alpha (active)", "|d alpha|",
          "beta (timed)", "beta (active)", "|d(beta-1)|", "check"], rows)


# =========================================================== 3  sub-periods
cc.h2("3) CP-only overlay, quarterly sub-periods")
rows = []
for name, s, e in SUBS:
    _, df = st.run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4, start=s, end=e)
    rows.append(row(name, factor_reg(df["timed"], df["static"], 4, 4)))
cc.table(HDR, rows,
         note="Quarterly, net 10bps. Sub-periods partition the sample.")


# =========================================================== 4  spread designs
cc.h2("4) The seven CP x spread designs, quarterly")
cc.note("All seven designs share one sample: the spread starts later than CP and its rolling")
cc.note("percentile needs 40 quarters.")
cp_q = cp.resample("QE").last().dropna()
c_raw = np.tanh(cc.expanding_z(cp_q, min_pts=12))
hml_q = st.period_hml(ff, "QE")
spread_a = sp.load_value_spread(6)
_, pct_raw = sp.quarterly_percentile(spread_a, c_raw.index)
base = pd.DataFrame({"c": c_raw, "p": pct_raw, "hml": hml_q}).dropna()
c, pct = base["c"], base["p"]
W = sp.weights(c, pct)
ORDER = ["CP-only", "Spread-only", "Additive", "Gated", "A monotonic", "B U-shaped", "S agreement"]

rows, dfs = [], {}
for name in ORDER:
    _, dfx = sp.evaluate(W[name], hml_q, start=START)
    dfs[name] = dfx
    rows.append(row(name, factor_reg(dfx["timed"], dfx["static"], 4, 4)))
cc.table(HDR, rows,
         note="On the spread's common sample, so the CP-only row differs from section 1's.")

cc.h2("4b) S agreement, quarterly sub-periods")
rows = []
for name, s, e in SUBS:
    _, dfx = sp.evaluate(W["S agreement"], hml_q, start=s, end=e)
    rows.append(row(name, factor_reg(dfx["timed"], dfx["static"], 4, 4)))
cc.table(HDR, rows,
         note="Quarterly only: the spread percentile uses a 60-quarter rolling window.")


# =========================================================== 5  FF5
cc.h2("5) Five-factor spec (quarterly): r = a + b1 Mkt-RF + b2 SMB + b3 HML + b4 RMW + b5 CMA")
cc.note("Quarterly factors, compounded from daily. NW 4 lags.")
ff_all = cc.load_ff5_all()
FCOLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
fq = pd.DataFrame({col: ff_all[col].resample("QE").apply(cc.compound) for col in FCOLS}).dropna()

rows = []
for tag, series in [("CP-only", keep["Quarterly"][0]["timed"]),
                    ("S agreement", dfs["S agreement"]["timed"])]:
    d = pd.concat([series.rename("r"), fq], axis=1, sort=False).dropna()
    m = sm.OLS(d["r"], sm.add_constant(d[FCOLS])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    rows.append([tag, int(m.nobs), f"{m.params['const'] * 400:+.2f}", f"{m.tvalues['const']:+.2f}"]
                + [f"{m.params[cnm]:+.3f}" for cnm in FCOLS] + [f"{m.rsquared:.3f}"])
cc.table(["strategy", "n", "alpha %/yr", "NW t(a)"] + FCOLS + ["R2"], rows,
         note="alpha is annualised percent; loadings are on quarterly factor returns.")
