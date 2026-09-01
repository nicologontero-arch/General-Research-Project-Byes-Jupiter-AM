"""The four CP-only robustness checks, re-run on the S-agreement design.

    w = 1 + c * (1 + LAM * d * sign(c))
    c = tanh(z_CP)          CP signal, in [-1, +1]
    d = 2 * (p - 0.5)       spread signal, in [-1, +1]
    p                       rank of the log BE/ME spread over 60 quarters
    LAM = 0.75              multiplier lands in [0.25, 1.75]

d*sign(c) is positive on agreement and negative on conflict, so the multiplier
scales conviction only. CP alone sets the side.

  1  truncation      the whole chain rebuilt on cut inputs, spread included
  2  rolling stability
  3  timing vs tilt
  4  spanning        against the yield curve, then against its own two inputs
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import cp_core as cc
import strategy as st
import spread as sp

cc.report("spread_checks", "S agreement: the same four checks",
          "truncation, rolling stability, timing versus tilt, and spanning "
          "against both the yield curve and its own two ingredients")

PPY, ZMIN, LAGS, START = 4, 12, 4, "1975-01-01"
DESIGN = "S agreement"

y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
spread_a = sp.load_value_spread(6)

cc.note("w = 1 + c (1 + 0.75 d sign(c)),  c = tanh(z_CP),  d = 2(p - 0.5)")
cc.note("The multiplier 1 + 0.75 d sign(c) lies in [0.25, 1.75] and scales conviction only.")


def build_S(ym_t, ff_t, spread_t, start=START, end=None):
    """Rebuild the whole S-agreement chain from raw inputs. Shared by the
    baseline and every truncated replay, so any difference is the data."""
    cp_t = cc.build_cp_realtime(ym_t, min_obs=120)
    c_t = np.tanh(cc.expanding_z(cp_t.resample("QE").last().dropna(), min_pts=ZMIN))
    hml_t = st.period_hml(ff_t, "QE")
    _, p_t = sp.quarterly_percentile(spread_t, c_t.index)
    b = pd.DataFrame({"c": c_t, "p": p_t, "hml": hml_t}).dropna()
    W_t = sp.weights(b["c"], b["p"])
    return sp.evaluate(W_t[DESIGN], hml_t, start=start, end=end)


res, df = build_S(ym, ff, spread_a)
cc.note(f"Baseline: n={res['n']}, {res['start']} to {res['end']}, "
        f"alpha {res['alpha']:+.2f}%/yr, IR {res['IR']:.2f}, NW t {res['NWt']:.2f}")


# =========================================================== 1
cc.h2("1) Real-time truncation test")
cc.note("Rebuild CP, the spread percentile, the weight and the traded return using only data")
cc.note("available up to T. The spread is truncated as well as the yield curve.")
rows = []
for T in ["2000-12-31", "2010-12-31", "2018-12-31"]:
    ym_t = ym[ym.index <= T]
    ff_t = ff[ff.index <= T]
    sp_t = spread_a[spread_a.index <= T]
    _, df_t = build_S(ym_t, ff_t, sp_t)
    common = df.index.intersection(df_t.index)
    d_w = float(np.nanmax(np.abs(df.loc[common, "w_lag"].values - df_t.loc[common, "w_lag"].values)))
    d_a = float(np.nanmax(np.abs(df.loc[common, "active"].values - df_t.loc[common, "active"].values)))
    rows.append([T, len(df_t), len(common), f"{d_w:.2e}", f"{d_a:.2e}",
                 "PASS" if max(d_w, d_a) < 1e-10 else "FAIL"])
cc.table(["truncated at T", "quarters kept", "overlap", "max |d weight|",
          "max |d active|", "verdict"], rows,
         note="Zero difference means no information dated after T influences any value before it.")


# =========================================================== 2
cc.h2("2) Rolling stability")
cc.note("Cov(w,h) on a moving 20-quarter window, annualised percent, against CP-only.")
ROLLQ = 20
roll_S = df["w_lag"].rolling(ROLLQ).cov(df["hml"], ddof=0).dropna() * PPY * 100

_, df_cp = st.run_strategy(cc.build_cp_realtime(ym, min_obs=120), ff, "QE", PPY,
                           z_minpts=ZMIN, nw_lags=LAGS, start=START)
roll_cp = df_cp["w_lag"].rolling(ROLLQ).cov(df_cp["hml"], ddof=0).dropna() * PPY * 100

cc.table(["design", "windows", "share positive", "worst", "best", "median"],
         [[nm, len(r), f"{(r > 0).mean():.0%}", f"{r.min():+.2f}", f"{r.max():+.2f}",
           f"{r.median():+.2f}"]
          for nm, r in [("CP-only", roll_cp), (DESIGN, roll_S)]],
         note="Windows overlap, so the share is descriptive and not a significance statement.")


# =========================================================== 3
cc.h2("3) Timing versus tilt")
turn = (df["w_lag"] - df["w_prev"]).abs()
D = cc.decompose(df["w_lag"], df["hml"], turn, PPY, LAGS, cost=sp.COST)
turn_cp = (df_cp["w_lag"] - df_cp["w_prev"]).abs()
Dcp = cc.decompose(df_cp["w_lag"], df_cp["hml"], turn_cp, PPY, LAGS, cost=st.COST)
cc.table(["design", "mean w", "timing %/yr", "t", "tilt %/yr", "t(w=1)", "cost %/yr",
          "total %/yr", "timing share", "|resid|"],
         [[nm, f"{d['mean_w']:.3f}", f"{d['timing']:+.2f}", f"{d['t_timing']:+.2f}",
           f"{d['tilt']:+.2f}", f"{d['t_tilt']:+.2f}", f"{d['cost']:+.2f}",
           f"{d['total']:+.2f}",
           f"{d['timing'] / (d['timing'] + d['tilt']) * 100:.0f}%", f"{abs(d['resid']):.1e}"]
          for nm, d in [("CP-only", Dcp), (DESIGN, D)]],
         note="All terms annualised percent. timing share = timing / (timing + tilt).")


# =========================================================== 4
cc.h2("4a) Spanning against the yield curve")
cc.note("Slope, level and curvature timed through identical machinery, then S agreement's")
cc.note("active return regressed on theirs.")
SIGNALS = {
    "Slope  y5 - y1": (ym["SVENY05"] - ym["SVENY01"]).dropna(),
    "Level  y1": ym["SVENY01"].dropna(),
    "Curvature 2y2-y1-y5": (2 * ym["SVENY02"] - ym["SVENY01"] - ym["SVENY05"]).dropna(),
}
act = {}
for nm, sig in SIGNALS.items():
    _, d_ = st.run_strategy(sig, ff, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS, start=START)
    act[nm] = d_["active"]
act[DESIGN] = df["active"]
A = pd.DataFrame(act).dropna()
X = A[list(SIGNALS)]
m = sm.OLS(A[DESIGN], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})
cc.kv([("S agreement active return, raw", f"{A[DESIGN].mean() * PPY * 100:+.2f} %/yr"),
       ("not explained by slope, level, curvature", f"{m.params['const'] * PPY * 100:+.2f} %/yr"),
       ("NW t on that intercept", f"{m.tvalues['const']:+.2f}"),
       ("R2 of S on the three controls", f"{m.rsquared:.3f}"),
       ("n quarters", int(m.nobs))],
      title="Spanning regression against the yield curve")

cc.h2("4b) Spanning against its own two ingredients")
cc.note("S agreement's active return regressed on the active returns of its two inputs.")
cp_q = cc.build_cp_realtime(ym, min_obs=120).resample("QE").last().dropna()
c_q = np.tanh(cc.expanding_z(cp_q, min_pts=ZMIN))
hml_q = st.period_hml(ff, "QE")
_, p_q = sp.quarterly_percentile(spread_a, c_q.index)
base = pd.DataFrame({"c": c_q, "p": p_q, "hml": hml_q}).dropna()
W = sp.weights(base["c"], base["p"])
ing = {}
for nm in ["CP-only", "Spread-only"]:
    _, d_ = sp.evaluate(W[nm], hml_q, start=START)
    ing[nm] = d_["active"]
ing[DESIGN] = df["active"]
B = pd.DataFrame(ing).dropna()
m2 = sm.OLS(B[DESIGN], sm.add_constant(B[["CP-only", "Spread-only"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": LAGS})
cc.kv([("not explained by CP-only and spread-only", f"{m2.params['const'] * PPY * 100:+.2f} %/yr"),
       ("NW t on that intercept", f"{m2.tvalues['const']:+.2f}"),
       ("loading on CP-only", f"{m2.params['CP-only']:+.3f}  (t {m2.tvalues['CP-only']:+.2f})"),
       ("loading on spread-only", f"{m2.params['Spread-only']:+.3f}  (t {m2.tvalues['Spread-only']:+.2f})"),
       ("R2", f"{m2.rsquared:.3f}"), ("n quarters", int(m2.nobs))],
      title="Spanning regression against CP-only and spread-only")
