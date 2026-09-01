"""Duration against business cycle, by predictive regression. Not the strategy.

Regress 12m-ahead bond / growth-leg / value-leg / HML returns on CP. Duration
predicts b_growth > b_value and b_HML < 0; the business cycle predicts the
reverse. Also the raw HML-on-CP regression.

Sections 1-2 use the full-sample fitted CP, the convention for descriptive
predictive regressions; no position is taken on their output. Sections 2c and 3
repeat on the real-time factor. Tradeable results elsewhere use
cc.build_cp_realtime only."""
import numpy as np, pandas as pd, warnings
import statsmodels.api as sm, cp_core as cc
warnings.filterwarnings("ignore")
cc.report("duration_test", "Duration vs business-cycle predictive regressions",
          "12m-ahead bond, growth-leg, value-leg and HML returns regressed on CP, the leg contrast, and a horizon profile")

ff = cc.load_ff5(); ym = cc.month_end_yields(cc.load_yields()); legs = cc.load_ff_legs()

# CP factor: full-sample fitted bond risk premium
f = cc.forwards(ym); rxb = cc.avg_excess_return_12m(ym)
dcp = pd.concat([f, rxb.rename("rx")], axis=1).dropna()
mcp = sm.OLS(dcp["rx"], sm.add_constant(dcp[["f1","f2","f3","f4","f5"]])).fit()
CP = (mcp.params["const"] + f[["f1","f2","f3","f4","f5"]].mul(mcp.params[["f1","f2","f3","f4","f5"]]).sum(axis=1))
cc.kv([("CP construction R2 (forwards -> 12m bond excess return)", f"{mcp.rsquared:.3f}"),
       ("coefs f1..f5", "  ".join(f"{mcp.params[c]:+.2f}" for c in ['f1','f2','f3','f4','f5']))])
CPz = (CP - CP.mean()) / CP.std()

def m_comp(s): return s.resample("ME").apply(lambda x: (np.prod(1+x.dropna().values/100)-1)*100)
hml_m = m_comp(ff["HML"]); H = m_comp(legs["H"]) - m_comp(ff["RF"]); L = m_comp(legs["L"]) - m_comp(ff["RF"])
def fwd12(s, idx): s = s.reindex(idx); cs = s.cumsum(); return cs.shift(-12) - cs
idx = ym.index
tgts = {"12m BONDS (excess)": cc.avg_excess_return_12m(ym),
        "12m GROWTH leg (excess)": fwd12(L, idx),
        "12m VALUE leg (excess)": fwd12(H, idx),
        "12m HML (value-growth)": fwd12(hml_m, idx)}

cc.h2("Predictive regression on standardized CP (per 1 SD), NW 18 lags")
rows = []
for name, tg in tgts.items():
    dd = pd.concat([CPz.rename("CP"), tg.rename("y")], axis=1).dropna()
    m = sm.OLS(dd["y"], sm.add_constant(dd["CP"])).fit(cov_type="HAC", cov_kwds={"maxlags": 18})
    rows.append([name, f"{m.params['CP']:+.2f}", f"{m.tvalues['CP']:.2f}", f"{m.rsquared:.3f}"])
cc.table(["target", "beta/SD", "NW t", "R2"], rows,
         note="Loadings are per standard deviation of CP. NW 18 lags for the 12m overlap.")

# ----------------------------------------------------------------------------
# 2) The discriminating restriction: the sign of b_value - b_growth.
#
# The legs enter linearly, so the contrast is itself a regression: VALUE - GROWTH
# is H - L, and regressing it on CP delivers the contrast with its own standard
# error. The hypotheses are directional, so the tests are one-sided against
# duration, which predicts the contrast <= 0.
# ----------------------------------------------------------------------------
from scipy import stats as _st

NW = {"cov_type": "HAC", "cov_kwds": {"maxlags": 18}}


def fit(y, x, **kw):
    dd = pd.concat([x.rename("CP"), y.rename("y")], axis=1).dropna()
    return sm.OLS(dd["y"], sm.add_constant(dd["CP"])).fit(**(kw or NW))


def one_sided_p(t):
    """P(observing a contrast this positive | duration true, i.e. beta <= 0)."""
    return 1.0 - _st.norm.cdf(t)


def two_sided_p(t):
    """The conventional null beta = 0, beside the directional one."""
    return 2.0 * (1.0 - _st.norm.cdf(abs(t)))


cc.h2("2) The leg contrast b_v - b_g")
cc.note("The contrast VALUE - GROWTH is H - L, so it is itself a regressand and carries its "
        "own standard error.")

contrast = (fwd12(H, idx) - fwd12(L, idx)).rename("contrast")
specs = [("GROWTH leg  b_g", fwd12(L, idx)),
         ("VALUE leg   b_v", fwd12(H, idx)),
         ("CONTRAST    b_v - b_g", contrast)]
rows = []
for name, tg in specs:
    m = fit(tg, CPz)
    b, t = m.params["CP"], m.tvalues["CP"]
    rows.append([name, f"{b:+.2f}", f"{t:+.2f}", f"{one_sided_p(t):.3f}",
                 f"{two_sided_p(t):.3f}"])
cc.table(["specification", "beta/SD", "NW t", "one-sided p vs duration",
          "two-sided p"], rows,
         note="One-sided p is P(beta <= 0); two-sided p tests beta = 0 against any "
              "alternative. The contrast row is the test, the two leg rows its components.")

m_c = fit(contrast, CPz)
m_g = fit(fwd12(L, idx), CPz)
cc.kv([("contrast b_v - b_g (per 1 SD of CP)", f"{m_c.params['CP']:+.3f} pp over 12m"),
       ("NW t on the contrast", f"{m_c.tvalues['CP']:+.2f}"),
       ("one-sided p against the duration null", f"{one_sided_p(m_c.tvalues['CP']):.3f}"),
       ("growth leg, two-sided p (is it moving at all?)",
        f"{2 * (1 - _st.norm.cdf(abs(m_g.tvalues['CP']))):.3f}")],
      title="The contrast, restated")

# 2c) The same contrast on the real-time factor.
cc.h2("2c) The same contrast, using the real-time CP factor instead")
cc.note("Sections 1 and 2 fit CP once over the full sample. Here CP is rebuilt by expanding, "
        "realised-only regression and the contrast re-estimated on it.")
CPrt = cc.build_cp_realtime(ym)
CPrtz = cc.expanding_z(CPrt, min_pts=36)   # 3y burn-in, as everywhere
rows = []
for name, tg in specs:
    m = fit(tg, CPrtz)
    b, t = m.params["CP"], m.tvalues["CP"]
    rows.append([name, int(m.nobs), f"{b:+.2f}", f"{t:+.2f}", f"{one_sided_p(t):.3f}",
                 f"{two_sided_p(t):.3f}"])
cc.table(["specification", "n", "beta/SD", "NW t", "one-sided p vs duration",
          "two-sided p"], rows,
         note="Real-time CP, standardised by an expanding z-score. Shorter sample than section 2.")

# ----------------------------------------------------------------------------
# 3) Horizon profile.
#
# CP is fitted to a 12-month-ahead bond return, so power should build toward 12
# months. The cumulative loading rises with h mechanically; the annualised
# loading and the t are the columns that carry information.
# ----------------------------------------------------------------------------
cc.h2("3) Horizon profile, h = 1 to 24 months")
cc.note("CP forecasts the 12m-ahead bond return, so h=12 is its own horizon. The cumulative "
        "loading grows with h by construction; the annualised column and the t do not.")


def fwd_h(s, index, h):
    """Cumulative return from t+1 to t+h. The regressor stays dated t."""
    s = s.reindex(index)
    cs = s.cumsum()
    return cs.shift(-h) - cs


rows = []
for h in (1, 3, 6, 12, 24):
    lags = max(1, int(1.5 * h))          # overlapping h-period returns
    m = fit(fwd_h(hml_m, idx, h), CPz,
            cov_type="HAC", cov_kwds={"maxlags": lags})
    b, t = m.params["CP"], m.tvalues["CP"]
    rows.append([f"{h}m", int(m.nobs), lags, f"{b:+.2f}", f"{b * 12 / h:+.2f}",
                 f"{t:+.2f}", f"{m.rsquared:.3f}",
                 "CP's fitting horizon" if h == 12 else ""])
cc.table(["horizon", "n", "NW lags", "cumulative beta/SD", "annualised beta/SD", "NW t", "R2", "note"],
         rows,
         note="Overlapping beyond h=1, so n overstates the effective sample; NW lags are 1.5h.")

# raw (non-standardised) HML-on-CP regression
cs = hml_m.reindex(idx).cumsum(); hml_12 = cs.shift(-12) - cs
d = pd.concat([CP.rename("CP"), hml_12.rename("y")], axis=1).dropna()
m = sm.OLS(d["y"], sm.add_constant(d["CP"])).fit(cov_type="HAC", cov_kwds={"maxlags": 18})
cc.h2("Raw regression  r_HML(t+1:t+12) = a + b*CP(t) + e  [percent]")
cc.kv([("a (intercept)", f"{m.params['const']:+.3f}  (t {m.tvalues['const']:+.2f})"),
       ("b (slope on CP)", f"{m.params['CP']:+.3f}  (t {m.tvalues['CP']:+.2f})"),
       ("R2", f"{m.rsquared:.4f}"), ("n", int(m.nobs))])
