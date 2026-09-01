"""Robustness checks.

  1  is CP just the curve slope?   competing signals + spanning regression
  2  multiple testing              White's Reality Check over the 7 designs
  3  stub contamination            trailing partial period at the sample end
  4  cost sensitivity
"""
import numpy as np, pandas as pd, warnings
import statsmodels.api as sm
import cp_core as cc, strategy as st, spread as sp
warnings.filterwarnings("ignore")

cc.report("robustness", "Robustness checks",
          "yield-curve controls, multiple-testing adjustment, stub-period fix, cost sensitivity")

PPY, ZMIN, LAGS, START = 4, 12, 4, "1975-01-01"
y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
cp = cc.build_cp_realtime(ym, min_obs=120)


# =============================================================== 1
cc.h2("1) Competing term-structure signals")
cc.note("Each signal runs through identical machinery: expanding z, 1+tanh(z), lag, net 10bps.")

# No tent-proxy row: on GSW forwards the fitted coefficients do not form a tent
# (Rebonato and Nyholm 2025). cc.cp_proxy_tent remains available if wanted.
SIGNALS = {
    "CP (baseline)": cp,
    "Slope  y5 - y1": (ym["SVENY05"] - ym["SVENY01"]).dropna(),
    "Level  y1": ym["SVENY01"].dropna(),
    "Curvature 2y2-y1-y5": (2 * ym["SVENY02"] - ym["SVENY01"] - ym["SVENY05"]).dropna(),
}
rows, actives = [], {}
for name, sig in SIGNALS.items():
    r, d = st.run_strategy(sig, ff, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS, start=START)
    actives[name] = d["active"]
    rows.append([name, r["n"], f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}",
                 f"{r['NW_t']:+.2f}", f"{r['mean_w']:.3f}", f"{r['avg_turnover_per_yr']:.2f}"])
cc.table(["timing signal", "n", "alpha%/yr", "IR", "NW t", "meanW", "turn/yr"], rows,
         title="Each term-structure signal used as the timing signal")

A = pd.DataFrame(actives).dropna()
X = A[["Slope  y5 - y1", "Level  y1", "Curvature 2y2-y1-y5"]]
m = sm.OLS(A["CP (baseline)"], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})
cc.kv([("CP-timed alpha not explained by slope/level/curvature timing",
        f"{m.params['const'] * PPY * 100:+.2f} %/yr"),
       ("NW t on that intercept", f"{m.tvalues['const']:+.2f}"),
       ("R2 of CP active on the three controls", f"{m.rsquared:.3f}"),
       ("raw CP-timed alpha, for comparison", f"{A['CP (baseline)'].mean() * PPY * 100:+.2f} %/yr")],
      title="Spanning regression: CP active return on slope/level/curvature active returns")

cc.note("")
f = cc.forwards(ym)
hml_m = ff["HML"].resample("ME").apply(cc.compound) * 100
cs = hml_m.reindex(ym.index).cumsum(); hml12 = cs.shift(-12) - cs
zs = lambda s: (s - s.mean()) / s.std()
P = pd.DataFrame({"CP": zs(cp), "slope": zs(ym["SVENY05"] - ym["SVENY01"]),
                  "level": zs(ym["SVENY01"]), "y": hml12}).dropna()
rows = []
for tag, cols in [("CP alone", ["CP"]), ("slope alone", ["slope"]), ("level alone", ["level"]),
                  ("CP + slope + level", ["CP", "slope", "level"])]:
    mm = sm.OLS(P["y"], sm.add_constant(P[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 18})
    rows.append([tag, f"{mm.params['CP']:+.2f}" if "CP" in cols else "",
                 f"{mm.tvalues['CP']:+.2f}" if "CP" in cols else "",
                 f"{mm.params['slope']:+.2f}" if "slope" in cols else "",
                 f"{mm.params['level']:+.2f}" if "level" in cols else "",
                 f"{mm.rsquared:.3f}", int(mm.nobs)])
cc.table(["specification", "b(CP)/SD", "t(CP)", "b(slope)", "b(level)", "R2", "n"], rows,
         title="Predictive regression: 12m-ahead HML on standardized signals (NW 18 lags)")
cc.note("Two spanning tests are reported: the strategy form above (CP active return on the\n"
        "three control active returns) and the predictive form here (12m-ahead HML on the\n"
        "standardised signals). The strategy uses tanh(z), a nonlinear function of the curve.")


# =============================================================== 2
cc.h2("2) Multiple testing across the seven CP x spread designs")
cc.note("Selecting the best of 7 designs makes a naive p conditional on that selection. The\n"
        "Reality Check bootstraps the maximum statistic across all 7 under the null of no skill.")

cp_q = cp.resample("QE").last().dropna()
c = np.tanh(cc.expanding_z(cp_q, min_pts=ZMIN))
hml_q = st.period_hml(ff, "QE")
spread_a = sp.load_value_spread(6)
_, pct = sp.quarterly_percentile(spread_a, c.index)
base = pd.DataFrame({"c": c, "p": pct, "hml": hml_q}).dropna()
c, pct = base["c"], base["p"]
W = sp.weights(c, pct)
ORDER = ["CP-only", "Spread-only", "Additive", "Gated", "A monotonic", "B U-shaped", "S agreement"]

act = {}
for name in ORDER:
    r, d = sp.evaluate(W[name], hml_q)
    act[name] = d["active"]
AA = pd.DataFrame(act).dropna()
nq = len(AA)
means = AA.mean().values * PPY * 100
sds = AA.std().values * PPY * 100
tstats = AA.mean().values / (AA.std().values / np.sqrt(nq))

rng = np.random.default_rng(7)
B, L = 5000, 4
blocks = np.empty((B, nq), dtype=int)
for b in range(B):
    idx = []
    while len(idx) < nq:
        s0 = rng.integers(0, nq); idx.extend(range(s0, min(s0 + L, nq)))
    blocks[b] = idx[:nq]

V = AA.values
obs_max_t = tstats.max()
null_max_t = np.empty(B)
naive_p = np.zeros(len(ORDER))
for b in range(B):
    samp = V[blocks[b]]
    centred = samp - V.mean(axis=0)                      # impose the null
    tb = centred.mean(axis=0) / (samp.std(axis=0) / np.sqrt(nq))
    null_max_t[b] = tb.max()
    naive_p += (centred.mean(axis=0) >= V.mean(axis=0)).astype(float)
naive_p /= B
adj_p = (null_max_t >= obs_max_t).mean()

rows = []
for i, name in enumerate(ORDER):
    rows.append([name, f"{means[i]:+.2f}", f"{tstats[i]:+.2f}", f"{naive_p[i]:.3f}",
                 "<- selected" if tstats[i] == obs_max_t else ""])
cc.table(["design", "alpha%/yr", "t (iid)", "naive p", ""], rows,
         title=f"All seven designs (n={nq} quarters)",
         note="t (iid) is the plain mean/SE used to rank designs. 'naive p' is the one-sided\n"
              "bootstrap p for each design on its own, ignoring that six others were tried.")
cc.kv([("best design by t-stat", ORDER[int(np.argmax(tstats))]),
       ("its naive one-sided p", f"{naive_p[int(np.argmax(tstats))]:.3f}"),
       ("Reality Check p, adjusted for searching 7 designs", f"{adj_p:.3f}")],
      title="White Reality Check, 5000 block bootstraps (block length 4)")


# =============================================================== 3
cc.h2("3) Trailing partial-period ('stub') contamination")
last_ff = ff.index[-1]
cc.note(f"Daily HML ends {last_ff.date()}. resample() labels each period with its END date, so\n"
        f"the final quarterly bucket is stamped 2026-06-30 while holding only April, and the\n"
        f"final annual bucket is stamped 2026-12-31 while holding only Jan-Apr. The annual row\n"
        f"carries the project's largest t-stat, so this needs checking rather than assuming.")
rows = []
for freq, ppy, nm, zmin, lags in [("ME", 12, "Monthly", 36, 12), ("QE", 4, "Quarterly", 12, 4),
                                  ("YE", 1, "Annual", 3, 2)]:
    for tag, co in [("as published", False), ("complete periods only", True)]:
        r, _ = st.run_strategy(cp, ff, freq, ppy, z_minpts=zmin, nw_lags=lags,
                               start=START, complete_only=co)
        rows.append([f"{nm} - {tag}", r["n"], r["end"], f"{r['alpha_ann_pct']:+.2f}",
                     f"{r['IR']:+.2f}", f"{r['NW_t']:+.2f}"])
cc.table(["frequency / treatment", "n", "end", "alpha%/yr", "IR", "NW t"], rows,
         title="Headline frequency table, before and after dropping the trailing stub period")


# =============================================================== 4
cc.h2("4) Transaction-cost sensitivity")
cc.note("Headline results assume 10bps per unit of turnover; quarterly turnover is ~1.2x/yr.")
rows = []
for bps in [0, 10, 25, 50, 100]:
    r, _ = st.run_strategy(cp, ff, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS,
                           start=START, cost=bps / 10000.0)
    rC, _ = sp.evaluate(W["S agreement"], hml_q, cost=bps / 10000.0)
    rows.append([f"{bps} bps", f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}", f"{r['NW_t']:+.2f}",
                 f"{rC['alpha']:+.2f}", f"{rC['IR']:+.2f}", f"{rC['NWt']:+.2f}"])
cc.table(["cost per unit turnover", "CP alpha", "CP IR", "CP t",
          "S alpha", "S IR", "S t"], rows,
         title="Alpha and IR versus the cost assumption (quarterly, 1975+)")
be = []
for name, getter in [("CP-only", lambda bp: st.run_strategy(cp, ff, "QE", PPY, z_minpts=ZMIN,
                                                            nw_lags=LAGS, start=START,
                                                            cost=bp / 10000.0)[0]["alpha_ann_pct"]),
                     ("S agreement", lambda bp: sp.evaluate(W["S agreement"], hml_q,
                                                            cost=bp / 10000.0)[0]["alpha"])]:
    lo, hi = 0.0, 5000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if getter(mid) > 0: lo = mid
        else: hi = mid
    be.append((name, lo))
cc.kv([(f"{n} break-even cost", f"{v:,.0f} bps per unit turnover") for n, v in be],
      title="Cost at which the alpha is fully eaten")
