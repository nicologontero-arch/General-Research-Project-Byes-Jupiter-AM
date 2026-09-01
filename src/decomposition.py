"""Timing/tilt/cost decomposition with t-stats. US, every design.

    E[active] = Cov(w, h) + (E[w] - 1)*E[h] - cost*E[turnover]
                 timing        static tilt        costs

The claim under test is Cov(w, h) alone. The tilt is a static long-value
position, and a regression alpha separates neither, since a linear beta absorbs
timing curvature. The identity has no fitted parameter.

Each term is the mean of a per-period series, tested at the usual HAC lags
(12 monthly / 4 quarterly / 2 annual). `t(slope)` reaches the timing null from
the HAC regression of h on w. Section 1b bootstraps the timing term.

Traded frames come from the runners unchanged. Arithmetic on realised returns.
"""
import numpy as np
import pandas as pd

import cp_core as cc
import strategy as st
import spread as sp

cc.report("decomposition", "Timing vs tilt decomposition, with t-stats",
          "E[active] = Cov(w,h) + (E[w]-1)E[h] - costs; the exact split behind "
          "both alpha definitions")

FREQS = [("ME", 12, "Monthly", 36, 12), ("QE", 4, "Quarterly", 12, 4), ("YE", 1, "Annual", 3, 2)]
SUBS = [("1975-1989", "1975-01-01", "1989-12-31"), ("1990-2004", "1990-01-01", "2004-12-31"),
        ("2005-2019", "2005-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", "2026-12-31")]
START = "1975-01-01"
COST = st.COST

y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
cp = cc.build_cp_realtime(ym, min_obs=120)

HDR = ["label", "n", "meanW", "TIMING %/yr", "t", "t(slope)", "tilt %/yr", "t(W=1)",
       "cost %/yr", "total %/yr", "t", "|resid|"]


def row(label, df, ppy, lags):
    turn = (df["w_lag"] - df["w_prev"]).abs()
    d = cc.decompose(df["w_lag"], df["hml"], turn, ppy, lags, cost=COST)
    return [label, d["n"], f"{d['mean_w']:.3f}",
            f"{d['timing']:+.2f}", f"{d['t_timing']:+.2f}", f"{d['t_timing_slope']:+.2f}",
            f"{d['tilt']:+.2f}", f"{d['t_tilt']:+.2f}", f"{d['cost']:+.2f}",
            f"{d['total']:+.2f}", f"{d['t_total']:+.2f}", f"{abs(d['resid']):.1e}"], d


cc.note("timing = Cov(w,h). tilt = (E[w]-1)E[h]. total = the benchmark-difference alpha.")
cc.note("|resid| is total minus the three terms, and is zero to machine precision.")


# =========================================================== 1
cc.h2("1) CP-only overlay  w = 1 + tanh(z_CP),  by frequency (1975+, net 10bps)")
rows, keep = [], {}
for freq, ppy, name, zmin, lags in FREQS:
    _, df = st.run_strategy(cp, ff, freq, ppy, z_minpts=zmin, nw_lags=lags, start=START)
    keep[name] = (df, ppy, lags)
    rw, d = row(name, df, ppy, lags)
    rows.append(rw)
cc.table(HDR, rows,
         note="All terms annualised percent. HAC lags 12 monthly, 4 quarterly, 2 annual.")


cc.h2("1b) Block bootstrap on the timing term (quarterly)")
cc.note("Resamples 4-quarter blocks and recomputes Cov(w,h) on each draw.")
def boot_timing(w, h, ppy, B=5000, L=4, seed=7):
    """Block bootstrap of Cov(w, h), annualised percent.

    Seeded per call, so a design's draw does not depend on what ran before it."""
    rng = np.random.default_rng(seed)
    wv, hv = np.asarray(w, float), np.asarray(h, float)
    n = len(wv)
    out = np.empty(B)
    for b in range(B):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend(range(s, min(s + L, n)))
        ws, hs = wv[idx[:n]], hv[idx[:n]]
        out[b] = np.mean((ws - ws.mean()) * (hs - hs.mean()))
    return out * ppy * 100


dfq, ppyq, lagsq = keep["Quarterly"]
bt = boot_timing(dfq["w_lag"], dfq["hml"], ppyq)
_, dq = row("Quarterly", dfq, ppyq, lagsq)
lo, hi = np.percentile(bt, [2.5, 97.5])
cc.kv([("timing term (point estimate)", f"{dq['timing']:+.2f} %/yr"),
       ("HAC t", f"{dq['t_timing']:+.2f}"),
       ("block-bootstrap 95% CI", f"[{lo:+.2f}, {hi:+.2f}] %/yr"),
       ("P(timing <= 0)", f"{(bt <= 0).mean():.3f}")],
      title="5,000 block bootstraps, block length 4 quarters")

# Block length 2, not 4. On 51 annual observations a 4-year block leaves ~13
# effective blocks and a narrower interval, which is a small-sample artefact.
# Two years keeps ~25 blocks and gives the conservative answer.
ANN_BLOCK = 2
dfa, ppya, lagsa = keep["Annual"]
bta = boot_timing(dfa["w_lag"], dfa["hml"], ppya, L=ANN_BLOCK)
_, da = row("Annual", dfa, ppya, lagsa)
loa, hia = np.percentile(bta, [2.5, 97.5])
cc.kv([("timing term (point estimate)", f"{da['timing']:+.2f} %/yr"),
       ("HAC t", f"{da['t_timing']:+.2f}"),
       ("block-bootstrap 95% CI", f"[{loa:+.2f}, {hia:+.2f}] %/yr"),
       ("P(timing <= 0)", f"{(bta <= 0).mean():.3f}")],
      title=f"The same, annual holding period, block length {ANN_BLOCK} years")


# =========================================================== 2
cc.h2("2) CP-only overlay, quarterly sub-periods")
rows = []
for name, s, e in SUBS:
    _, df = st.run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4, start=s, end=e)
    rows.append(row(name, df, 4, 4)[0])
cc.table(HDR, rows,
         note="The four blocks partition the sample.")


# =========================================================== 2b
cc.h2("2b) 1990-2004 split at the dot-com peak")
_, df9004 = st.run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4,
                            start="1990-01-01", end="2004-12-31")
rows = []
for name, s, e in [("1990-1999", "1990-01-01", "1999-12-31"),
                   ("2000-2004", "2000-01-01", "2004-12-31")]:
    _, dfx = st.run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4, start=s, end=e)
    rows.append(row(name, dfx, 4, 4)[0])
cc.table(HDR, rows)

# A wealth ratio, not a sum of quarterly active returns: the arithmetic sum the
# decomposition is denominated in is not the change in relative wealth. The
# full-sample minimum settles whether the timed position is ever behind.
_, df_all = st.run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4, start=START)
ratio = (1 + df_all["timed"]).cumprod() / (1 + df_all["static"]).cumprod()


def at(year):
    return ratio[ratio.index.year == year].iloc[-1]


act = (df9004["timed"] - df9004["static"]) * 100
cc.kv([("timed / static wealth at end-1989", f"{at(1989):.2f}"),
       ("  at end-1999", f"{at(1999):.2f}"),
       ("  at end-2001 (the trough of the episode)", f"{at(2001):.2f}"),
       ("  at end-2004", f"{at(2004):.2f}"),
       ("minimum of the ratio over the whole sample", f"{ratio.min():.3f}"),
       ("  ever below 1 (i.e. ever behind the benchmark)?", str(bool((ratio < 1).any()))),
       ("worst quarter", f"{act.idxmin().date()}"),
       ("  its active return", f"{act.min():+.1f}%"),
       ("  weight held into it", f"{df9004['w_lag'].loc[act.idxmin()]:.2f}"),
       ("  HML that quarter", f"{df9004['hml'].loc[act.idxmin()] * 100:+.1f}%")],
      title="Timed wealth divided by static wealth, quarterly compounding")


# =========================================================== 3
cc.h2("3) The seven CP x spread designs, quarterly")
cp_q = cp.resample("QE").last().dropna()
c_raw = np.tanh(cc.expanding_z(cp_q, min_pts=12))
hml_q = st.period_hml(ff, "QE")
spread_a = sp.load_value_spread(6)
_, pct_raw = sp.quarterly_percentile(spread_a, c_raw.index)
base = pd.DataFrame({"c": c_raw, "p": pct_raw, "hml": hml_q}).dropna()
W = sp.weights(base["c"], base["p"])
ORDER = ["CP-only", "Spread-only", "Additive", "Gated", "A monotonic", "B U-shaped", "S agreement"]

rows, dsn = [], {}
for name in ORDER:
    _, dfx = sp.evaluate(W[name], hml_q, start=START)
    rw, d = row(name, dfx, 4, 4)
    rows.append(rw); dsn[name] = (d, dfx)
cc.table(HDR, rows,
         note="The timing share of each design's gross alpha is in the next table.")

rows = []
for name in ORDER:
    d, _ = dsn[name]
    gross = d["timing"] + d["tilt"]
    share = d["timing"] / gross * 100 if abs(gross) > 1e-9 else float("nan")
    rows.append([name, f"{d['mean_w']:.3f}", f"{d['timing']:+.2f}", f"{d['tilt']:+.2f}",
                 f"{share:5.0f}%"])
cc.table(["design", "meanW", "timing %/yr", "tilt %/yr", "timing share of gross"], rows,
         title="Timing and tilt by design")

cc.h2("3b) S agreement, quarterly sub-periods")
rows = []
for name, s, e in SUBS:
    _, dfx = sp.evaluate(W["S agreement"], hml_q, start=s, end=e)
    rows.append(row(name, dfx, 4, 4)[0])
cc.table(HDR, rows)

cc.h2("3c) Block bootstrap on the timing term, CP-only vs S agreement (quarterly)")
rows = []
for name in ["CP-only", "S agreement"]:
    d, dfx = dsn[name]
    b = boot_timing(dfx["w_lag"], dfx["hml"], 4)
    lo, hi = np.percentile(b, [2.5, 97.5])
    rows.append([name, f"{d['timing']:+.2f}", f"{d['t_timing']:+.2f}",
                 f"[{lo:+.2f}, {hi:+.2f}]", f"{(b <= 0).mean():.3f}"])
cc.table(["design", "timing %/yr", "HAC t", "95% CI", "P(timing <= 0)"], rows,
         note="5,000 block bootstraps, block length 4 quarters.")
