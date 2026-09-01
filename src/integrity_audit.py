"""Integrity audit of the CP -> weight -> return chain.

Mean weight ~1.00 is what w = 1 + tanh(z) on an expanding z-score intends, and
also what a degenerate signal would produce. These six tests separate the two.

  A  signal distribution        does the signal move, or is it flat?
  B  alpha decomposition        timing vs tilt vs cost
  C  real-time truncation       rebuild on truncated data, compare
  D  placebo and lag decay      shuffled and over-lagged weights
  E  out-of-sample bond R2      against an expanding historical mean
  F  burn-in stability          alpha by first traded quarter
"""
import numpy as np, pandas as pd, warnings
import cp_core as cc, strategy as st
warnings.filterwarnings("ignore")

cc.report("integrity_audit", "Upstream integrity audit",
          "signal distribution, alpha decomposition, real-time truncation test, placebos, OOS bond R2")

PPY, ZMIN, LAGS, START = 4, 12, 4, "1975-01-01"
COST = st.COST

y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
cp = cc.build_cp_realtime(ym, min_obs=120)
res, df = st.run_strategy(cp, ff, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS, start=START)
cc.note(f"Baseline quarterly strategy: n={res['n']} {res['start']}..{res['end']}, "
        f"alpha={res['alpha_ann_pct']:+.2f}%/yr, IR={res['IR']:.2f}, NW t={res['NW_t']:.2f}")


# ------------------------------------------------------------------ A
cc.h2("A) Signal distribution")
cp_q = cp.resample("QE").last().dropna()
z_all = cc.expanding_z(cp_q, min_pts=ZMIN)
w_lag = df["w_lag"]
z = z_all.reindex(df.index)

cc.kv([("mean / SD of z", f"{z.mean():+.3f} / {z.std():.3f}"),
       ("mean / SD of weight", f"{w_lag.mean():.3f} / {w_lag.std():.3f}"),
       ("weight min .. max", f"{w_lag.min():.3f} .. {w_lag.max():.3f}"),
       ("share of quarters |z| > 1", f"{(z.abs() > 1).mean():.2f}"),
       ("share of quarters |w-1| > 0.5", f"{((w_lag - 1).abs() > 0.5).mean():.2f}"),
       ("weight AR(1)", f"{w_lag.autocorr(1):+.3f}")])
qs = [0, .1, .25, .5, .75, .9, 1.0]
cc.table(["quantile"] + [f"{int(q*100)}%" for q in qs],
         [["weight"] + [f"{w_lag.quantile(q):.2f}" for q in qs]],
         title="Weight distribution",
         note="Weight is 1 + tanh(z), so it is bounded in [0, 2] by construction.")


# ------------------------------------------------------------------ B
cc.h2("B) Alpha decomposition")
# active = (w_lag - 1) * hml - turnover * COST, so
# E[active] = Cov(w_lag, hml) + (E[w_lag] - 1) * E[hml] - COST * E[turnover]
turn = (df["w_lag"] - df["w_prev"]).abs()
cov_term = np.cov(w_lag, df["hml"], ddof=0)[0, 1] * PPY * 100
tilt_term = (w_lag.mean() - 1.0) * df["hml"].mean() * PPY * 100
cost_term = -COST * turn.mean() * PPY * 100
total = cov_term + tilt_term + cost_term
cc.table(["component", "ann. %", "share of gross"],
         [["timing  Cov(w, r)", f"{cov_term:+.3f}", f"{cov_term/(cov_term+tilt_term)*100:5.1f}%"],
          ["tilt    (E[w]-1) E[r]", f"{tilt_term:+.3f}", f"{tilt_term/(cov_term+tilt_term)*100:5.1f}%"],
          ["cost    -c E[turnover]", f"{cost_term:+.3f}", ""],
          ["total", f"{total:+.3f}", ""]],
         title="Alpha decomposition (quarterly, 1975+)")
cc.kv([("reported alpha", f"{res['alpha_ann_pct']:+.4f} %/yr"),
       ("decomposition total", f"{total:+.4f} %/yr"),
       ("identity holds (< 1e-8)", f"{abs(total - res['alpha_ann_pct']) < 1e-8}"),
       ("mean weight", f"{w_lag.mean():.4f}"),
       ("=> tilt contribution", f"{tilt_term:+.3f} %/yr of {res['alpha_ann_pct']:+.2f} "
                                f"({abs(tilt_term/res['alpha_ann_pct'])*100:.1f}%)")])


# ------------------------------------------------------------------ C
cc.h2("C) Real-time truncation test")
cc.note("Rebuild the chain using only data available up to T, then compare every value\n"
        "dated before T against the full-sample run.")
rows = []
for T in ["2000-12-31", "2010-12-31", "2018-12-31"]:
    ym_t = ym[ym.index <= T]
    cp_t = cc.build_cp_realtime(ym_t, min_obs=120)
    a = cp.reindex(cp_t.index)
    d_cp = float(np.nanmax(np.abs(a.values - cp_t.values)))
    zt = cc.expanding_z(cp_t.resample("QE").last().dropna(), min_pts=ZMIN)
    za = z_all.reindex(zt.index)
    d_z = float(np.nanmax(np.abs(za.values - zt.values)))
    ff_t = ff[ff.index <= T]
    _, df_t = st.run_strategy(cp_t, ff_t, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS, start=START)
    common = df.index.intersection(df_t.index)
    d_a = float(np.nanmax(np.abs(df.loc[common, "active"].values - df_t.loc[common, "active"].values)))
    rows.append([T, len(cp_t), f"{d_cp:.2e}", f"{d_z:.2e}", f"{d_a:.2e}",
                 "PASS" if max(d_cp, d_z, d_a) < 1e-10 else "FAIL"])
cc.table(["truncated at T", "CP obs", "max |dCP|", "max |dz|", "max |d active|", "verdict"], rows,
         note="Zero difference => no information from after T influences any value dated before T.")

# structural assertions on the estimation window
f_all = cc.forwards(ym); rx_all = cc.avg_excess_return_12m(ym)
months = ym.index; n = len(ym)
realized_by = months.to_series().shift(-12).values
yv = rx_all.values
worst_gap, checked = None, 0
for t in range(119, n):
    elig = [s for s in range(t + 1)
            if (not pd.isna(yv[s])) and (not pd.isna(realized_by[s]))
            and realized_by[s] <= np.datetime64(months[t])]
    if len(elig) < 120: continue
    gap = (months[t].to_period("M") - months[max(elig)].to_period("M")).n
    worst_gap = gap if worst_gap is None else min(worst_gap, gap)
    checked += 1
nan_tail = int(rx_all.tail(12).isna().sum())
cc.kv([("estimation dates checked", checked),
       ("smallest gap between last training obs and CP date", f"{worst_gap} months"),
       ("required (12m return must be realized)", ">= 12 months"),
       ("assertion max(elig) <= t-12", "PASS" if worst_gap >= 12 else "FAIL"),
       ("last 12 realized returns are NaN (unobservable)", f"{nan_tail}/12")],
      title="Estimation-window structure")


# ------------------------------------------------------------------ D
cc.h2("D) Placebo and lag-decay tests")
hml_q = st.period_hml(ff, "QE")
w_full = 1.0 + np.tanh(z_all)


def alpha_at_shift(k):
    """Alpha with the weight lagged k periods. k=1 is the baseline."""
    d = pd.DataFrame({"w": w_full, "hml": hml_q}).dropna()
    d["wk"] = d["w"].shift(k); d["wk1"] = d["w"].shift(k + 1)
    d = d.dropna()
    d = d[d.index >= pd.Timestamp(START)]
    to = (d["wk"] - d["wk1"]).abs()
    act = d["wk"] * d["hml"] - to * COST - d["hml"]
    _, t = cc.newey_west_t(act.values, LAGS)
    return act.mean() * PPY * 100, act.mean() / act.std() * np.sqrt(PPY), t, len(d)


rows = []
for k, tag in [(0, "k=0 CONTEMPORANEOUS (cheating)"), (1, "k=1 baseline (honest)"),
               (2, "k=2 extra lag"), (3, "k=3"), (4, "k=4 (= 1 year)")]:
    a, ir, t, nn = alpha_at_shift(k)
    rows.append([tag, nn, f"{a:+.2f}", f"{ir:+.2f}", f"{t:+.2f}"])
se = df["active"].std() / np.sqrt(len(df)) * PPY * 100
cc.table(["weight lag", "n", "alpha%/yr", "IR", "NW t"], rows,
         title="Alpha against the weight lag k",
         note=f"Baseline standard error is ~{se:.2f}%/yr. k=1 is the traded convention.")

rng = np.random.default_rng(11)
obs_alpha = res["alpha_ann_pct"]
perm = np.empty(2000); flip = np.empty(2000)
wl = w_lag.values; hm = df["hml"].values
for b in range(2000):
    p = rng.permutation(len(wl))
    perm[b] = ((wl[p] - 1) * hm).mean() * PPY * 100
    s = rng.choice([-1.0, 1.0], size=len(wl))
    flip[b] = ((1 + s * (wl - 1) - 1) * hm).mean() * PPY * 100
cc.table(["placebo", "mean alpha", "95% range", "observed", "pctile of observed"],
         [["weight permuted in time", f"{perm.mean():+.2f}",
           f"[{np.percentile(perm,2.5):+.2f}, {np.percentile(perm,97.5):+.2f}]",
           f"{obs_alpha:+.2f}", f"{(perm < obs_alpha).mean()*100:.1f}%"],
          ["weight deviation sign-randomised", f"{flip.mean():+.2f}",
           f"[{np.percentile(flip,2.5):+.2f}, {np.percentile(flip,97.5):+.2f}]",
           f"{obs_alpha:+.2f}", f"{(flip < obs_alpha).mean()*100:.1f}%"]],
         title="Randomisation placebos (2000 draws, gross of cost)",
         note="2000 draws, gross of cost. Each placebo breaks the weight-return pairing.")


# ------------------------------------------------------------------ E
cc.h2("E) Out-of-sample bond R2")
cc.note("Campbell-Thompson OOS R2 against an expanding historical-mean forecast, using at\n"
        "each date only returns already realised.")
bench = pd.Series(index=months, dtype=float)
for t in range(n):
    elig = [s for s in range(t + 1)
            if (not pd.isna(yv[s])) and (not pd.isna(realized_by[s]))
            and realized_by[s] <= np.datetime64(months[t])]
    if len(elig) < 120: continue
    bench.iloc[t] = np.mean(yv[elig])
ev = pd.DataFrame({"cp": cp, "bench": bench, "actual": rx_all}).dropna()
sse_cp = ((ev["actual"] - ev["cp"]) ** 2).sum()
sse_bm = ((ev["actual"] - ev["bench"]) ** 2).sum()
r2_oos = 1 - sse_cp / sse_bm
ev_late = ev[ev.index >= "1990-01-01"]
r2_late = 1 - ((ev_late["actual"] - ev_late["cp"]) ** 2).sum() / \
              ((ev_late["actual"] - ev_late["bench"]) ** 2).sum()
cc.kv([("evaluation months", len(ev)),
       ("span", f"{ev.index[0].date()} .. {ev.index[-1].date()}"),
       ("OOS R2, full evaluation sample", f"{r2_oos:+.4f}"),
       ("OOS R2, 1990+", f"{r2_late:+.4f}"),
       ("in-sample full-sample R2, for reference", "0.153")])


# ------------------------------------------------------------------ F
cc.h2("F) Burn-in stability: alpha by first traded quarter")
rows = []
for tag, s0 in [("1975+ baseline", "1975-01-01"), ("1980+ (drop first 5y)", "1980-01-01"),
                ("1985+ (drop first 10y)", "1985-01-01"), ("1990+ (drop first 15y)", "1990-01-01")]:
    r, _ = st.run_strategy(cp, ff, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS, start=s0)
    rows.append([tag, r["n"], f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}",
                 f"{r['NW_t']:+.2f}", f"{r['mean_w']:.3f}"])
cc.table(["sample", "n", "alpha%/yr", "IR", "NW t", "meanW"], rows)
