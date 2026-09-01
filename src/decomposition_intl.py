"""Timing/tilt decomposition, international.

Foreign mean weight runs 0.86 to 1.25 against 1.00 in the US, so a standing tilt
sits inside every foreign benchmark-difference and regression alpha. Against
value factors paying 6-10%/yr that tilt is worth 1-2%/yr on its own.

  1  is the drift a burn-in artefact, or small-sample?
  1c US curve truncated to each foreign start date, whole chain rebuilt
  2  the decomposition per market
  3  Cov(w, h) is invariant to a constant in w, so the timing term is immune
  4  a real-time recentred signal, as contrast

Sections 1-3 build no signal. The recentring in 4 subtracts an expanding mean
through t, so it stays causal; it is a diagnostic, not adopted.
"""
import warnings

import numpy as np
import pandas as pd

import cp_core as cc
import strategy as st
import cp_intl as ci

warnings.filterwarnings("ignore")

cc.report("decomposition_intl", "Timing vs tilt decomposition - international",
          "mean weight, the timing/tilt split per market, and a recentred variant")

PPY, ZMIN, LAGS = 4, 12, 4
COST = st.COST

us_y = cc.load_yields()
us_cp = cc.build_cp_realtime(cc.month_end_yields(us_y), min_obs=120)
ffus = cc.load_ff5()
FF = {c: pd.DataFrame({"HML": ci.load_aqr_hml(ci.HMLCODE[c])}) for c in ci.MARKETS}
ALL = [("US", us_cp, ffus, "1975-01-01")] + [(c, ci.CP[c], FF[c], None) for c in ci.MARKETS]


# =========================================================== 1
cc.h2("1) Mean weight by market")
cc.note("The design intends E[w] = 1: an expanding z centres on zero, so E[tanh z] ~ 0.")
rows = []
for name, cp_c, _, _ in ALL:
    cpq = cp_c.resample("QE").last().dropna()
    z = cc.expanding_z(cpq, min_pts=ZMIN)
    n3 = len(z) // 3
    drift = np.polyfit(np.arange(len(cpq)), cpq.values, 1)[0] * PPY
    rows.append([name, len(cpq), f"{z.mean():+.3f}", f"{np.tanh(z).mean():+.3f}",
                 f"{z.iloc[:n3].mean():+.3f}", f"{z.iloc[-n3:].mean():+.3f}", f"{drift:+.3f}"])
cc.table(["market", "n CP qtrs", "mean z", "E[tanh z]", "mean z, first third",
          "mean z, last third", "CP drift /yr"], rows,
         note="z is the expanding z-score of each market's own CP. CP drift is the OLS slope\n"
              "of CP on time, per year. First and last thirds split the CP quarters evenly.")

cc.h2("1b) Mean weight against the z-score burn-in")
rows = []
for name, cp_c, ff_c, start in ALL:
    cells = [name]
    for zmin in [12, 24, 40]:
        r, _ = st.run_strategy(cp_c, ff_c, "QE", PPY, z_minpts=zmin, nw_lags=LAGS, start=start)
        cells += [f"{r['mean_w']:.3f}", f"{r['alpha_ann_pct']:+.2f}"]
    rows.append(cells)
cc.table(["market", "meanW z12", "alpha z12", "meanW z24", "alpha z24", "meanW z40", "alpha z40"],
         rows,
         note="z12 / z24 / z40 are the z-score burn-in, in quarters.")


cc.h2("1c) US curve truncated to each foreign start date")
cc.note("Truncate the US yield curve to each foreign market's start date and rebuild the whole")
cc.note("chain from there, then run the full 1961 history over the identical traded window.")

# Each foreign curve's first observation, from data/intl/yields_*.csv.
MATCH = [("Germany", "1972-09-30"), ("Japan", "1974-09-24"),
         ("UK", "1979-01-02"), ("Canada", "1986-01-02")]

SAT_HI, SAT_LO = 1.8, 0.2     # w = 1 + tanh(z): the ends of the map


def wstats(df):
    """Decomposition plus two shape statistics.

    'sat' is the share of traded quarters within 0.2 of the tanh bounds, where
    the signal has stopped discriminating."""
    turn = (df["w_lag"] - df["w_prev"]).abs()
    d = cc.decompose(df["w_lag"], df["hml"], turn, PPY, LAGS, cost=COST)
    w = df["w_lag"]
    d["sd_w"] = w.std()
    d["sat"] = float(((w > SAT_HI) | (w < SAT_LO)).mean())
    return d


rows = []
for tag, ystart in MATCH:
    ytr = us_y[us_y.index >= ystart]
    cp_short = cc.build_cp_realtime(cc.month_end_yields(ytr), min_obs=120)
    r_s, df_s = st.run_strategy(cp_short, ffus, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS)
    # Same traded window, full 1961 history behind the expanding mean.
    _, df_f = st.run_strategy(us_cp, ffus, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS,
                              start=r_s["start"])
    s, f = wstats(df_s), wstats(df_f)
    rows.append([f"{tag}-length ({ystart[:4]}+)", s["n"], r_s["start"],
                 f"{s['mean_w']:.3f}", f"{f['mean_w']:.3f}",
                 f"{100*s['sat']:.1f}%", f"{100*f['sat']:.1f}%",
                 f"{s['timing']:+.2f}", f"{f['timing']:+.2f}",
                 f"{s['t_timing']:+.2f}", f"{f['t_timing']:+.2f}"])
cc.table(["US history matched to", "n", "traded from", "meanW short", "meanW full",
          "satur. short", "satur. full", "timing short", "timing full",
          "t short", "t full"], rows,
         note="'short' is the US rebuilt from the foreign start date; 'full' is the US on its\n"
              "1961 history over the same quarters. Saturation is the share of traded quarters\n"
              "within 0.2 of the tanh bounds.")

def satrow(label, df):
    d = wstats(df)
    return [label, d["n"], f"{d['mean_w']:.3f}", f"{100*d['sat']:.1f}%",
            f"{d['timing']:+.2f}", f"{d['t_timing']:+.2f}"]


_, df_us = st.run_strategy(us_cp, ffus, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS,
                           start="1975-01-01")
rows = [satrow("US (full history)", df_us)]
for cty in ci.MARKETS:
    rows.append(satrow(cty, ci.run(cty)[1]))
cc.table(["market", "n", "meanW", "saturation", "timing %/yr", "t"], rows,
         note="Saturation is the share of traded quarters within 0.2 of the tanh bounds.")


# =========================================================== 2
HDR = ["market", "n", "meanW", "TIMING %/yr", "t", "t(slope)", "tilt %/yr", "t(W=1)",
       "cost %/yr", "total %/yr", "t", "|resid|"]


def row(label, df):
    turn = (df["w_lag"] - df["w_prev"]).abs()
    d = cc.decompose(df["w_lag"], df["hml"], turn, PPY, LAGS, cost=COST)
    return [label, d["n"], f"{d['mean_w']:.3f}",
            f"{d['timing']:+.2f}", f"{d['t_timing']:+.2f}", f"{d['t_timing_slope']:+.2f}",
            f"{d['tilt']:+.2f}", f"{d['t_tilt']:+.2f}", f"{d['cost']:+.2f}",
            f"{d['total']:+.2f}", f"{d['t_total']:+.2f}", f"{abs(d['resid']):.1e}"], d


cc.h2("2) Timing and tilt by market")
cc.note("Cov(w,h) is unchanged by adding a constant to w, so timing is invariant to mean weight.")
rows, dec = [], {}
for name, cp_c, ff_c, start in ALL:
    _, df = st.run_strategy(cp_c, ff_c, "QE", PPY, z_minpts=ZMIN, nw_lags=LAGS, start=start)
    rw, d = row(name, df)
    rows.append(rw); dec[name] = (d, df)
cc.table(HDR, rows,
         note="All terms annualised percent. |resid| is total minus the three terms.")


cc.h2("2b) First / second half by market, timing term only")
rows = []
for cty in ci.MARKETS:
    _, df = dec[cty]
    mid = df.index[len(df) // 2]
    for tag, s, e in [("H1", None, str((mid - pd.Timedelta(days=1)).date())),
                      ("H2", str(mid.date()), None)]:
        _, dfh = ci.run(cty, start=s, end=e)
        rows.append(row(f"{cty} {tag}", dfh)[0])
cc.table(HDR, rows)


# =========================================================== 3
cc.h2("3) Summary across the four non-US markets")
tim = [dec[c][0]["timing"] for c in ci.MARKETS]
tot = [dec[c][0]["total"] for c in ci.MARKETS]
tlt = [dec[c][0]["tilt"] for c in ci.MARKETS]
cc.kv([("mean TIMING term", f"{np.mean(tim):+.2f} %/yr"),
       ("positive on timing", f"{sum(t > 0 for t in tim)}/{len(ci.MARKETS)}"),
       ("markets with |t(timing)| > 2", f"{sum(abs(dec[c][0]['t_timing']) > 2 for c in ci.MARKETS)}/4"),
       ("mean tilt term", f"{np.mean(tlt):+.2f} %/yr"),
       ("mean total (cp_intl's alpha)", f"{np.mean(tot):+.2f} %/yr"),
       ("positive on total", f"{sum(t > 0 for t in tot)}/{len(ci.MARKETS)}"),
       ("US timing, for reference", f"{dec['US'][0]['timing']:+.2f} %/yr "
                                    f"(t {dec['US'][0]['t_timing']:+.2f})")])


# =========================================================== 4
cc.h2("4) Recentred signal, for contrast")
cc.note("w = 1 + [tanh(z) - expanding mean of tanh(z)], which forces E[w] toward 1.")
cc.note("A change to the strategy, not to the reporting. Not adopted.")


def run_recentred(cp_c, ff_c, start):
    """w = 1 + [tanh(z) - expanding mean of tanh(z)]. The mean at t uses history
    through t, the same convention as expanding_z."""
    cpq = cp_c.resample("QE").last().dropna()
    c = np.tanh(cc.expanding_z(cpq, min_pts=ZMIN))
    c = c - c.expanding(min_periods=ZMIN).mean()
    d = pd.DataFrame({"w": 1.0 + c, "hml": st.period_hml(ff_c, "QE")}).dropna()
    d["w_lag"] = d["w"].shift(1); d["w_prev"] = d["w"].shift(2)
    d = d.dropna()
    if start:
        d = d[d.index >= pd.Timestamp(start)]
    return d


rows = []
for name, cp_c, ff_c, start in ALL:
    d0 = dec[name][0]
    dr = row(name, run_recentred(cp_c, ff_c, start))[1]
    rows.append([name, f"{d0['mean_w']:.3f}", f"{dr['mean_w']:.3f}",
                 f"{d0['timing']:+.2f}", f"{dr['timing']:+.2f}",
                 f"{d0['tilt']:+.2f}", f"{dr['tilt']:+.2f}",
                 f"{d0['total']:+.2f}", f"{dr['total']:+.2f}", f"{dr['t_total']:+.2f}"])
cc.table(["market", "meanW base", "meanW recentred", "timing base", "timing recentred",
          "tilt base", "tilt recentred", "total base", "total recentred", "t"], rows,
         note="'base' is the shipped design, 'recentred' the variant above.")
