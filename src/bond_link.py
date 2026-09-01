"""Does real-time CP predict bonds in shape, if not in level?

    LEVEL   R2_oos = 1 - SSE(CP) / SSE(expanding mean)
    SHAPE   Cov( tanh(z_CP,t), rx_bond,t+12 )

The strategy consumes tanh((CP - mu)/sigma), invariant to any affine transform
of CP, so the two tests are near-orthogonal and can disagree.

Nothing is fitted. rx(t) runs from t to t+12, so the monthly panel overlaps:
NW lags are 18, and a December-only panel is reported beside it.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import cp_core as cc

cc.report("bond_link", "Real-time CP against bond returns: level and shape",
          "out-of-sample R2 on the point forecast, and Cov(tanh(z_CP), 12m bond excess return)")

MIN_OBS, ZMIN, LAGS = 120, 36, 18

ym = cc.month_end_yields(cc.load_yields())
cp = cc.build_cp_realtime(ym, min_obs=MIN_OBS)
rx = cc.avg_excess_return_12m(ym)          # return from t to t+12, dated at t
z = cc.expanding_z(cp, min_pts=ZMIN)
c = np.tanh(z)

d = pd.DataFrame({"cp": cp, "c": c, "rx": rx}).dropna()
cc.note(f"Evaluation sample: {len(d)} months, {d.index[0].date()} -> {d.index[-1].date()}")
cc.note("rx = realised average 12m excess return on 2-5y bonds, dated at the month the")
cc.note("position is opened.")


# =========================================================== 1
cc.h2("1) Level and shape tests on one sample")
months = ym.index
realized_by = months.to_series().shift(-12).values
yv = rx.values
bench = pd.Series(index=months, dtype=float)
for t in range(len(months)):
    elig = [s for s in range(t + 1)
            if (not pd.isna(yv[s])) and (not pd.isna(realized_by[s]))
            and realized_by[s] <= np.datetime64(months[t])]
    if len(elig) < MIN_OBS:
        continue
    bench.iloc[t] = np.mean(yv[elig])

ev = pd.DataFrame({"cp": cp, "bench": bench, "rx": rx}).dropna()
r2 = 1 - ((ev["rx"] - ev["cp"]) ** 2).sum() / ((ev["rx"] - ev["bench"]) ** 2).sum()
ev90 = ev[ev.index >= "1990-01-01"]
r2_90 = 1 - ((ev90["rx"] - ev90["cp"]) ** 2).sum() / ((ev90["rx"] - ev90["bench"]) ** 2).sum()

g = (d["c"] - d["c"].mean()) * (d["rx"] - d["rx"].mean())   # covariance influence function
cov, t_cov = cc.newey_west_t(g.values, LAGS)
m = sm.OLS(d["rx"], sm.add_constant(d["c"])).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})

cc.table(["test", "what it measures", "statistic"],
         [["R2_oos, full", "level and scale of the point forecast", f"{r2:+.4f}"],
          ["R2_oos, 1990+", "level and scale, 1990 onward", f"{r2_90:+.4f}"],
          ["Cov(tanh z, rx)", "standardised shape, the input to the weight",
           f"{cov:+.4f} (NW t {t_cov:+.2f})"],
          ["slope of rx on tanh z", "the same null in regression form",
           f"{m.params.iloc[1]:+.3f} (NW t {m.tvalues.iloc[1]:+.2f})"]])


# =========================================================== 2
cc.h2("2) Bond returns by CP bucket")
dd = d.copy()
dd["bucket"] = pd.cut(dd["c"], [-1.01, -0.5, 0.0, 0.5, 1.01],
                      labels=["deep low CP", "low CP", "high CP", "deep high CP"])
rows = []
for k, gp in dd.groupby("bucket", observed=True):
    rows.append([str(k), len(gp), f"{gp['rx'].mean():+.3f}", f"{gp['c'].mean():+.2f}"])
cc.table(["CP bucket", "n months", "next-12m bond excess return (%)", "mean tanh(z)"], rows,
         note="Overlapping 12m windows, so n is not the effective sample size.")

hi, lo = dd[dd["c"] > 0]["rx"], dd[dd["c"] <= 0]["rx"]
spread = hi.mean() - lo.mean()
cc.kv([("mean 12m bond excess return when CP above its own history", f"{hi.mean():+.3f}%"),
       ("mean when CP below", f"{lo.mean():+.3f}%"),
       ("spread", f"{spread:+.3f}%"),
       ("unconditional mean", f"{d['rx'].mean():+.3f}%"),
       ("share of months CP is above", f"{(dd['c'] > 0).mean():.2f}")])


# =========================================================== 3
cc.h2("3) Non-overlapping check, December only")
dec = d[d.index.month == 12]
gd = (dec["c"] - dec["c"].mean()) * (dec["rx"] - dec["rx"].mean())
cov_d, t_d = cc.newey_west_t(gd.values, 2)
md = sm.OLS(dec["rx"], sm.add_constant(dec["c"])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
cc.kv([("non-overlapping observations", len(dec)),
       ("Cov(tanh z, rx)", f"{cov_d:+.4f}  (NW t {t_d:+.2f})"),
       ("slope of rx on tanh z", f"{md.params.iloc[1]:+.3f}  (NW t {md.tvalues.iloc[1]:+.2f})"),
       ("R2", f"{md.rsquared:.3f}")],
      title="December only: one observation per year, 12m horizon")

