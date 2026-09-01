"""US CP-timed value: frequency, sub-periods, weight buckets, bootstrap CI."""
import numpy as np, pandas as pd, cp_core as cc, strategy as st
cc.report("results", "Problem 1 - US CP-timed value",
          "frequency comparison, quarterly sub-periods, tilt-vs-timing decomposition, look-ahead audit + bootstrap")

MIN_OBS, Z_MIN = 120, 36
y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
cp = cc.build_cp_realtime(ym, min_obs=MIN_OBS)
cc.note(f"Real-time CP: {len(cp)} obs, {cp.index[0].date()} -> {cp.index[-1].date()}")

FREQS = [("ME", 12, "Monthly", 36, 12), ("QE", 4, "Quarterly", 12, 4), ("YE", 1, "Annual", 3, 2)]
HDR = ["period", "n", "start", "end", "staticSR", "timedSR", "alpha%/yr", "IR", "NW t", "meanW", "corr(a,v)", "turn/yr"]

def row(r):
    return [r["label"], r["n"], r["start"], r["end"], f"{r['static_SR']:.2f}", f"{r['timed_SR']:.2f}",
            f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:.2f}", f"{r['NW_t']:.2f}", f"{r['mean_w']:.3f}",
            f"{r['corr_active_value']:+.2f}", f"{r['avg_turnover_per_yr']:.2f}"]

cc.h2("1) Frequency comparison (start 1975, calendar-consistent 3y z-burn-in)")
fdfs = {}; rows = []
for freq, ppy, name, zmin, lags in FREQS:
    r, df = st.run_strategy(cp, ff, freq, ppy, z_minpts=zmin, nw_lags=lags, label=name, start="1975-01-01")
    fdfs[name] = (df, ppy); rows.append(row(r))
cc.table(HDR, rows)

cc.h2("2) Quarterly sub-periods")
SUBS = [("1975-1989","1975-01-01","1989-12-31"),("1990-2004","1990-01-01","2004-12-31"),
        ("2005-2019","2005-01-01","2019-12-31"),("2020-2026","2020-01-01","2026-12-31")]
rows = []
for name, s, e in SUBS:
    r, _ = st.run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4, label=name, start=s, end=e); rows.append(row(r))
cc.table(HDR, rows)

cc.h2("3) Tilt-vs-timing decomposition + asymmetry (quarterly)")
df, ppy = fdfs["Quarterly"]
ow = df[df["w_lag"] > 1.0]; uw = df[df["w_lag"] < 1.0]
df["bucket"] = pd.cut(df["w_lag"], [-0.01, 0.5, 1.0, 1.5, 2.01], labels=["deep-UW","UW","OW","deep-OW"])
cond = df.groupby("bucket", observed=True)["hml"].mean() * ppy * 100
cc.kv([("mean weight (~1 => no static tilt)", f"{df['w_lag'].mean():.3f}"),
       ("corr(active, value)", f"{np.corrcoef(df['active'], df['static'])[0,1]:+.2f}"),
       ("active alpha OVERWEIGHT", f"{ow['active'].mean()*ppy*100:+.2f}%/yr (n={len(ow)})"),
       ("active alpha UNDERWEIGHT", f"{uw['active'].mean()*ppy*100:+.2f}%/yr (n={len(uw)})")])
cc.table(["weight bucket", "next-qtr value return (ann.%)"], [[k, f"{v:+.2f}"] for k, v in cond.items()],
         title="Next-quarter value return by weight bucket")

cc.h2("4) Look-ahead audit + bootstrap (quarterly)")
chk = fdfs["Quarterly"][0]
rng = np.random.default_rng(7); act = chk["active"].values; nq = len(act)
def boot(x, B=5000, L=4):
    out = np.empty(B)
    for b in range(B):
        idx = []
        while len(idx) < nq:
            s = rng.integers(0, nq); idx.extend(range(s, min(s+L, nq)))
        out[b] = x[idx[:nq]].mean()
    return out
bm = boot(act) * 4 * 100; lo, hi = np.percentile(bm, [2.5, 97.5])
cc.kv([("corr(w_lag_t, hml_t)  (timing skill, weight set at t-1)", f"{np.corrcoef(chk['w_lag'], chk['hml'])[0,1]:+.3f}"),
       ("block-bootstrap quarterly alpha 95% CI", f"[{lo:+.2f}, {hi:+.2f}] %/yr"),
       ("P(alpha <= 0)", f"{(bm<=0).mean():.3f}")])
