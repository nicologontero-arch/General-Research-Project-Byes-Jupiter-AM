"""Rolling-window CP against the expanding baseline (US)."""
import numpy as np, pandas as pd, cp_core as cc, strategy as st
cc.report("rolling_cp", "Rolling-window CP vs expanding baseline (US)",
          "rolling-window CP re-estimation against the expanding baseline, 120/180/240 months")

y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
cp_exp = cc.build_cp_realtime(ym, min_obs=120)

FREQS = [("ME",12,"Monthly",36,12),("QE",4,"Quarterly",12,4),("YE",1,"Annual",3,2)]
SUBS = [("1975-1989","1975-01-01","1989-12-31"),("1990-2004","1990-01-01","2004-12-31"),
        ("2005-2019","2005-01-01","2019-12-31"),("2020-2026","2020-01-01","2026-12-31")]

def battery(cp, tag):
    cc.h2(tag)
    rows = []
    for freq, ppy, nm, zmin, lags in FREQS:
        r, _ = st.run_strategy(cp, ff, freq, ppy, z_minpts=zmin, nw_lags=lags, start="1975-01-01")
        rows.append([nm, r["n"], f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}",
                     f"{r['NW_t']:+.2f}", f"{r['mean_w']:.2f}", f"{r['avg_turnover_per_yr']:.2f}"])
    cc.table(["frequency","n","alpha%/yr","IR","NW t","meanW","turn/yr"], rows, title="Frequency (start 1975)")
    rows = []
    for nm, s, e in SUBS:
        r, _ = st.run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4, start=s, end=e)
        rows.append([nm, r["n"], f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}", f"{r['NW_t']:+.2f}"])
    cc.table(["sub-period","n","alpha%/yr","IR","NW t"], rows, title="Quarterly sub-periods")

cc.h2("Rolling-window sensitivity (quarterly, 1975+)")
rows = []
for W in [120, 180, 240]:
    cpr = cc.build_cp_rolling(ym, window_months=W, min_obs=96)
    r, _ = st.run_strategy(cpr, ff, "QE", 4, z_minpts=12, nw_lags=4, start="1975-01-01")
    rows.append([f"{W}m", r["n"], f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}", f"{r['NW_t']:+.2f}"])
cc.table(["window","n","alpha%/yr","IR","NW t"], rows)

battery(cp_exp, "Expanding CP (baseline)")
battery(cc.build_cp_rolling(ym, 180, 96), "Rolling CP, 15-year window")
