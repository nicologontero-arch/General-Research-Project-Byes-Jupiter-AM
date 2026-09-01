"""Placebo: CP fit to 12m-ahead HML instead of bond returns, real-time."""
import numpy as np, pandas as pd, warnings
import statsmodels.api as sm
import cp_core as cc, strategy as st
warnings.filterwarnings("ignore")
cc.report("hybrid", "Hybrid CP-HML placebo",
          "forwards fit directly to 12m-ahead HML instead of to bond returns, real-time")

def monthly_hml(ff):
    return ff["HML"].resample("ME").apply(lambda s: (np.prod(1+s.dropna().values/100)-1)*100)

def fwd12_hml(hml_m, idx):
    h = hml_m.reindex(idx); cs = h.cumsum(); return cs.shift(-12) - cs

def build_cp_hml_realtime(ym, hml_m, min_obs=120):
    f = cc.forwards(ym); target = fwd12_hml(hml_m, ym.index)
    months = ym.index; n = len(ym)
    realized_by = months.to_series().shift(-12).values
    Xd = np.column_stack([np.ones(n), f.values]); yv = target.values
    cp = pd.Series(index=months, dtype=float)
    for t in range(n):
        td = months[t]
        elig = [s for s in range(t+1) if (not pd.isna(yv[s])) and (not pd.isna(realized_by[s]))
                and realized_by[s] <= np.datetime64(td)]
        if len(elig) < min_obs: continue
        beta, *_ = np.linalg.lstsq(Xd[elig], yv[elig], rcond=None); cp.iloc[t] = Xd[t] @ beta
    return cp.dropna()

ff = cc.load_ff5(); ym = cc.month_end_yields(cc.load_yields()); hml_m = monthly_hml(ff)
f = cc.forwards(ym)
cc.h2("Diagnostic: full-sample regression of 12m target on 5 forwards")
rows = []
for name, tgt in [("bonds (standard CP)", cc.avg_excess_return_12m(ym)), ("HML (hybrid)", fwd12_hml(hml_m, ym.index))]:
    d = pd.concat([f, tgt.rename("y")], axis=1).dropna()
    m = sm.OLS(d["y"], sm.add_constant(d[["f1","f2","f3","f4","f5"]])).fit()
    rows.append([name, f"{m.rsquared:.3f}"] + [f"{m.params[c]:+.2f}" for c in ['f1','f2','f3','f4','f5']])
cc.table(["target","R2","f1","f2","f3","f4","f5"], rows)

cp_std = cc.build_cp_realtime(ym, min_obs=120)
cp_hyb = build_cp_hml_realtime(ym, hml_m, min_obs=120)
sig = lambda cp, f, z: np.tanh(cc.expanding_z(cp.resample(f).last().dropna(), min_pts=z))
j = pd.concat([sig(cp_std,"QE",12).rename("CP"), sig(cp_hyb,"QE",12).rename("h")], axis=1).dropna()
cc.kv([("corr(CP signal, hybrid signal), quarterly", f"{j['CP'].corr(j['h']):.2f}")])

cc.h2("HML timing: standard CP vs hybrid CP-HML (net 10bps, 1975+)")
for freq, ppy, nm, zmin, lags in [("QE",4,"Quarterly",12,4), ("YE",1,"Annual",3,2)]:
    rows = []
    for tag, cp in [("standard CP", cp_std), ("hybrid CP-HML", cp_hyb)]:
        r, _ = st.run_strategy(cp, ff, freq, ppy, z_minpts=zmin, nw_lags=lags, start="1975-01-01")
        rows.append([tag, r["n"], f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}",
                     f"{r['NW_t']:+.2f}", f"{r['mean_w']:.2f}"])
    cc.table(["design","n","alpha%/yr","IR","NW t","meanW"], rows, title=nm)
