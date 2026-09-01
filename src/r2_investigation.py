"""CP regression R2 by era and by sampling frequency."""
import numpy as np, pandas as pd
import statsmodels.api as sm, cp_core as cc
cc.report("r2_investigation", "CP regression R2 by era",
          "R2 of the forwards-on-12m-bond-return regression, by era and by sampling frequency")

ym = cc.month_end_yields(cc.load_yields())
f = cc.forwards(ym); rx = cc.avg_excess_return_12m(ym)
d = pd.concat([f, rx.rename("rx")], axis=1).dropna()

def r2(s, e):
    w = d[(d.index >= s) & (d.index <= e)]
    m = sm.OLS(w["rx"], sm.add_constant(w[["f1","f2","f3","f4","f5"]])).fit()
    return m.rsquared, int(m.nobs)

cc.h2("Forwards -> 12m avg bond excess return (overlapping monthly), R2 by era")
rows = []
for s, e, lab in [("1961","2026","full 1962-2025"),("1964","2003","CP05 window 1964-2003"),
                  ("1964","2008","1964-2008"),("1971","2003","1971-2003"),
                  ("1990","2026","1990-2026"),("2004","2026","post-2003 (ZIRP era)"),
                  ("2009","2021","2009-2021 ZIRP core")]:
    R, n = r2(s, e); rows.append([lab, f"{R:.3f}", n])
cc.table(["era", "R2", "n"], rows)

cc.h2("Annual (December-only, non-overlapping) regression")
dec = d[d.index.month == 12]
m = sm.OLS(dec["rx"], sm.add_constant(dec[["f1","f2","f3","f4","f5"]])).fit()
rows = [["full sample", f"{m.rsquared:.3f}", int(m.nobs), ""]]
decw = dec[(dec.index >= "1964") & (dec.index <= "2003")]
m = sm.OLS(decw["rx"], sm.add_constant(decw[["f1","f2","f3","f4","f5"]])).fit()
rows.append(["1964-2003", f"{m.rsquared:.3f}", int(m.nobs), "CP05 window"])
cc.table(["annual (Dec only)", "R2", "n", "window"], rows)
