"""CP x value spread: alignment diagnostic, seven designs, mechanism, bootstrap."""
import numpy as np, pandas as pd, cp_core as cc, strategy as st, spread as sp
cc.report("p2_driver", "Problem 2 - CP x value-spread",
          "alignment diagnostic, seven weighting schemes, mechanism, bootstrap on the best design")

y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
cp = cc.build_cp_realtime(ym, min_obs=120)
cp_q = cp.resample("QE").last().dropna()
c = np.tanh(cc.expanding_z(cp_q, min_pts=12))
hml = st.period_hml(ff, "QE")
spread_a = sp.load_value_spread(6)
_, p = sp.quarterly_percentile(spread_a, c.index)
base = pd.DataFrame({"c": c, "p": p, "hml": hml}).dropna()
c, p = base["c"], base["p"]
cc.note(f"Common quarterly sample: {len(base)} obs {base.index[0].date()}..{base.index[-1].date()}")

cc.h2("1) Alignment diagnostic (CP vs spread)")
d = 2 * (p - 0.5)
agree = (np.sign(c) == np.sign(d)); extreme = p.apply(lambda x: x <= 0.2 or x >= 0.8)
cc.kv([("corr(CP signal c, spread pctile p)", f"{np.corrcoef(c,p)[0,1]:+.3f}"),
       ("sign-agreement overall", f"{agree.mean():.2f}"),
       ("sign-agreement in tails", f"{agree[extreme].mean():.2f}")])
terc = pd.qcut(p, 3, labels=["narrow","mid","wide"])
cc.table(["spread tercile", "mean CP signal"],
         [[k, f"{v:+.3f}"] for k, v in c.groupby(terc, observed=True).mean().items()],
         title="Mean CP signal by spread tercile")

cc.h2("2) Strategy comparison (quarterly, 1975+, net 10bps)")
W = sp.weights(c, p)
order = ["CP-only","Spread-only","Additive","Gated","A monotonic","B U-shaped","S agreement"]
res_all = {}; rows = []
for name in order:
    r, df = sp.evaluate(W[name], hml); res_all[name] = (r, df)
    rows.append([name, r["n"], f"{r['alpha']:+.2f}", f"{r['IR']:.2f}", f"{r['NWt']:.2f}",
                 f"{r['meanW']:.3f}", f"{r['turn']:.2f}", f"{r['corr_av']:+.2f}"])
cc.table(["scheme", "n", "alpha", "IR", "NWt", "meanW", "turn", "corr(a,v)"], rows)

cc.h2("3) Mechanism: next-qtr value return by CP direction x spread tail (ann.%)")
m = pd.DataFrame({"c": c, "p": p, "hml": hml}).dropna()
m["cp_dir"] = np.where(m["c"] > 0, "CP overweight", "CP underweight")
m["sp_tail"] = pd.cut(m["p"], [-.01, .33, .67, 1.01], labels=["narrow","mid","wide"])
tab = m.groupby(["cp_dir","sp_tail"], observed=True)["hml"].agg(["mean","count"]); tab["mean"] *= 400
cc.table(["CP direction", "spread tail", "next-qtr value (ann.%)", "n"],
         [[i[0], i[1], f"{r['mean']:+.2f}", int(r['count'])] for i, r in tab.iterrows()])
_, dfcp = res_all["CP-only"]
rows = []
for name in ["A monotonic","B U-shaped","S agreement"]:
    _, dfx = res_all[name]
    j = dfcp[["active"]].rename(columns={"active":"cp"}).join(dfx[["active"]].rename(columns={"active":"x"}), how="inner")
    pj = p.reindex(j.index); cj = c.reindex(j.index); agreej = (np.sign(cj) == np.sign(2*(pj-0.5)))
    inc = j["x"] - j["cp"]
    rows.append([name, f"{inc[agreej].mean()*400:+.2f}", f"{inc[~agreej].mean()*400:+.2f}"])
cc.table(["scheme", "agree (ann.%)", "conflict (ann.%)"], rows,
         title="Incremental active alpha vs CP-only, by CP/spread agreement")

cc.h2("3b) Return against risk")


def max_drawdown(r):
    """Worst peak-to-trough fall in compounded wealth, percent."""
    w = (1 + r).cumprod()
    return (w / w.cummax() - 1).min() * 100


rows = []
for name in ["CP-only", "Spread-only", "S agreement"]:
    r, d = res_all[name]
    rows.append([name, f"{r['alpha']:+.2f}",
                 f"{d['active'].std() * np.sqrt(4) * 100:.2f}",
                 f"{r['IR']:.2f}",
                 f"{d['timed'].std() * np.sqrt(4) * 100:.2f}",
                 f"{r['timedSR']:.2f}",
                 f"{max_drawdown(d['timed']):.1f}"])
# The benchmark is identical across designs, so take it from any one of them.
_, d0 = res_all["CP-only"]
rows.append(["Static value (benchmark)", "0.00", "0.00", "n/a",
             f"{d0['static'].std() * np.sqrt(4) * 100:.2f}",
             f"{res_all['CP-only'][0]['staticSR']:.2f}",
             f"{max_drawdown(d0['static']):.1f}"])
cc.table(["design", "alpha %/yr", "active vol %/yr", "IR",
          "traded vol %/yr", "traded SR", "max drawdown %"], rows,
         title="Return and risk, quarterly, 1975+, net 10bps")


cc.h2("4) Bootstrap on best-IR multiplicative design")
best = max(["A monotonic","B U-shaped","S agreement"], key=lambda k: res_all[k][0]["IR"])
act = res_all[best][1]["active"].values; nq = len(act); rng = np.random.default_rng(7)
def boot(x, B=5000, L=4):
    out = np.empty(B)
    for b in range(B):
        idx = []
        while len(idx) < nq:
            s = rng.integers(0, nq); idx.extend(range(s, min(s+L, nq)))
        out[b] = x[idx[:nq]].mean()
    return out
bm = boot(act) * 400; lo, hi = np.percentile(bm, [2.5, 97.5])
cc.kv([("best-IR multiplicative design", best),
       ("alpha 95% CI", f"[{lo:+.2f}, {hi:+.2f}] %/yr"),
       ("P(alpha <= 0)", f"{(bm<=0).mean():.3f}"),
       ("CP-only IR vs best IR", f"{res_all['CP-only'][0]['IR']:.2f} vs {res_all[best][0]['IR']:.2f}")])
