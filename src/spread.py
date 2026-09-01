"""Value spread (French BE/ME) and the seven CP x spread weighting designs.
CP sets direction, the spread sets conviction."""
import re
import numpy as np
import pandas as pd
import cp_core as cc

UPLOADS = cc.UPLOADS
COST = 0.001
GMIN, GMAX = 0.25, 1.75   # conviction envelope
LAM = 0.75                # agreement scaler; keeps S inside the envelope


def load_value_spread(known_month=6):
    """Annual log(BE/ME Hi10 / Lo10), dated at June, when it is public."""
    lines = open(f"{UPLOADS}/Portfolios_Formed_on_BE-ME.csv").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "Value Weight Average of BE / ME")
    cols = [c.strip() for c in lines[start + 1].split(",")]
    lo_i, hi_i = cols.index("Lo 10"), cols.index("Hi 10")
    rec = {}
    for l in lines[start + 2:]:
        if not re.match(r"^\s*\d{4},", l): break
        p = l.split(","); yr = int(p[0]); lo = float(p[lo_i]); hi = float(p[hi_i])
        if lo in (-99.99, -999) or hi in (-99.99, -999) or lo <= 0 or hi <= 0: continue
        rec[yr] = np.log(hi / lo)
    s = pd.Series(rec).sort_index()
    s.index = pd.to_datetime([f"{y}-{known_month:02d}-30" for y in s.index])
    return s


def quarterly_percentile(spread_annual, q_index, window_q=60, min_q=40):
    """Backward-looking 60-quarter percentile of the spread, on q_index."""
    full_q = pd.date_range(spread_annual.index[0], q_index[-1], freq="QE")
    lvl_full = spread_annual.reindex(spread_annual.index.union(full_q)).ffill().reindex(full_q)
    vals = lvl_full.values
    p_full = pd.Series(index=full_q, dtype=float)
    for i in range(len(full_q)):
        win = vals[max(0, i - window_q + 1):i + 1]; win = win[~np.isnan(win)]
        if len(win) < min_q or np.isnan(vals[i]): continue
        p_full.iloc[i] = (win <= vals[i]).mean()
    return lvl_full.reindex(q_index), p_full.reindex(q_index)


def weights(c, p):
    """c = tanh(z_CP), p = spread percentile. Returns one weight Series per
    design; benchmark is static HML at weight 1."""
    d = 2 * (p - 0.5)
    out = {}
    out["CP-only"] = 1 + c
    out["Spread-only"] = 1 + d
    out["Additive"] = 1 + 0.5 * (c + d)
    out["Gated"] = 1 + c * (p >= 2 / 3).astype(float)
    out["A monotonic"] = 1 + (GMIN + (GMAX - GMIN) * p) * c
    out["B U-shaped"] = 1 + (GMIN + (GMAX - GMIN) * 2 * (p - 0.5).abs()) * c
    out["S agreement"] = 1 + c * (1 + LAM * d * np.sign(c))
    return out


def evaluate(w, hml, ppy=4, nw_lags=4, start="1975-01-01", end=None, cost=COST):
    df = pd.DataFrame({"w": w, "hml": hml}).dropna()
    df["w_lag"] = df["w"].shift(1); df["w_prev"] = df["w"].shift(2); df = df.dropna()
    to = (df["w_lag"] - df["w_prev"]).abs()
    if start: df = df[df.index >= pd.Timestamp(start)]
    if end:   df = df[df.index <= pd.Timestamp(end)]
    to = to.reindex(df.index)
    df["static"] = df["hml"]
    df["timed"] = df["w_lag"] * df["hml"] - to * cost
    df["active"] = df["timed"] - df["static"]
    sr = lambda x: x.mean() / x.std() * np.sqrt(ppy)
    _, t = cc.newey_west_t(df["active"].values, nw_lags)
    return dict(n=len(df), start=str(df.index[0].date()), end=str(df.index[-1].date()),
                staticSR=sr(df["static"]), timedSR=sr(df["timed"]),
                alpha=df["active"].mean() * ppy * 100,
                IR=df["active"].mean() / df["active"].std() * np.sqrt(ppy),
                NWt=t, meanW=df["w_lag"].mean(),
                corr_av=np.corrcoef(df["active"], df["static"])[0, 1],
                turn=to.mean() * ppy), df
