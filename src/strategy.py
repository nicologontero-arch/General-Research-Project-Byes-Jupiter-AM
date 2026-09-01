"""Strategy runner: w = 1 + tanh(z_CP) on HML, lagged one period, net of costs."""
import numpy as np
import pandas as pd
import cp_core as cc

COST = 0.001  # 10bps per unit turnover


def period_hml(ff, freq, complete_only=False):
    """Compound daily HML to `freq`.

    complete_only drops a trailing partial period. resample labels by period
    end, which can post-date the last observation: daily data ending 2026-04-30
    yields a '2026-06-30' quarter holding April alone."""
    s = ff["HML"].resample(freq).apply(cc.compound).dropna()
    if complete_only and len(s):
        s = s[s.index <= ff.index[-1]]
    return s


def run_strategy(cp_me, ff, freq, ppy, z_minpts, nw_lags, label="", start=None, end=None,
                 cost=COST, complete_only=False):
    cp_p = cp_me.resample(freq).last().dropna()
    z = cc.expanding_z(cp_p, min_pts=z_minpts)
    w = 1.0 + np.tanh(z)
    hml = period_hml(ff, freq, complete_only=complete_only)

    df = pd.DataFrame({"w": w, "hml": hml}).dropna()
    df["w_lag"] = df["w"].shift(1)
    df["w_prev"] = df["w"].shift(2)
    df = df.dropna()
    turnover = (df["w_lag"] - df["w_prev"]).abs()
    df["static"] = df["hml"]
    df["timed"] = df["w_lag"] * df["hml"] - turnover * cost
    df["active"] = df["timed"] - df["static"]

    if start is not None: df = df[df.index >= pd.Timestamp(start)]
    if end is not None:   df = df[df.index <= pd.Timestamp(end)]
    turnover = turnover.reindex(df.index)

    sharpe = lambda x: x.mean() / x.std() * np.sqrt(ppy)
    a_mean, a_t = cc.newey_west_t(df["active"].values, nw_lags)
    res = dict(
        label=label, freq=freq, n=len(df),
        start=str(df.index[0].date()), end=str(df.index[-1].date()),
        static_SR=sharpe(df["static"]), timed_SR=sharpe(df["timed"]),
        alpha_ann_pct=df["active"].mean() * ppy * 100,
        IR=df["active"].mean() / df["active"].std() * np.sqrt(ppy),
        NW_t=a_t, mean_w=df["w_lag"].mean(),
        corr_active_value=np.corrcoef(df["active"], df["static"])[0, 1],
        avg_turnover_per_yr=turnover.mean() * ppy)
    return res, df


if __name__ == "__main__":
    y = cc.load_yields(); ff = cc.load_ff5(); ym = cc.month_end_yields(y)
    cp = cc.build_cp_realtime(ym, min_obs=120)
    r, _ = run_strategy(cp, ff, "QE", 4, z_minpts=12, nw_lags=4, start="1975-01-01")
    print("US quarterly CP-timed value:",
          f"alpha={r['alpha_ann_pct']:+.2f}%/yr IR={r['IR']:.2f} NWt={r['NW_t']:.2f}")
