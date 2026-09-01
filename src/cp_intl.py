"""Each market's own CP timing its own HML (AQR HML FF). US for reference."""
import numpy as np, pandas as pd, warnings
import cp_core as cc, strategy as st, country_cp as ccp
warnings.filterwarnings("ignore")

AQR = cc.DATA_DIR / "intl" / "hml_daily_intl.csv"


def _aqr_table():
    """The four foreign value factors, percent, cached per process.
    NaNs are kept so each market drops on its own start date."""
    if not hasattr(_aqr_table, "_cache"):
        _aqr_table._cache = pd.read_csv(AQR, index_col=0, parse_dates=True,
                                        float_precision="round_trip")
    return _aqr_table._cache


def load_aqr_hml(country):
    return _aqr_table()[country].dropna()

CP = {"UK": cc.build_cp_realtime(cc.month_end_yields(ccp.load_boe()), min_obs=120),
      "Canada": cc.build_cp_realtime(cc.month_end_yields(ccp.load_canada()), min_obs=120),
      "Japan": cc.build_cp_realtime(cc.month_end_yields(ccp.load_japan()), min_obs=120),
      "Germany": cc.build_cp_realtime(cc.month_end_yields(ccp.load_germany()), min_obs=120)}
HMLCODE = {"UK": "GBR", "Canada": "CAN", "Japan": "JPN", "Germany": "DEU"}
MARKETS = ["UK", "Canada", "Japan", "Germany"]

def run(country, start=None, end=None):
    ff = pd.DataFrame({"HML": load_aqr_hml(HMLCODE[country])})
    return st.run_strategy(CP[country], ff, "QE", 4, z_minpts=12, nw_lags=4, start=start, end=end)

HDR = ["market","n","start","end","staticSR","timedSR","alpha%/yr","IR","NW t","meanW","turn"]
def row(r, tag):
    return [tag, r["n"], r["start"], r["end"], f"{r['static_SR']:+.2f}", f"{r['timed_SR']:+.2f}",
            f"{r['alpha_ann_pct']:+.2f}", f"{r['IR']:+.2f}", f"{r['NW_t']:+.2f}",
            f"{r['mean_w']:.2f}", f"{r['avg_turnover_per_yr']:.2f}"]

if __name__ == "__main__":
    cc.report("cp_intl", "International CP-only timing",
              "each country's own CP times its own HML (AQR Devil-in-HMLs, HML FF); US shown for reference")
    us = cc.build_cp_realtime(cc.month_end_yields(cc.load_yields()), min_obs=120)
    ffus = cc.load_ff5()
    rows = [row(st.run_strategy(us, ffus, "QE", 4, z_minpts=12, nw_lags=4, start="1975-01-01")[0], "US 1975+"),
            row(st.run_strategy(us, ffus, "QE", 4, z_minpts=12, nw_lags=4, start="1992-01-01")[0], "US 1992+")]
    results = {}
    for cty in MARKETS:
        r, df = run(cty); results[cty] = df; rows.append(row(r, cty))
    cc.table(HDR, rows, title="Own CP -> own HML FF (quarterly, net 10bps)")

    rows = []
    for cty in MARKETS:
        mid = results[cty].index[len(results[cty]) // 2]
        rows.append(row(run(cty, end=str((mid - pd.Timedelta(days=1)).date()))[0], f"{cty} H1"))
        rows.append(row(run(cty, start=str(mid.date()))[0], f"{cty} H2"))
    cc.table(HDR, rows, title="First / second half")

    alphas = [st.run_strategy(CP[c], pd.DataFrame({"HML": load_aqr_hml(HMLCODE[c])}),
              "QE", 4, z_minpts=12, nw_lags=4)[0]["alpha_ann_pct"] for c in MARKETS]
    cc.kv([("mean non-US alpha", f"{np.mean(alphas):+.2f}%/yr"),
           ("positive markets", f"{sum(a>0 for a in alphas)}/{len(MARKETS)}"),
           ("naive US-CP-on-foreign test (for contrast)", "-0.78%/yr, 2/7 positive")])
