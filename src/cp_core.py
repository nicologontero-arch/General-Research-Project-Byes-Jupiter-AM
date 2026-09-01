"""Shared core: loaders, CP construction, z-scores, HAC t-stats, report API.

Look-ahead rules, applied throughout:
  1. CP(t) fits an expanding regression of the 12m-ahead average bond excess
     return on the 5 forwards, over observations already realised at t
     (s + 12m <= t). The last 12 months are always excluded.
  2. z-scores expand over history up to t.
  3. The signal at t trades the return over t+1 (strategy.py).
"""
import sys, atexit, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent   # src/ -> project root
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"; OUT_DIR.mkdir(exist_ok=True)
UPLOADS = str(DATA_DIR)

# Every writer goes through one of these.
TABLES_DIR = OUT_DIR / "tables"      # <script>.txt, console mirror
SECTIONS_DIR = OUT_DIR / "sections"  # <script>.json, machine-readable
FIG_DIR = OUT_DIR / "figures"        # declared for callers; not created here
EXHIBITS_DIR = OUT_DIR / "exhibits"  # declared for callers; not created here
for _d in (TABLES_DIR, SECTIONS_DIR):
    _d.mkdir(exist_ok=True)


# ------------------------- output-table capture -----------------------------
class _Tee:
    """Write to several streams at once."""
    def __init__(self, *streams): self._streams = streams
    def write(self, data):
        for s in self._streams: s.write(data)
        return len(data)
    def flush(self):
        for s in self._streams: s.flush()


def save_output(name):
    """Mirror stdout to TABLES_DIR/<name>.txt. Call once, at the top."""
    path = TABLES_DIR / f"{name}.txt"
    f = open(path, "w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, f)
    def _restore():                 # restore before closing: the final flush
        sys.stdout = original       # must not land on a closed file
        f.flush(); f.close()
    atexit.register(_restore)
    return path


# ---------------------------- structured report -----------------------------
# report()/h2()/table()/kv()/note() print a block and record the same block in
# sections/<name>.json.
_BLOCKS = []
_META = {}


def report(name, title="", subtitle=""):
    """Start a report: stdout -> <name>.txt, tables -> <name>.json."""
    save_output(name)
    _META.clear(); _META.update(name=name, title=title or name, subtitle=subtitle)
    _BLOCKS.clear()
    atexit.register(_flush_section)
    if title:
        print("=" * 96 + f"\n{title}" + (f"\n{subtitle}" if subtitle else "") + "\n" + "=" * 96)


def _flush_section():
    with open(SECTIONS_DIR / f"{_META['name']}.json", "w", encoding="utf-8") as fh:
        json.dump({"name": _META["name"], "title": _META["title"],
                   "subtitle": _META["subtitle"], "blocks": _BLOCKS}, fh, indent=1)


def h2(text):
    """Sub-section heading."""
    print("\n" + "-" * 96 + f"\n{text}\n" + "-" * 96)
    _BLOCKS.append({"type": "h2", "text": text})


def note(text):
    """Free-text line."""
    print(text)
    _BLOCKS.append({"type": "note", "text": text})


def table(headers, rows, title=None, note=None):
    """One table. First column left-aligned, the rest right. Cells go out as given."""
    headers = [str(h) for h in headers]
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    _print_ascii_table(headers, rows, title, note)
    _BLOCKS.append({"type": "table", "title": title, "headers": headers,
                    "rows": rows, "note": note})


def kv(pairs, title=None):
    """Key/value block."""
    pairs = [[str(k), ("" if v is None else str(v))] for k, v in pairs]
    if title: print(title)
    w = max((len(k) for k, _ in pairs), default=0)
    for k, v in pairs: print(f"  {k:<{w}} : {v}")
    _BLOCKS.append({"type": "kv", "title": title, "pairs": pairs})


def _print_ascii_table(headers, rows, title, note):
    ncol = len(headers)
    width = [len(headers[i]) for i in range(ncol)]
    for r in rows:
        for i in range(ncol):
            width[i] = max(width[i], len(r[i]) if i < len(r) else 0)

    def line(cells):
        out = []
        for i in range(ncol):
            c = cells[i] if i < len(cells) else ""
            out.append(c.ljust(width[i]) if i == 0 else c.rjust(width[i]))
        return "| " + " | ".join(out) + " |"

    rule = "|" + "|".join("-" * (w + 2) for w in width) + "|"
    if title: print(title)
    print(line(headers)); print(rule)
    for r in rows: print(line(r))
    if note: print(note)


# ------------------------------- loaders ------------------------------------
def load_yields():
    """GSW zero-coupon yields, daily, percent. Columns SVENY01..SVENY05."""
    df = pd.read_csv(f"{UPLOADS}/gsw_yields_daily.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [f"SVENY0{i}" for i in range(1, 6)]
    return df[cols].dropna()


FF5_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def load_ff5_all():
    """FF5 daily, all six columns, percent."""
    df = pd.read_csv(f"{UPLOADS}/F-F_Research_Data_5_Factors_2x3_daily.csv", skiprows=4)
    df = df.rename(columns={df.columns[0]: "date"})
    df = df[pd.to_numeric(df["date"], errors="coerce").notna()].copy()
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df = df.set_index("date").sort_index()
    for c in FF5_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[FF5_COLS]


def load_ff5():
    """FF5 daily, HML and RF only, percent."""
    return load_ff5_all()[["HML", "RF"]]


def load_ff_legs():
    """Daily HML legs from the 6 size-B/M portfolios, percent.
    H = 1/2(SMALL HiBM + BIG HiBM), L = 1/2(SMALL LoBM + BIG LoBM)."""
    import re
    lines = open(f"{UPLOADS}/6_Portfolios_2x3_Daily.csv").read().splitlines()
    s = next(i for i, l in enumerate(lines) if l.strip().startswith("Average Value Weighted Returns -- Daily"))
    hdr = [c.strip() for c in lines[s + 1].split(",")]
    col = {n: hdr.index(n) for n in ["SMALL LoBM", "SMALL HiBM", "BIG LoBM", "BIG HiBM"]}
    rec = {}
    for l in lines[s + 2:]:
        if not re.match(r"^\s*\d{8},", l): break
        p = l.split(","); v = {k: float(p[col[k]]) for k in col}
        if any(x <= -99 for x in v.values()): continue
        rec[p[0].strip()] = (0.5 * (v["SMALL HiBM"] + v["BIG HiBM"]),
                             0.5 * (v["SMALL LoBM"] + v["BIG LoBM"]))
    idx = pd.to_datetime(list(rec.keys()), format="%Y%m%d")
    return pd.DataFrame(list(rec.values()), index=idx, columns=["H", "L"]).sort_index()


# --------------------------- CP construction --------------------------------
def month_end_yields(y):
    return y.resample("ME").last().dropna()


def forwards(ym):
    """Forwards f(n) = n*y(n) - (n-1)*y(n-1), n = 1..5."""
    y = {n: ym[f"SVENY0{n}"] for n in range(1, 6)}
    f = pd.DataFrame(index=ym.index)
    f["f1"] = y[1]
    for n in range(2, 6):
        f[f"f{n}"] = n * y[n] - (n - 1) * y[n - 1]
    return f


def avg_excess_return_12m(ym):
    """12m bond excess return averaged over 2-5y, percent, indexed at the open."""
    y = {n: ym[f"SVENY0{n}"] for n in range(1, 6)}
    rx = pd.DataFrame(index=ym.index)
    for n in range(2, 6):
        rx[f"rx{n}"] = n * y[n] - (n - 1) * y[n - 1].shift(-12) - y[1]
    return rx.mean(axis=1)


def build_cp_realtime(ym, min_obs=120):
    """Expanding-window CP. Estimation sees realised observations only."""
    f = forwards(ym); avg_rx = avg_excess_return_12m(ym)
    months = ym.index; n = len(ym)
    realized_by = months.to_series().shift(-12).values
    Xd = np.column_stack([np.ones(n), f.values]); yv = avg_rx.values
    cp = pd.Series(index=months, dtype=float)
    for t in range(n):
        td = months[t]
        elig = [s for s in range(t + 1)
                if (not pd.isna(yv[s])) and (not pd.isna(realized_by[s]))
                and realized_by[s] <= np.datetime64(td)]
        if len(elig) < min_obs: continue
        beta, *_ = np.linalg.lstsq(Xd[elig], yv[elig], rcond=None)
        cp.iloc[t] = Xd[t] @ beta
    return cp.dropna()


def build_cp_rolling(ym, window_months=180, min_obs=96):
    """As build_cp_realtime, restricted to the last `window_months`."""
    f = forwards(ym); avg_rx = avg_excess_return_12m(ym)
    months = ym.index; n = len(ym)
    realized_by = months.to_series().shift(-12).values
    Xd = np.column_stack([np.ones(n), f.values]); yv = avg_rx.values
    cp = pd.Series(index=months, dtype=float)
    for t in range(n):
        td = months[t]
        lo = np.datetime64(td - pd.DateOffset(months=window_months))
        elig = [s for s in range(t + 1)
                if (not pd.isna(yv[s])) and (not pd.isna(realized_by[s]))
                and realized_by[s] <= np.datetime64(td) and np.datetime64(months[s]) >= lo]
        if len(elig) < min_obs: continue
        beta, *_ = np.linalg.lstsq(Xd[elig], yv[elig], rcond=None)
        cp.iloc[t] = Xd[t] @ beta
    return cp.dropna()


def cp_proxy_tent(ym):
    """Estimation-free tent proxy: -f1 + 2*f3 - f5."""
    f = forwards(ym)
    return (-f["f1"] + 2 * f["f3"] - f["f5"]).dropna()


# ------------------------------- helpers ------------------------------------
def expanding_z(s, min_pts=24):
    """Expanding-window z-score."""
    mu = s.expanding(min_periods=min_pts).mean()
    sd = s.expanding(min_periods=min_pts).std()
    return ((s - mu) / sd).dropna()


def newey_west_t(x, lags):
    """HAC mean and t-stat of a series against zero."""
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    m = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.params[0]), float(m.tvalues[0])


def factor_reg(r, hml, ppy, lags):
    """HAC regression of a strategy return on its benchmark:

        r(t) = alpha + beta*hml(t) + e(t)

    Inputs must already be aligned on the traded sample. Returns the regression
    alpha beside the benchmark-difference alpha (beta forced to 1); the two
    reconcile by alpha_diff - alpha_reg = (beta - 1)*E[hml]."""
    import statsmodels.api as sm
    d = pd.concat([pd.Series(r).rename("r"), pd.Series(hml).rename("h")],
                  axis=1, sort=False).dropna()
    m = sm.OLS(d["r"], sm.add_constant(d["h"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    beta = float(m.params["h"])
    t_b1 = float(np.squeeze(m.t_test("h = 1").tvalue))    # null: beta = 1
    alpha_diff = (d["r"].mean() - d["h"].mean()) * ppy * 100
    alpha_reg = float(m.params["const"]) * ppy * 100
    return dict(n=int(m.nobs), alpha_reg=alpha_reg, t_alpha=float(m.tvalues["const"]),
                beta=beta, t_beta_1=t_b1, r2=float(m.rsquared),
                alpha_diff=alpha_diff, gap=alpha_diff - alpha_reg,
                mean_hml=d["h"].mean() * ppy * 100)


def decompose(w_lag, hml, turn, ppy, lags, cost=0.001):
    """Exact split of the benchmark-difference alpha:

        E[active] = Cov(w, h) + (E[w] - 1)*E[h] - cost*E[turnover]

    An identity on the traded frame. `w_lag` is the weight actually traded
    (already shifted); `hml` and `turn` are its contemporaneous return and
    turnover.

    Each term is the mean of a per-period series, so its t-stat is HAC:
      timing  on (w - wbar)(h - hbar), the covariance's influence function.
              `t_timing_slope` is the same null via the slope of h on w.
      tilt    on (w - 1); multiplying by E[h] rescales, so the t-stat carries.
      total   on the active return.

    Levels are annualised percent."""
    import statsmodels.api as sm
    w = pd.Series(w_lag).astype(float)
    h = pd.Series(hml).astype(float)
    tn = pd.Series(turn).astype(float).reindex(w.index)

    g = (w - w.mean()) * (h - h.mean())            # covariance influence function
    cov, t_cov = newey_west_t(g.values, lags)
    m = sm.OLS(h.values, sm.add_constant(w.values)).fit(cov_type="HAC",
                                                        cov_kwds={"maxlags": lags})
    dw, t_w = newey_west_t((w - 1.0).values, lags)  # null: no static tilt
    tilt = dw * h.mean()
    cost_term = -cost * tn.mean()
    active = w * h - cost * tn - h
    tot, t_tot = newey_west_t(active.values, lags)

    return dict(n=len(w), mean_w=float(w.mean()),
                timing=cov * ppy * 100, t_timing=t_cov,
                t_timing_slope=float(m.tvalues[1]),
                tilt=tilt * ppy * 100, t_tilt=t_w,
                cost=cost_term * ppy * 100,
                total=tot * ppy * 100, t_total=t_tot,
                resid=(cov + tilt + cost_term - tot) * ppy * 100)


def compound(daily_pct):
    s = pd.Series(daily_pct).dropna()
    return np.prod(1.0 + s.values / 100.0) - 1.0
