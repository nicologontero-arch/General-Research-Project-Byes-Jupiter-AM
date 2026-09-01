"""Rebuild every working data file from the archives in data/raw/.

    python prepare_data.py            rebuild, verify, write MANIFEST.txt
    python prepare_data.py --check    verify only, write nothing

The analysis never runs this; it reads the committed output.

    raw/feds200628.csv                  -> gsw_yields_daily.csv
    raw/*_CSV.zip                       -> the Fama-French files
    raw/glcnominalddata.zip             -> intl/yields_UK.csv
    raw/yield_curves.csv                -> intl/yields_CA.csv
    raw/jgbcme_all.csv                  -> intl/yields_JP.csv
    raw/bbk_paket1.csv                  -> intl/yields_DE.csv
    raw/The Devil in HMLs ....xlsx      -> intl/hml_daily_intl.csv

The four yield files come out indexed by date with columns SVENY01..SVENY05, so
country_cp.py reduces to a read_csv. Derived files are checked by round-trip:
the CSV is read back and compared against the frame the parser produced.
Verbatim files are checked by hash against their archive.

File-format work only. Sample periods are whatever the sources contain.
"""
import glob
import hashlib
import io
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")   # the BoE workbooks have mixed-format date cells

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
INTL = DATA / "intl"

CHECK_ONLY = "--check" in sys.argv
MANIFEST = []


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(frame, name, parser_label):
    """Write a derived frame to data/intl/<name> and assert it round-trips.

    Both float settings are load-bearing: %.17g out (pandas' default formatting
    is lossy at the last bit) and float_precision='round_trip' in (the default
    C parser is lossy too, even given all 17 digits). Without the pair the
    curves shift by ~1e-15 and every foreign number moves in its last decimal."""
    path = INTL / name
    if not CHECK_ONLY:
        INTL.mkdir(exist_ok=True)
        # lineterminator is pinned: otherwise pandas follows the platform and a
        # Windows rebuild writes CRLF, breaking every checksum in MANIFEST.txt.
        frame.to_csv(path, index_label="date", float_format="%.17g",
                     lineterminator="\n")
    back = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
    back.index.name = frame.index.name
    if not frame.equals(back):
        diff = (frame - back).abs().max().max()
        raise SystemExit(f"FAIL {name}: CSV does not round-trip (max |diff| {diff:.3g})")
    MANIFEST.append((f"data/intl/{name}", parser_label, len(frame),
                     f"{frame.index[0].date()}..{frame.index[-1].date()}",
                     f"{path.stat().st_size / 1e6:.2f} MB", sha256(path)[:16]))
    print(f"  {name:<24} {len(frame):>6} rows  "
          f"{frame.index[0].date()}..{frame.index[-1].date()}  round-trip OK")


# --------------------------------------------------------------- US, verbatim
def us_from_raw():
    """The four US files are the public downloads with line endings normalised.

    Ken French ships CRLF; the working copies are LF. Nothing else is touched -
    no rows dropped, no columns selected - so the check below is an equality
    test against the committed file, not a tolerance."""
    print("US files (verbatim from data/raw, CRLF -> LF)")
    for zname, member in [
        ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Research_Data_5_Factors_2x3_daily.csv"),
        ("Portfolios_Formed_on_BE-ME_CSV.zip", "Portfolios_Formed_on_BE-ME.csv"),
        ("6_Portfolios_2x3_daily_CSV.zip", "6_Portfolios_2x3_Daily.csv"),
    ]:
        with zipfile.ZipFile(RAW / zname) as z:
            inner = z.namelist()[0]
            rebuilt = z.read(inner).replace(b"\r\n", b"\n")
        committed = (DATA / member).read_bytes()
        ok = rebuilt == committed
        print(f"  {member:<44} {'matches committed' if ok else 'DIFFERS'}")
        if not ok:
            raise SystemExit(f"FAIL {member}: rebuild from {zname} does not match the committed file")
        MANIFEST.append((f"data/{member}", f"raw/{zname}, CRLF->LF", "-", "-",
                         f"{(DATA / member).stat().st_size / 1e6:.2f} MB",
                         sha256(DATA / member)[:16]))

    # The Fed file is wide and carries a prose preamble. Keep the date and the
    # ten zero-coupon yield columns.
    txt = (RAW / "feds200628.csv").read_text()
    lines = txt.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Date,BETA0"))
    gsw = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), low_memory=False)
    gsw = gsw.rename(columns={"Date": "date"})
    cols = ["date"] + [f"SVENY{n:02d}" for n in range(1, 11)]
    gsw = gsw[cols]
    buf = io.StringIO()
    # Same as emit(): without lineterminator this test fails on Windows against
    # a correct LF file.
    gsw.to_csv(buf, index=False, lineterminator="\n")
    rebuilt = buf.getvalue().encode()
    committed = (DATA / "gsw_yields_daily.csv").read_bytes()
    ok = rebuilt == committed
    print(f"  {'gsw_yields_daily.csv':<44} {'matches committed' if ok else 'DIFFERS'}")
    if not ok:
        raise SystemExit("FAIL gsw_yields_daily.csv: rebuild from feds200628.csv does not match")
    MANIFEST.append(("data/gsw_yields_daily.csv", "raw/feds200628.csv, SVENY01..10 extracted",
                     len(gsw), f"{gsw['date'].iloc[0]}..{gsw['date'].iloc[-1]}",
                     f"{(DATA / 'gsw_yields_daily.csv').stat().st_size / 1e6:.2f} MB",
                     sha256(DATA / "gsw_yields_daily.csv")[:16]))


# ------------------------------------------------- international yield parsers
# Each returns a frame indexed by date with columns SVENY01..SVENY05 in percent,
# so cp_core's CP construction applies to every market unchanged.
def parse_uk(workbook_dir):
    """UK: Bank of England nominal gilt zero-coupon spot curve (1979+).

    Eight workbooks, each with a spot-rate sheet whose maturity row is labelled
    'years:'. Sheet '4. nominal spot curve' is quoted on a half-year grid running
    0.5, 1.0, 1.5, ... so the nearest-column match below resolves to exactly 1..5
    years. It is written as a nearest match rather than a lookup so that a future
    vintage on a different grid degrades instead of raising. (An earlier version
    of this docstring claimed the maturities were not integers; checked 8 Aug 2026
    against the first workbook, they are.)"""
    import openpyxl
    frames = []
    for f in sorted(glob.glob(f"{workbook_dir}/GLC Nominal daily data*.xlsx")):
        wb = openpyxl.load_workbook(f, read_only=True); names = wb.sheetnames; wb.close()
        sheet = [s for s in names if "spot" in s.lower() and "short" not in s.lower()][0]
        raw = pd.read_excel(f, sheet_name=sheet, header=None)
        mrow = raw.index[raw[0].astype(str).str.strip() == "years:"][0]
        mats = pd.to_numeric(raw.iloc[mrow, 1:], errors="coerce")
        col = {n: (mats - n).abs().idxmin() for n in range(1, 6)}
        body = raw.iloc[mrow + 1:].copy()
        out = pd.DataFrame({"date": pd.to_datetime(body[0], errors="coerce")})
        for n in range(1, 6):
            out[f"SVENY0{n}"] = pd.to_numeric(body[col[n]], errors="coerce")
        frames.append(out.dropna(subset=["date"]).set_index("date"))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")].dropna()


def parse_ca():
    """Canada: Bank of Canada zero-coupon yields (1986+, decimals -> percent)."""
    df = pd.read_csv(RAW / "yield_curves.csv")
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    out = pd.DataFrame(index=df.index)
    for n, c in {1: "ZC100YR", 2: "ZC200YR", 3: "ZC300YR", 4: "ZC400YR", 5: "ZC500YR"}.items():
        out[f"SVENY0{n}"] = pd.to_numeric(df[c], errors="coerce") * 100.0
    return out.dropna()


def parse_jp():
    """Japan: MoF JGB constant-maturity yields (1974+, coupon-yield proxy)."""
    df = pd.read_csv(RAW / "jgbcme_all.csv", skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    out = pd.DataFrame(index=df.index)
    for n in range(1, 6):
        out[f"SVENY0{n}"] = pd.to_numeric(df[f"{n}Y"], errors="coerce")
    return out.dropna()


def parse_de():
    """Germany: Bundesbank Svensson zero-coupon spot 1-5y (monthly, 1972+)."""
    df = pd.read_csv(RAW / "bbk_paket1.csv", skiprows=5, header=None, dtype=str)
    df = df[df[0].astype(str).str.match(r"\d{4}-\d{2}$")].copy()
    idx = pd.PeriodIndex(df[0], freq="M").to_timestamp("M")
    out = pd.DataFrame(index=idx)
    for n, c in zip(range(1, 6), [13, 15, 17, 19, 21]):
        out[f"SVENY0{n}"] = pd.to_numeric(df[c].values, errors="coerce")
    return out.dropna()


def parse_aqr():
    """Foreign value factors: AQR 'Devil in HML's Details', sheet 'HML FF'.

    Kept wide with one column per market and NaNs preserved, because each market
    starts at a different date and `load_aqr_hml` drops per column. Decimal
    returns are scaled to percent here, as the original loader did."""
    src = RAW / "The Devil in HMLs Details Factors Daily.xlsx"
    df = pd.read_excel(src, sheet_name="HML FF", header=18)
    df = df.rename(columns={df.columns[0]: "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["DATE"]).set_index("DATE")
    out = pd.DataFrame(index=df.index)
    for code in ["GBR", "CAN", "JPN", "DEU"]:
        out[code] = pd.to_numeric(df[code], errors="coerce") * 100.0
    out.index.name = "date"
    return out.dropna(how="all")


def main():
    print(f"{'checking' if CHECK_ONLY else 'rebuilding'} data from {RAW}\n")
    us_from_raw()

    print("\nInternational yield curves (derived: parser output, SVENY01..05, percent)")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(RAW / "glcnominalddata.zip") as z:
            z.extractall(tmp)
        emit(parse_uk(tmp), "yields_UK.csv", "raw/glcnominalddata.zip (BoE spot curve)")
    emit(parse_ca(), "yields_CA.csv", "raw/yield_curves.csv (BoC zero-coupon)")
    emit(parse_jp(), "yields_JP.csv", "raw/jgbcme_all.csv (MoF JGB constant maturity)")
    emit(parse_de(), "yields_DE.csv", "raw/bbk_paket1.csv (Bundesbank Svensson)")

    print("\nInternational value factors (derived)")
    emit(parse_aqr(), "hml_daily_intl.csv", "raw/The Devil in HMLs Details Factors Daily.xlsx")

    hdr = ["file", "derived from", "rows", "span", "size", "sha256[:16]"]
    w = [max([len(hdr[i])] + [len(str(r[i])) for r in MANIFEST]) for i in range(6)]
    lines = ["  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)),
             "  ".join("-" * w[i] for i in range(6))]
    lines += ["  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in MANIFEST]
    table = "\n".join(lines)

    print("\nManifest")
    print("\n".join("  " + l for l in lines))
    if not CHECK_ONLY:
        # A separate file from data/README.md, which is authored: regenerating
        # this one cannot clobber that one.
        (DATA / "MANIFEST.txt").write_text(
            "Generated by src/prepare_data.py. Checksums cover the committed files.\n\n"
            + table + "\n", encoding="utf-8")
        print(f"\nwrote {DATA / 'MANIFEST.txt'}")
    print("\nAll files verified.")


if __name__ == "__main__":
    main()
