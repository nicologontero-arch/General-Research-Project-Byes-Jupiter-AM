# Bond risk premia and the timing of the value factor

Replication package: the input data, the code that reads it, and the output the code
writes.

## Requirements

Python 3.9 or later, and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the pipeline

All scripts live in `src/` and are run from there.

```bash
cd src
python run_all.py
```

That runs sixteen scripts in sequence and takes roughly two minutes. Other forms:

```bash
python run_all.py --list                  # print the pipeline and exit
python run_all.py results decomposition   # run only the named scripts
```

Script names may be given with or without the `.py` suffix. Each script is independent of
the others and of run order, so any one of them can also be run on its own:

```bash
python results.py
```

`prepare_data.py` is not part of `run_all.py`. It rebuilds `data/` and `data/intl/` from
the archives in `data/raw/`, and only needs running if those archives change; the files it
writes are already shipped.

```bash
python prepare_data.py            # rebuild from raw/, verify, rewrite data/MANIFEST.txt
python prepare_data.py --check    # verify the shipped files against raw/, write nothing
```

## Output

Every script writes two files, both named after the script:

- `outputs/tables/<name>.txt`, a mirror of what the script printed to the console
- `outputs/sections/<name>.json`, the same tables in machine-readable form

Both are written by a single call inside the script, so the two always carry the same
content. Re-running a script overwrites its own two files and touches nothing else.

`RESULTS.md` at the top level is assembled from `outputs/sections/*.json`.

Paths are resolved relative to `src/cp_core.py`: data is read from `../data` and output is
written under `../outputs`, whatever the shell's working directory.

## Layout

```
gonteron_supporting_files/
  README.md          this file
  RESULTS.md         the console output of every script, in one document
  requirements.txt   package floors
  src/               analysis code
  data/              input files, with README.md and MANIFEST.txt
    intl/            derived foreign yield curves and value factors
    raw/             the original public downloads (101MB)
  outputs/
    tables/          <script>.txt
    sections/        <script>.json
```

## Scripts

Five files in `src/` are imported by the others rather than run directly:

| Module | Contains |
|---|---|
| `cp_core.py` | data loaders, real-time CP construction, expanding z-score, Newey-West standard errors, the `decompose` identity, and the reporting API every table is written through |
| `strategy.py` | the overlay runner: weight on HML, signal lag, turnover and costs |
| `spread.py` | the value-spread loader and the seven CP-by-spread weighting designs |
| `country_cp.py` | the four foreign curve loaders (also runnable) |
| `cp_intl.py` | the international strategy runner (also runnable) |

The runnable scripts, in the order `run_all.py` executes them:

| Script | What it computes |
|---|---|
| `data_summary.py` | coverage of every input series as its loader sees it |
| `duration_test.py` | predictive regressions of twelve-month returns on CP, and the value-minus-growth leg contrast |
| `robustness.py` | yield-curve controls, White's Reality Check across the seven designs, a stub check, and cost sensitivity |
| `hybrid.py` | fits the five forwards directly to HML instead of to bond returns |
| `results.py` | the US strategy by frequency and sub-period, weight buckets, and a bootstrap |
| `decomposition.py` | the timing/tilt/cost split with t-stats, US, by frequency, sub-period and design |
| `integrity_audit.py` | truncation tests, placebos, burn-in stability, and out-of-sample bond R2 |
| `bond_link.py` | covariance of the traded position with realised bond excess returns, overlapping and non-overlapping |
| `p2_driver.py` | the seven CP-by-spread designs, an orthogonality diagnostic, and a bootstrap |
| `spread_checks.py` | truncation, rolling stability, decomposition and spanning, re-run on the agreement design |
| `country_cp.py` | builds each market's CP factor and reports its regression diagnostics |
| `cp_intl.py` | each market's own CP applied to its own value factor, with the US for reference |
| `decomposition_intl.py` | the timing/tilt split per market, and the mean-weight diagnostic |
| `factor_alpha.py` | the strategy return regressed on HML and on the five factors, reconciled against the benchmark-difference alpha |
| `r2_investigation.py` | the CP regression R2 by sampling frequency, window and data source |
| `rolling_cp.py` | rolling-window CP estimation against the expanding baseline |

Not in the pipeline:

| Script | What it does |
|---|---|
| `prepare_data.py` | rebuilds `data/` and `data/intl/` from `data/raw/` and verifies them |

## Verifying a fresh checkout

```bash
cd src
python prepare_data.py --check
python run_all.py
```

The first ends on `All files verified.`. The second leaves the 32 files under `outputs/`
byte-identical to the shipped copies.

## Environment

The shipped outputs were produced under Python 3.13.14 with pandas 3.0.1, numpy 2.4.2,
statsmodels 0.14.6 and scipy 1.17.0, and reproduce under pandas 2.x as well.
`requirements.txt` therefore sets floors and no ceilings.
