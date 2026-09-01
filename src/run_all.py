"""Run the whole pipeline, in the order the argument is made.

    python run_all.py                          everything, ~2 minutes
    python run_all.py --list                   show the pipeline and exit
    python run_all.py results decomposition    only the named scripts

Each script writes outputs/tables/<name>.txt and outputs/sections/<name>.json.

prepare_data.py is not in this list. It rebuilds data/ from data/raw/ and is
only needed if the raw sources change; the files it writes are shipped.
"""
import runpy
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

# (script, section it feeds + one-line purpose). Ordered by the argument, not
# alphabetically: mechanism (S5), trading it (S6), the spread (S7), abroad (S8).
PIPELINE = [
    ("data_summary.py",       "S3  coverage of every input series as its loader sees it"),
    ("duration_test.py",      "S5  mechanism: business cycle against duration"),
    ("robustness.py",         "S5  curve controls; S6 costs; S7 Reality Check"),
    ("hybrid.py",             "S5  placebo: fit forwards straight to HML instead"),
    ("results.py",            "S6  frequency, sub-periods, weight buckets"),
    ("decomposition.py",      "S6  the estimand: timing vs tilt vs cost, with t-stats"),
    ("integrity_audit.py",    "S6  truncation, burn-in, out-of-sample bond R2"),
    ("bond_link.py",          "S6  does CP predict bonds in shape, if not in level?"),
    ("p2_driver.py",          "S7  CP x value spread: seven designs"),
    ("spread_checks.py",      "S7  S agreement under the same challenges"),
    ("country_cp.py",         "S8  build and validate each market's CP factor"),
    ("cp_intl.py",            "S8  international: own CP times own value factor"),
    ("decomposition_intl.py", "S8  international timing vs tilt, and the weight drift"),
    ("factor_alpha.py",       "S6  the three alpha measures reconciled"),
    ("r2_investigation.py",   "S4  why the CP regression R2 is 0.153, not 0.35"),
    ("rolling_cp.py",         "--  rolling vs expanding estimation: why S4 expands"),
]


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        w = max(len(s) for s, _ in PIPELINE)
        for i, (script, why) in enumerate(PIPELINE, 1):
            print(f"  {i:>2}. {script:<{w}}  {why}")
        return 0

    wanted = [a if a.endswith(".py") else a + ".py" for a in argv]
    todo = [p for p in PIPELINE if not wanted or p[0] in wanted]
    if wanted and not todo:
        print(f"no such script: {', '.join(wanted)}\nrun --list to see the pipeline")
        return 2

    print(f"running {len(todo)} scripts from {SRC}\n")
    t_all = time.time()
    failed = []
    for i, (script, why) in enumerate(todo, 1):
        print(f"[{i:>2}/{len(todo)}] {script:<24} {why}")
        t0 = time.time()
        # Subprocess, not import: each script registers atexit handlers to flush
        # its own .txt/.json, so they must not share an interpreter.
        r = subprocess.run([sys.executable, script], cwd=SRC,
                           capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"           ok  {dt:>5.1f}s")
        else:
            failed.append(script)
            print(f"           FAILED ({dt:.1f}s)")
            print("           " + "\n           ".join(r.stderr.strip().splitlines()[-6:]))

    print(f"\ntotal {time.time() - t_all:.0f}s")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("outputs/tables/*.txt, outputs/sections/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
