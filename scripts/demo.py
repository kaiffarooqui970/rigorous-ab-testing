"""One-command demo on synthetic data only: no credentials, no downloads (`make demo`).

1. Trust-checked reports for the two bundled example experiments (one has an SRM bug).
2. Quick (low-replication) runs of the Stage 2-4 validation studies.
3. A summary of what was produced and where.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str], allow: tuple[int, ...] = (0,)) -> int:
    print(f"$ {' '.join(args)}", flush=True)
    code = subprocess.call(args, cwd=ROOT)
    if code not in allow:
        sys.exit(f"demo step failed with exit code {code}: {' '.join(args)}")
    return code


def main() -> None:
    t0 = time.time()
    (ROOT / "reports").mkdir(exist_ok=True)
    report = [PY, "-m", "xpkit.report.cli"]
    run([*report, "examples/checkout_test.csv", "examples/checkout_test.yaml",
         "-o", "reports/checkout_test.html"])
    code = run([*report, "examples/checkout_test_srm.csv", "examples/checkout_test_srm.yaml",
                "-o", "reports/checkout_test_srm.html"], allow=(3,))
    assert code == 3, "the SRM example must be flagged"
    for stage in ("stage2_experiments", "stage3_causal", "stage4_uplift"):
        run([PY, f"validation/{stage}.py", "--quick"])
    print(f"\nDemo finished in {time.time() - t0:.0f}s.")
    print("  reports/checkout_test.html      healthy test: trust checks pass, CUPED effect")
    print("  reports/checkout_test_srm.html  logging bug: SRM detected, warning banner")
    print("  results/validation/stage*_quick.json  quick Monte Carlo runs (full: make validate)")
    print("  Interactive app: make app")


if __name__ == "__main__":
    main()
