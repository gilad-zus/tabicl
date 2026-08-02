"""Run the paired DirectSpline low-rank cross-column headroom experiment.

The baseline learns the current best independent-column transform: bounded
location/scale residuals plus a monotone spline per numerical column.  The
second condition adds a zero-initialized, bounded rank-r residual mixing after
that transform.  It tests whether multivariate preprocessing raises held-out
DirectSpline headroom rather than merely fitting adaptation queries.

All options from ``direct_spline_basis_ablation.py`` remain available except
``--experiment``, which this wrapper fixes to ``cross_column``.
"""

from __future__ import annotations

import sys

try:  # Support both package-style imports and ``python scripts/file.py``.
    from scripts.direct_spline_basis_ablation import main
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_basis_ablation import main


if __name__ == "__main__":
    if "--experiment" in sys.argv:
        raise SystemExit("direct_spline_cross_column_ablation.py fixes --experiment to cross_column")
    sys.argv[1:1] = ["--experiment", "cross_column"]
    main()
