"""Run the DirectSpline shape-versus-location/scale decomposition.

This is a convenience entry point for the clean three-condition experiment:

* ``shape_only`` learns spline controls and the nonlinear gate only;
* ``location_scale_only`` freezes that nonlinear spline at identity and learns
  only bounded per-column location/scale residuals; and
* ``joint_shape_location_scale`` learns both parameter blocks jointly.

It delegates the paired train-only protocol and its margin diagnostics to
``direct_spline_basis_ablation.py``.  All options from that script remain
available; the experiment is fixed to ``decomposition``.
"""

from __future__ import annotations

import sys

try:  # Support both package-style imports and ``python scripts/file.py``.
    from scripts.direct_spline_basis_ablation import main
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_basis_ablation import main


if __name__ == "__main__":
    if "--experiment" in sys.argv:
        raise SystemExit("direct_spline_decomposition_ablation.py fixes --experiment to decomposition")
    sys.argv[1:1] = ["--experiment", "decomposition"]
    main()
