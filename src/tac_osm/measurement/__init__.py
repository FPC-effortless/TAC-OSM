"""Layer 0 measurement infrastructure: contracts, records, verdicts.

Nothing in this package is a scientific result. ``contract.py`` makes an
experiment's specification machine-readable, ``results.py`` makes its outcome
machine-readable, and ``verdicts.py`` defines the decision rule's threshold
once instead of once per script. Together they make a registered run
*checkable* — which is a narrower thing than making it true. See
``docs/EVIDENCE_REGISTER.md``, Layer 0.
"""

from __future__ import annotations
