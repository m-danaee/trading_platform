"""Capital-allocation extension point for the portfolio package.

Rule allocation is intentionally deferred to the next robustness task.  This
module exists so callers can import the future portfolio boundary without
adding a second allocation implementation to the RB governor.
"""

from __future__ import annotations

__all__: list[str] = []
