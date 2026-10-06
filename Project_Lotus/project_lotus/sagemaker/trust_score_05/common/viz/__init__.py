"""Figures, saved with the numbers behind them.

One module, :mod:`~trust_score_05.common.viz.plots`. It exists so that no stage
has to decide where a plot goes, in what formats, or whether a failed plot should
end a run — and so that every figure the pipeline writes has a machine-readable
``plot_data`` sibling that can be checked long after the figure stopped being
persuasive.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []
