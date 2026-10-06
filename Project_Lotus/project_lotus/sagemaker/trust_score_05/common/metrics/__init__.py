"""How a scored population is judged.

Two modules, because there are two questions.
:mod:`~trust_score_05.common.metrics.classification` answers "how good is this
ranking overall" with ROC AUC and average precision.
:mod:`~trust_score_05.common.metrics.ranking` answers "how good is the top of
this ranking, at the queue length we can actually work", which is the question
the business asks and the one a model is chosen on.

Both are threshold-free. Nothing in this pipeline converts a score into a
decision, so nothing here computes a confusion matrix at a fixed cutoff.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []
