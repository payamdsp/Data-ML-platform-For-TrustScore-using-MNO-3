"""Utilities shared by the ML pipeline stages, with no stage-specific knowledge.

Nothing in here knows about anomaly detection, fraud, or Trust Score. Every
module answers a question that any of the four ML stages — discovery, selection,
training, comparison — would otherwise answer for itself, differently:

``io.s3``
    Reading and writing text, JSON, markers and tables against either S3 or a
    local directory.
``io.serialization``
    Persisting a fitted model as one bundle that carries everything needed to
    reproduce a score.
``metrics.classification``
    Threshold-free ranking quality: ROC AUC, average precision.
``metrics.ranking``
    Top-*k* metrics, which is what a review queue of finite length actually cares
    about.
``splitting``
    Deciding what a row *is* — record, customer, or fraud case — and making sure
    each one is counted once.
``viz.plots``
    Saving a figure together with the numbers behind it.

The submodules are not re-exported here. Unlike
:mod:`trust_score_05.features`, whose public surface is a flat set of
calculators, this package's names only make sense with their module for context:
``read_text`` and ``compute_k_metrics`` want to be spelled
``s3.read_text`` and ``ranking.compute_k_metrics`` at the call site. Importing
matplotlib and joblib at package-import time would also be a real cost for a
caller that only wanted the S3 helpers.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []
