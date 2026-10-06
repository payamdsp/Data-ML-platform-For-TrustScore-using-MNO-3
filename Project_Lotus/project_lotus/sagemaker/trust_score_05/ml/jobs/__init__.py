"""The three ``spark-submit`` entry points of the ML pipeline, in run order.

``discovery_job``
    Inventory the schema of every configured prefix and evaluate the
    data-quality gates. Writes ``_READY.json`` only when no check is an error,
    which is the whole point of running it first: the two stages after it read
    that marker.
``selection_job``
    Profile the candidate features on the training union and write each model's
    ranked list and its ``top_n`` truncations.
``sweep_job``
    Walk the arms, fit and evaluate every hyperparameter configuration, and
    write the leaderboards.

They are three jobs rather than one because they have three different shapes.
Discovery is metadata-only and finishes in minutes. Selection is one profiling
pass over the training months, shared by every arm, and its output is reused
across many training runs — which is why it is versioned separately, under
``feature_selection_root`` (see
:func:`trust_score_05.ml.paths.feature_selection_prefix`). The sweep is the
expensive one, is fanned out across submissions, and is resumable. Fusing them
would mean re-profiling on every resume and re-running discovery on every arm.

The order is a real dependency and nothing enforces it beyond the markers, so
each job says what it needs: the sweep refuses to start if the selection it was
pointed at is absent, and reports rather than guesses when discovery never ran.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []
