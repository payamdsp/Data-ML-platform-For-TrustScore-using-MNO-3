"""The anomaly-detection pipeline for Trust Score 0.5.

The package trains unsupervised models on a non-fraud population and evaluates
them against a labelled fraud population, which is the whole shape of the
problem: fraud labels exist, but far too few and far too late to train on, so
they are held back for evaluation only. Nothing in here fits a model to a label.

The modules are ordered the way a run moves through them:

``config``
    :class:`~trust_score_05.ml.config.MLConfig`, frozen, built from
    ``conf/ml/*.yaml``. Every threshold, row cap and sweep dimension lives here.
``taxonomy``
    The ten fraud categories, and the two feeds' vocabularies mapped onto them.
``paths``
    Where every artifact goes, as ``key=value`` segments, and
    :func:`~trust_score_05.ml.paths.parse_run_prefix` to read them back.
``datasets``
    Discovering the parquet prefixes, reading them without tripping Spark's
    partition inference, and assembling the training matrix and the per-scenario
    evaluation frames.
``discovery``
    Schema inventory and data-quality gates over the feature tables.
``selection``
    Feature selection: leakage exclusion, null-rate and variance profiling, and
    the per-model ranked feature lists.
``preprocessing``
    The nine imputation/scaling variants, fitted on training rows only.
``models``
    Ten model adapters behind one interface, each fitting and scoring in the
    same space so that the adapter which fits is the adapter which scores.
``grids``
    What the sweep searches for each model: the parameter values that change a
    ranking, and only those.
``evaluation``
    Scoring a population, the metric tables, and best-configuration selection.
``drift``
    Whether the population has moved since training: the anomaly score's
    stability index for every model, and per-cluster share and distance drift
    for the three that partition.
``sweep``
    The resumable driver that walks the cross-product and writes the markers.

Unlike :mod:`trust_score_05.features`, this package does not re-export its
modules' names. Importing it would otherwise pull in scikit-learn, joblib and
matplotlib for a caller that only wanted a path string, and ``paths.run_prefix``
carries information at the call site that a bare ``run_prefix`` does not.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []
