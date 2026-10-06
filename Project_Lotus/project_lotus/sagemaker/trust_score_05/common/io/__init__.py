"""Reading and writing: object storage, and fitted-model bundles.

:mod:`~trust_score_05.common.io.s3` handles text, JSON, stage markers and
tabular files, against either an ``s3://`` URI or a local path.
:mod:`~trust_score_05.common.io.serialization` handles the one composite artifact
the pipeline produces — a fitted model together with the feature order,
imputation values and scaler that a score depends on.

Import the submodules rather than names from here; see
:mod:`trust_score_05.common` for why.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []
