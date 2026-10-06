"""Deciding what a row is, and making sure each one is counted once.

Three concerns, all of them about the *unit of evaluation* rather than about the
model.

**Aggregation level.** A single fraudulent customer can appear as many scored
rows: several reference timestamps, several phone numbers, several transactions
in one fraud case. Reporting recall over rows therefore answers a question nobody
asked — "what fraction of fraudulent *rows* did we surface" — when the business
question is what fraction of fraudulent *customers*, or *cases*, we surfaced.
:func:`aggregate_level` collapses a scored frame to one row per entity at one of
three grains, taking the maximum score, because an entity is as suspicious as its
most suspicious observation.

**Duplicate fraud cases.** The fraud feed contains the same case more than once:
once per transaction in the case, and again where the same event arrived from two
sources. Counting those separately inflates the denominator of recall and lets a
model look good by catching one heavily-duplicated case.
:func:`deduplicate_fraud` collapses them on the natural key.

**Contamination of the negative class.** The non-fraud sample is drawn from the
general population, which contains the fraud customers. If they are not removed,
a fraudulent customer appears twice — once labelled 1 and once labelled 0 — and
the second copy is counted as a false positive precisely when the model is right.
The notebook did this correctly on the *training* path (a ``left_anti`` join
against the fraud customer IDs) and omitted it on the *evaluation* path, so every
metric it reported understated precision by the amount of overlap, and understated
it more for the better models.

Nothing here touches Spark. These operate on pandas frames, after the population
has been collected to the driver, because the evaluation population is at most a
few million rows and the metrics are computed in numpy anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "AGGREGATION_LEVELS",
    "CASE_KEY_COLUMN",
    "CUSTOMER_KEY_COLUMN",
    "LABEL_COLUMN",
    "aggregate_level",
    "deduplicate_fraud",
    "record_entity_ids",
    "remove_fraud_customers",
    "stratified_sample",
]

#: The grains a scored population is evaluated at, coarsest last.
#:
#: ``record`` is every scored row, and it is the only level that is always
#: available. ``customer`` is the level the business reasons about. ``fraud_case``
#: is the level the fraud team reasons about, and it differs from ``customer``
#: because one customer can be the victim or subject of several distinct cases.
AGGREGATION_LEVELS = ("record", "customer", "fraud_case")

CUSTOMER_KEY_COLUMN = "customer_id"
CASE_KEY_COLUMN = "fraud_case_natural_key"

#: The fraud indicator: 1 for fraud, 0 or null for everything else.
#:
#: Defined here, with the two key columns, rather than in
#: :mod:`trust_score_05.ml.datasets`, because :mod:`trust_score_05.ml.evaluation`
#: needs it and that module has no business importing a Spark module to learn the
#: name of a pandas column. ``datasets`` re-exports it.
LABEL_COLUMN = "label"

#: Columns an aggregated frame always has. ``entity_id`` is a string at every
#: level so that the tie-break in
#: :func:`~trust_score_05.common.metrics.ranking.compute_k_metrics` compares like
#: with like.
_AGGREGATED_COLUMNS = ("entity_id", "label", "anomaly_score")


def record_entity_ids(n_records: int) -> "np.ndarray":
    """Zero-padded string identifiers ``"000000"``, ``"000001"``, … for *n* rows.

    Padded, and this is the point. Record-level entity IDs exist only to break
    ties in the ranking, and the ranking compares them as strings. Unpadded
    decimal strings sort ``'10'`` before ``'2'``, so the notebook's
    ``np.arange(n).astype(str)`` broke ties in an order that is neither the
    original row order nor anything else meaningful — and that changes wholesale
    when the population size crosses a power of ten, so the same model scored on
    999,999 rows and on 1,000,001 rows had its ties broken differently.

    Padding to the width of the largest index restores the property that string
    order equals numeric order, which makes the record-level tie-break "keep the
    input order", which is at least stable and explicable.
    """
    total = int(n_records)
    if total <= 0:
        return np.array([], dtype=object)
    width = len(str(total - 1))
    return np.array([f"{index:0{width}d}" for index in range(total)], dtype=object)


def aggregate_level(
    scored: "pd.DataFrame",
    level: str,
    score_col: str = "anomaly_score",
    label_col: str = "label",
) -> "pd.DataFrame":
    """Collapse ``scored`` to one row per entity at ``level``.

    Returns a frame with exactly ``entity_id``, ``label`` and ``anomaly_score``.
    The label is the group maximum — an entity is fraudulent if any of its rows
    is — and so is the score, because a detector that flags one of a customer's
    five reference points has flagged that customer.

    Args:
        scored: One row per scored record. Must carry ``score_col`` and
            ``label_col``, plus the key column the level needs.
        level: One of :data:`AGGREGATION_LEVELS`.
        score_col: The score column name.
        label_col: The label column name.

    Raises:
        KeyError: If ``level`` is not one of the three, or if the frame lacks the
            key column that level requires.

    Raising on a missing key column is the change from the notebook, which
    returned an empty DataFrame in that case. An empty frame flows onward, the
    metric loop finds nothing to compute, and the level is simply absent from the
    output — so a run whose fraud feed had lost its case key reported
    record-level and customer-level metrics and no case-level ones, with nothing
    anywhere saying why. Worse, its ``record`` branch was guarded by
    ``if level == "record" or "customer_id" not in pdf.columns``, so a frame
    missing ``customer_id`` silently returned *record*-level rows labelled as
    customer-level ones.
    """
    import pandas as pd

    if level not in AGGREGATION_LEVELS:
        raise KeyError(
            f"unknown aggregation level {level!r}; expected one of {list(AGGREGATION_LEVELS)}"
        )
    for column in (score_col, label_col):
        if column not in scored.columns:
            raise KeyError(f"scored frame has no {column!r} column")

    if level == "record":
        out = pd.DataFrame(
            {
                "entity_id": record_entity_ids(len(scored)),
                "label": _as_int_label(scored[label_col]),
                "anomaly_score": scored[score_col].to_numpy(dtype=float),
            }
        )
        return out

    key = CUSTOMER_KEY_COLUMN if level == "customer" else CASE_KEY_COLUMN
    if key not in scored.columns:
        raise KeyError(
            f"cannot aggregate to {level!r}: the scored frame has no {key!r} "
            f"column. It has {sorted(scored.columns)[:12]}…"
        )

    frame = scored.loc[:, [key, label_col, score_col]].copy()
    frame[label_col] = _as_int_label(frame[label_col])

    if level == "fraud_case":
        # A non-fraud row has no case key, and every one of them must stay a
        # separate entity: collapsing them all into a single "no case" group
        # would turn the entire negative class into one row, whose max score is
        # the highest-scoring non-fraud record in the population. Recall would
        # then be computed against a denominator of one.
        #
        # Falling back to the customer ID gives each of them its own group. Where
        # the customer ID is also absent the row index is used, which is the same
        # thing one level less informative.
        case_key = frame[key].astype("string")
        fallback = (
            scored[CUSTOMER_KEY_COLUMN].astype("string")
            if CUSTOMER_KEY_COLUMN in scored.columns
            else pd.Series(
                [f"__row_{index}" for index in range(len(frame))],
                index=frame.index,
                dtype="string",
            )
        )
        blank = case_key.isna() | (case_key.str.strip() == "")
        # `.mask` rather than `.fillna`: a key present but empty is as unusable
        # as one that is null, and only `.mask` covers both.
        frame[key] = case_key.mask(blank, fallback)
        still_blank = frame[key].isna()
        if still_blank.any():
            frame.loc[still_blank, key] = [
                f"__row_{index}" for index in np.flatnonzero(still_blank.to_numpy())
            ]

    grouped = (
        frame.groupby(frame[key].astype(str), dropna=False)
        .agg(label=(label_col, "max"), anomaly_score=(score_col, "max"))
        .reset_index()
        .rename(columns={key: "entity_id"})
    )
    grouped.columns = list(_AGGREGATED_COLUMNS)
    grouped["label"] = grouped["label"].astype(int)
    grouped["anomaly_score"] = grouped["anomaly_score"].astype(float)
    return grouped


def _as_int_label(series: "pd.Series") -> "np.ndarray":
    """Labels as ints, null becoming 0.

    See :func:`~trust_score_05.common.metrics.classification.compute_binary_metrics`
    for why null is a negative rather than a dropped row.
    """
    return series.fillna(0).astype(float).astype(int).to_numpy()


def deduplicate_fraud(
    fraud: "pd.DataFrame",
    case_key_col: str = CASE_KEY_COLUMN,
    order_by: Optional[Sequence[str]] = None,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """One row per fraud case. Returns the frame and a summary of what was dropped.

    Args:
        fraud: The fraud population, possibly with several rows per case.
        case_key_col: The natural key identifying a case.
        order_by: Columns deciding which row survives, earliest first. Defaults
            to the fraud timestamp when present, so the *first* observation of a
            case is kept — the point at which it became detectable.

    Returns:
        ``(deduplicated, summary)``. The summary carries ``rows_in``,
        ``rows_out``, ``duplicate_rows_dropped`` and ``cases``, and is written
        into the run manifest. It is returned rather than logged because the
        duplicate rate is a property of the *data* that a later reader of the
        metrics needs in order to interpret them: recall against 400 cases and
        recall against 400 case-transactions are different numbers and the
        summary is the only thing that says which one is on the report.

    Rows whose case key is missing are kept, each as its own case. A fraud event
    with no natural key is still a fraud event, and dropping it would remove real
    positives from the denominator.
    """
    import pandas as pd

    rows_in = int(len(fraud))
    if case_key_col not in fraud.columns:
        return fraud.copy(), {
            "rows_in": rows_in,
            "rows_out": rows_in,
            "duplicate_rows_dropped": 0,
            "cases": rows_in,
            "note": f"no {case_key_col!r} column; every row treated as its own case",
        }

    frame = fraud.copy()
    key = frame[case_key_col].astype("string")
    blank = key.isna() | (key.str.strip() == "")
    synthetic = pd.Series(
        [f"__unkeyed_{index}" for index in range(len(frame))],
        index=frame.index,
        dtype="string",
    )
    frame["__case_key"] = key.mask(blank, synthetic)

    sort_columns = list(order_by or ())
    if not sort_columns and "fraud_timestamp" in frame.columns:
        sort_columns = ["fraud_timestamp"]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort")

    kept = frame.drop_duplicates(subset="__case_key", keep="first").drop(columns="__case_key")
    rows_out = int(len(kept))
    return kept, {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "duplicate_rows_dropped": rows_in - rows_out,
        "cases": rows_out,
        "unkeyed_rows": int(blank.sum()),
    }


def remove_fraud_customers(
    non_fraud: "pd.DataFrame",
    fraud_customer_ids: Sequence[Any],
    customer_col: str = CUSTOMER_KEY_COLUMN,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """Drop rows whose customer appears in the fraud set. Returns frame and summary.

    This is the correction described in the module docstring. Without it, a
    fraudulent customer sampled into the non-fraud population appears twice, once
    with each label; the model scores both copies identically and the copy
    labelled 0 is counted as a false positive at exactly the *k* where the copy
    labelled 1 is counted as a true positive. Precision at *k* is understated by
    roughly the overlap rate, and — because the two copies sit next to each other
    in the ranking — the understatement is worst for the models that rank the
    fraud highest.

    Comparison is on the string form of the identifier. Customer IDs arrive as
    strings from one feed and as integers from another, and ``"12345" != 12345``
    would let every such customer through.
    """
    rows_in = int(len(non_fraud))
    if customer_col not in non_fraud.columns:
        raise KeyError(
            f"cannot de-contaminate the non-fraud population: it has no "
            f"{customer_col!r} column."
        )

    excluded = {str(value) for value in fraud_customer_ids if value is not None}
    if not excluded:
        return non_fraud.copy(), {
            "rows_in": rows_in,
            "rows_out": rows_in,
            "contaminated_rows_dropped": 0,
            "fraud_customers": 0,
        }

    keep = ~non_fraud[customer_col].astype("string").fillna("").isin(excluded)
    kept = non_fraud.loc[keep]
    rows_out = int(len(kept))
    return kept.copy(), {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "contaminated_rows_dropped": rows_in - rows_out,
        "fraud_customers": len(excluded),
    }


def stratified_sample(
    frame: "pd.DataFrame",
    max_rows: int,
    label_col: str = "label",
    seed: int = 42,
) -> "pd.DataFrame":
    """Cap ``frame`` at ``max_rows`` while keeping every positive.

    Used when a scenario is too large to plot or to collect. A uniform sample
    would discard fraud rows in proportion to their rarity, and at a 0.1% base
    rate a cap that keeps a tenth of the population keeps a tenth of the fraud —
    forty cases out of four hundred, from which no metric at *k* is stable. So
    the positives are kept whole and the cap is spent on the negatives.

    The returned frame is *not* a random sample of the population, and the
    metrics computed on it are therefore not estimates of the population
    metrics. That is acceptable for a plot and not acceptable for a reported
    number, which is why the pipeline only uses this for plotting.

    ``max_rows <= 0`` means no cap, matching the config convention where 0
    disables a limit.
    """
    if max_rows is None or int(max_rows) <= 0 or len(frame) <= int(max_rows):
        return frame.copy()

    import pandas as pd

    cap = int(max_rows)
    if label_col not in frame.columns:
        return frame.sample(cap, random_state=seed).copy()

    labels = frame[label_col].fillna(0).astype(float).astype(int)
    positives = frame.loc[labels == 1]
    negatives = frame.loc[labels != 1]

    if len(positives) >= cap:
        return positives.sample(cap, random_state=seed).copy()

    negative_budget = cap - len(positives)
    sampled_negatives = (
        negatives
        if len(negatives) <= negative_budget
        else negatives.sample(negative_budget, random_state=seed)
    )
    return pd.concat([positives, sampled_negatives], ignore_index=True)
