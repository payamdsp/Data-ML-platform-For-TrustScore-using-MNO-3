"""Turning a frame of features into a matrix, the same way twice.

There are nine preprocessing variants, each named by an identifier like
``domain_impute_robust_scaler``, and the sweep treats the identifier as one of
its dimensions. A variant is two decisions: how to fill a missing value, and
whether and how to rescale. Both are *fitted*, and that is the whole reason this
module exists as something other than a call to ``sklearn.pipeline``.

Filling a missing value with a median means filling it with the median *of the
training population*. The notebook's domain imputation recomputed every median,
mean and tenure fallback from whatever frame it was handed, and it was handed
the evaluation frame at scoring time — so a fraud row's own value contributed to
the statistic used to fill its neighbours, and the same trained model produced
different scores for the same record depending on which other records happened
to be scored alongside it. The function even accepted a ``domain_rules``
argument carrying the training-time values, and never read it; the fitted rules
were assigned to ``_`` at the call site. :class:`DomainImputer` separates
:meth:`~DomainImputer.fit` from :meth:`~DomainImputer.transform` so that cannot
recur, and the fitted rules travel with the model in
:class:`~trust_score_05.common.io.serialization.ModelBundle`.

Identifiers are parsed, not pattern-matched. :func:`parse_preprocessing_id`
decomposes an identifier against a closed vocabulary and raises on anything it
cannot account for. The notebook asked ``"standard_scaler" in preprocessing_id``
and friends, so an identifier with a typo in it matched no scaler branch, fell
through to median imputation with no scaling, and produced a complete run under
a directory named for the variant it had not applied.

The domain rules themselves — zero for counts, tenure for elapsed-time
features, mean for entropies, median for everything else — encode what a missing
value *means* for each feature family. A count is absent because nothing
happened, so zero is the measurement, not a guess. Time since an event is absent
because the event never happened, so the honest fill is the age of the entity
itself. An entropy is absent because there was too little activity to compute
one, and neither zero (perfectly predictable) nor the maximum (perfectly random)
is defensible, so the population mean is used as the least-committal value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "DOMAIN_ENTROPY_PATTERN",
    "DOMAIN_TENURE_PATTERN",
    "DOMAIN_ZERO_PATTERN",
    "IMPUTATION_STRATEGIES",
    "SCALERS",
    "TENURE_SOURCE_CANDIDATES",
    "DomainImputer",
    "Preprocessor",
    "PreprocessingError",
    "PreprocessingSpec",
    "build_preprocessor",
    "fit_preprocessor",
    "parse_preprocessing_id",
]

LOGGER = logging.getLogger(__name__)


class PreprocessingError(ValueError):
    """A preprocessing identifier is unknown, or a transform was misapplied."""


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------

#: Imputation tokens an identifier may open with, mapped to the strategy.
#: ``raw_numeric_zero`` is a legacy spelling of ``zero`` kept because the
#: identifier is a directory name in the existing output tree and renaming it
#: would orphan every artifact written under the old name.
IMPUTATION_STRATEGIES: Dict[str, str] = {
    "raw_numeric_zero": "zero",
    "zero": "zero",
    "median": "median",
    "domain": "domain",
}

#: Scaler tokens an identifier may end with, mapped to the scaler. ``None``
#: means the matrix is used at its natural scale, which several of the models
#: — the tree ensembles and the univariate-CDF detectors — are invariant to.
SCALERS: Dict[str, Optional[str]] = {
    "no_scaling": None,
    "standard_scaler": "standard",
    "robust_scaler": "robust",
    "minmax_scaler": "minmax",
    "quantile_normal": "quantile_normal",
}

_INDICATOR_TOKEN = "with_missing_indicators"


@dataclass(frozen=True)
class PreprocessingSpec:
    """The decomposition of a preprocessing identifier.

    ``imputation`` is one of ``zero``, ``median`` or ``domain``; ``scaler`` is
    one of the :data:`SCALERS` values; ``add_indicators`` appends one binary
    column per feature that had a missing value in training, which changes the
    matrix width and is why :class:`Preprocessor` reports
    :attr:`~Preprocessor.output_feature_names` separately from the input list.
    """

    preprocessing_id: str
    imputation: str
    scaler: Optional[str]
    add_indicators: bool = False

    @property
    def uses_domain_rules(self) -> bool:
        return self.imputation == "domain"


def parse_preprocessing_id(preprocessing_id: str) -> PreprocessingSpec:
    """Decompose an identifier, raising on anything unaccounted for.

    The grammar is ``<imputation>_impute[_with_missing_indicators]_<scaler>``,
    parsed right to left because the scaler token is the only one that can
    contain an underscore ambiguously (``quantile_normal`` versus
    ``minmax_scaler``). Longest match wins, and whatever remains must be exactly
    an imputation token followed by ``_impute``.

    Raising is the point. Every unmatched token in the notebook's substring tests
    silently selected a default, so an identifier could name one variant and
    apply another for a whole sweep.
    """
    text = str(preprocessing_id).strip().lower()
    if not text:
        raise PreprocessingError("preprocessing id is empty")

    scaler_token = next(
        (token for token in sorted(SCALERS, key=len, reverse=True) if text.endswith(token)),
        None,
    )
    if scaler_token is None:
        raise PreprocessingError(
            f"preprocessing id {preprocessing_id!r} does not end with a known "
            f"scaler token; expected one of {sorted(SCALERS)}"
        )
    rest = text[: -len(scaler_token)].rstrip("_")

    add_indicators = False
    if rest.endswith(_INDICATOR_TOKEN):
        add_indicators = True
        rest = rest[: -len(_INDICATOR_TOKEN)].rstrip("_")

    if not rest.endswith("_impute"):
        raise PreprocessingError(
            f"preprocessing id {preprocessing_id!r} has {rest!r} where "
            "'<imputation>_impute' was expected"
        )
    imputation_token = rest[: -len("_impute")]
    if imputation_token not in IMPUTATION_STRATEGIES:
        raise PreprocessingError(
            f"preprocessing id {preprocessing_id!r} names imputation "
            f"{imputation_token!r}, which is not one of "
            f"{sorted(IMPUTATION_STRATEGIES)}"
        )

    return PreprocessingSpec(
        preprocessing_id=str(preprocessing_id),
        imputation=IMPUTATION_STRATEGIES[imputation_token],
        scaler=SCALERS[scaler_token],
        add_indicators=add_indicators,
    )


# --------------------------------------------------------------------------
# domain imputation
# --------------------------------------------------------------------------

#: Counts, flags, rates and dispersion statistics. A missing value here means
#: the underlying events did not occur, so zero is the measurement.
#:
#: Widened from the notebook's pattern, which required underscores on *both*
#: sides of ``cnt`` and ``any``. A feature named ``msisdn_device_change_cnt``
#: therefore did not match, fell through to the generic median, and was filled
#: with a positive number of events for the customers who had none — the
#: opposite of the intended rule, and applied to every count feature whose name
#: happened to end at ``_cnt`` rather than continue into a window suffix.
DOMAIN_ZERO_PATTERN = re.compile(
    r"(^|_)(cnt|any)(_|$)"
    r"|_weighted_cnt(_|$)"
    r"|_distinct_cnt(_|$)"
    r"|_event_cnt(_|$)"
    r"|_event_frac(_|$)"
    r"|_velocity(_|$)"
    r"|_stddev(_|$)"
    r"|_cv(_|$)"
    r"|_slope(_|$)"
    r"|_peak_to_avg(_|$)"
    r"|_over_"
)

#: Elapsed-time features. A missing value means the event never happened, and
#: the largest defensible elapsed time is the age of the entity, so these are
#: filled from a tenure column when one is available.
DOMAIN_TENURE_PATTERN = re.compile(
    r"days_since"
    r"|interval_(mean|median|min|max)"
    r"|spread_days"
    r"|transition_days"
    r"|days_for_last"
)

#: Entropies. Missing because there was not enough activity to compute one.
DOMAIN_ENTROPY_PATTERN = re.compile(r"entropy")

#: Customer-level tenure sources, used when no scope-matched tenure column is
#: present. All three are customer-scoped, so an account or msisdn feature that
#: reaches them is filled with the age of the customer rather than of the
#: account or the line — an over-estimate for anything created after the
#: customer was first seen, and the reason :data:`_SCOPE_TENURE_SOURCES` is
#: consulted first. They are still a better fill than the feature's own median,
#: which asserts that a customer whose event never happened is typical of the
#: customers whose event did.
TENURE_SOURCE_CANDIDATES: Tuple[str, ...] = (
    "cust_node_tenure_days",
    "cust_days_since_first_seen",
    "cust_days_since_current_assignment_start",
)

_SCOPE_PREFIXES = {
    "customer": ("cust_", "customer_"),
    "account": ("acct_", "account_"),
}

#: The tenure column of each scope, most-used spelling first.
#:
#: Spelled out per scope rather than built as ``f"{scope}_node_tenure_days"``,
#: which is what this module did until the tests below were written. The scope
#: keys are words — ``customer``, ``account`` — while the columns are prefixed
#: with the abbreviations the feature library emits — ``cust_``, ``acct_`` — so
#: interpolating the key produced ``customer_node_tenure_days``, a column that
#: exists in no frame. ``_tenure_source`` therefore never matched a scope and
#: every elapsed-time feature fell through to
#: :data:`TENURE_SOURCE_CANDIDATES`, which is customer-scoped: exactly the
#: substitution the scope match was added to prevent, and silent because a
#: fallback that is present and numeric raises nothing.
_SCOPE_TENURE_SOURCES: Dict[str, Tuple[str, ...]] = {
    "customer": ("cust_node_tenure_days", "customer_node_tenure_days"),
    "account": ("acct_node_tenure_days", "account_node_tenure_days"),
    "msisdn": ("msisdn_node_tenure_days",),
}


def _scope_of(feature: str) -> str:
    lowered = feature.lower()
    for scope, prefixes in _SCOPE_PREFIXES.items():
        if lowered.startswith(prefixes):
            return scope
    return "msisdn"


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce to float, with the infinities treated as missing.

    An infinity reaches here from a ratio whose denominator was zero. It is not
    a large value, it is an undefined one, so it is imputed rather than scaled —
    otherwise ``StandardScaler`` propagates it to every row of the column and
    ``MinMaxScaler`` collapses the whole column to zero.
    """
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


@dataclass
class DomainImputer:
    """Per-feature fill values, learned once from the training population.

    ``generic`` decides what an unmatched feature gets, and exists so the same
    class serves both ``domain_impute_*`` (generic median) and a future
    domain-with-zero-fallback variant without a second implementation.

    :meth:`fit` records a rule per feature; :meth:`transform` applies exactly
    those rules and computes nothing. Calling :meth:`transform` before
    :meth:`fit` raises rather than silently falling back, because a silent
    fallback is what the notebook did.
    """

    generic: str = "median"
    rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    fitted: bool = False

    def fit(self, frame: pd.DataFrame, features: Sequence[str]) -> "DomainImputer":
        if self.generic not in ("median", "zero"):
            raise PreprocessingError(
                f"generic fill must be 'median' or 'zero', got {self.generic!r}"
            )
        self.features = [str(name) for name in features]
        self.rules = {}
        available = set(frame.columns)

        for name in self.features:
            if name not in available:
                raise PreprocessingError(
                    f"cannot fit an impute rule for {name!r}: it is not a column "
                    "of the training frame"
                )
            values = _numeric(frame[name])
            lowered = name.lower()

            if DOMAIN_ZERO_PATTERN.search(lowered):
                self.rules[name] = {"rule": "domain_zero", "fill_value": 0.0}
                continue

            if DOMAIN_TENURE_PATTERN.search(lowered):
                source = self._tenure_source(name, available)
                fallback = self._median(values)
                rule = {
                    "rule": "domain_tenure" if source else "domain_tenure_median_fallback",
                    "fill_value": fallback,
                }
                if source:
                    rule["source"] = source
                    rule["source_fill_value"] = self._median(_numeric(frame[source]))
                self.rules[name] = rule
                continue

            if DOMAIN_ENTROPY_PATTERN.search(lowered):
                mean = float(values.mean()) if values.notna().any() else 0.0
                self.rules[name] = {
                    "rule": "domain_entropy_mean",
                    "fill_value": 0.0 if not np.isfinite(mean) else mean,
                }
                continue

            if self.generic == "zero":
                self.rules[name] = {"rule": "generic_zero", "fill_value": 0.0}
            else:
                self.rules[name] = {
                    "rule": "generic_median",
                    "fill_value": self._median(values),
                }

        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted rules. Computes no statistics from ``frame``."""
        if not self.fitted:
            raise PreprocessingError(
                "DomainImputer.transform called before fit; the fill values must "
                "come from the training population, not from the frame being "
                "transformed"
            )
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise PreprocessingError(
                f"frame is missing {len(missing)} fitted features, first 10: "
                f"{missing[:10]}"
            )

        out = pd.DataFrame(index=frame.index)
        for name in self.features:
            values = _numeric(frame[name])
            rule = self.rules[name]
            source = rule.get("source")
            if source and source in frame.columns:
                source_values = _numeric(frame[source]).fillna(rule["source_fill_value"])
                values = values.fillna(source_values)
            out[name] = values.fillna(rule["fill_value"])
        return out

    def fit_transform(self, frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        return self.fit(frame, features).transform(frame)

    def fill_values(self) -> Dict[str, float]:
        """The flat ``feature -> fill value`` map, for the model bundle.

        The tenure rules also carry a source column, which this map cannot
        express; the full rules are preserved on the imputer itself, which is
        pickled inside the bundle's preprocessor. This map is what
        :class:`~trust_score_05.common.io.serialization.ModelBundle` validates
        against the feature list and what a human reads to see what a missing
        value became.
        """
        return {name: float(rule["fill_value"]) for name, rule in self.rules.items()}

    def rule_counts(self) -> Dict[str, int]:
        """How many features got each rule, for the run's manifest."""
        counts: Dict[str, int] = {}
        for rule in self.rules.values():
            counts[rule["rule"]] = counts.get(rule["rule"], 0) + 1
        return counts

    @staticmethod
    def _median(values: pd.Series) -> float:
        if not values.notna().any():
            return 0.0
        median = float(values.median())
        return 0.0 if not np.isfinite(median) else median

    @staticmethod
    def _tenure_source(feature: str, available: set) -> Optional[str]:
        scope = _scope_of(feature)
        candidates = _SCOPE_TENURE_SOURCES.get(scope, ()) + TENURE_SOURCE_CANDIDATES
        return next((name for name in candidates if name in available and name != feature), None)


# --------------------------------------------------------------------------
# the full preprocessor
# --------------------------------------------------------------------------

def build_preprocessor(spec: PreprocessingSpec, n_train_rows: int, seed: int = 42):
    """The sklearn pipeline for one spec: always an imputer, optionally a scaler.

    An imputer is always present even when :class:`DomainImputer` has already
    filled everything. Two reasons: an empty ``Pipeline`` cannot be fitted at
    all, and a value that is missing at *scoring* time in a column that had none
    in training — a new feature, a changed upstream join — otherwise reaches the
    model as NaN. The second imputer's statistics are fitted on the
    domain-imputed matrix, so it inherits the domain rules' answers.

    ``QuantileTransformer`` is given ``n_quantiles`` capped at the training row
    count. Left at 1000 with fewer rows than that, sklearn warns and silently
    reduces it, which means the transform a bundle reproduces depends on how
    many rows the run happened to sample.

    ``RobustScaler`` uses the 5th–95th percentiles rather than the default
    quartiles. These features are heavy-tailed by construction — a burst count
    over a week has most of its mass at zero and a long right tail — and the
    interquartile range of such a column is often exactly zero, which makes the
    default scaler divide by zero and emit infinities.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    if spec.imputation == "zero":
        imputer = SimpleImputer(
            strategy="constant", fill_value=0.0, add_indicator=spec.add_indicators
        )
    else:
        imputer = SimpleImputer(strategy="median", add_indicator=spec.add_indicators)
    steps: List[Tuple[str, Any]] = [("imputer", imputer)]

    if spec.scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif spec.scaler == "robust":
        steps.append(("scaler", RobustScaler(quantile_range=(5.0, 95.0))))
    elif spec.scaler == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif spec.scaler == "quantile_normal":
        n_quantiles = int(max(2, min(1000, n_train_rows)))
        steps.append(
            (
                "scaler",
                QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=n_quantiles,
                    random_state=seed,
                ),
            )
        )
    return Pipeline(steps)


@dataclass
class Preprocessor:
    """A fitted, picklable feature-frame-to-matrix transform.

    Holds the spec, the optional :class:`DomainImputer` and the sklearn pipeline,
    and knows the matrix's column names — which are *not* the input feature
    names when ``add_indicators`` is set. The notebook returned the bare matrix
    and kept using the input feature list, so for the one variant with
    indicators every downstream mapping from matrix column to feature name
    (cluster-centroid inspection, per-feature reconstruction error) was silently
    off by however many indicator columns sklearn had appended.
    """

    spec: PreprocessingSpec
    pipeline: Any
    domain_imputer: Optional[DomainImputer] = None
    features: List[str] = field(default_factory=list)
    output_feature_names: List[str] = field(default_factory=list)
    n_train_rows: int = 0

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = self._prepare(frame)
        matrix = self.pipeline.transform(prepared)
        return self._finalize(matrix)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise PreprocessingError(
                f"frame is missing {len(missing)} of the preprocessor's features, "
                f"first 10: {missing[:10]}"
            )
        if self.domain_imputer is not None:
            return self.domain_imputer.transform(frame)
        return pd.DataFrame(
            {name: _numeric(frame[name]) for name in self.features}, index=frame.index
        )

    def _finalize(self, matrix: np.ndarray) -> np.ndarray:
        """Cast to float32 and refuse a matrix that is not finite.

        float32 halves the memory of a 130,000 by 200 matrix and every model here
        is indifferent to the last few bits of mantissa. The finiteness check is
        the guard the notebook lacked: a NaN reaching a scikit-learn estimator
        raises deep inside the fit with a message naming no column, and a NaN
        reaching an estimator that tolerates it produces a NaN score, which
        :func:`~trust_score_05.common.metrics.classification.compute_binary_metrics`
        then rejects — at the end of the run, having discarded which feature was
        responsible.
        """
        array = np.asarray(matrix, dtype=np.float32)
        if not np.isfinite(array).all():
            bad_columns = np.where(~np.isfinite(array).all(axis=0))[0]
            names = [
                self.output_feature_names[i]
                if i < len(self.output_feature_names)
                else f"column_{i}"
                for i in bad_columns[:10]
            ]
            raise PreprocessingError(
                f"preprocessing {self.spec.preprocessing_id} produced non-finite "
                f"values in {len(bad_columns)} of {array.shape[1]} columns, "
                f"first 10: {names}"
            )
        return array

    def metadata(self) -> Dict[str, Any]:
        """A JSON-serialisable description, for the run's config directory."""
        return {
            "preprocessing_id": self.spec.preprocessing_id,
            "imputation": self.spec.imputation,
            "scaler": self.spec.scaler,
            "add_indicators": self.spec.add_indicators,
            "n_features_in": len(self.features),
            "n_features_out": len(self.output_feature_names),
            "n_train_rows": self.n_train_rows,
            "domain_rule_counts": (
                self.domain_imputer.rule_counts() if self.domain_imputer else {}
            ),
        }


def fit_preprocessor(
    train: pd.DataFrame,
    features: Sequence[str],
    preprocessing_id: str,
    seed: int = 42,
) -> Tuple[np.ndarray, Preprocessor]:
    """Fit one variant on the training frame and return its matrix.

    This is the only place a preprocessing statistic is computed. Everything that
    scores a population calls :meth:`Preprocessor.transform`, which applies the
    values fitted here and computes nothing.
    """
    spec = parse_preprocessing_id(preprocessing_id)
    feature_list = [str(name) for name in features]
    if not feature_list:
        raise PreprocessingError("no features supplied")
    missing = [name for name in feature_list if name not in train.columns]
    if missing:
        raise PreprocessingError(
            f"training frame is missing {len(missing)} features, first 10: {missing[:10]}"
        )
    if train.empty:
        raise PreprocessingError("training frame has no rows")

    domain_imputer = None
    if spec.uses_domain_rules:
        domain_imputer = DomainImputer(generic="median").fit(train, feature_list)
        prepared = domain_imputer.transform(train)
    else:
        prepared = pd.DataFrame(
            {name: _numeric(train[name]) for name in feature_list}, index=train.index
        )

    pipeline = build_preprocessor(spec, n_train_rows=len(prepared), seed=seed)
    matrix = pipeline.fit_transform(prepared)

    output_names = _output_feature_names(pipeline, feature_list, spec)
    preprocessor = Preprocessor(
        spec=spec,
        pipeline=pipeline,
        domain_imputer=domain_imputer,
        features=feature_list,
        output_feature_names=output_names,
        n_train_rows=len(prepared),
    )
    return preprocessor._finalize(matrix), preprocessor


def _output_feature_names(pipeline, features: Sequence[str], spec: PreprocessingSpec) -> List[str]:
    """The matrix's column names, including any appended missing indicators.

    Read from the fitted imputer's ``indicator_.features_``, which is the index
    of the columns that had a missing value in training — the only place the
    appended columns' identities exist. Falls back to positional names if a
    future sklearn moves the attribute, so a rename cannot make this raise in the
    middle of a sweep.
    """
    names = [str(name) for name in features]
    if not spec.add_indicators:
        return names
    imputer = pipeline.named_steps.get("imputer")
    indicator = getattr(imputer, "indicator_", None)
    indices = getattr(indicator, "features_", None)
    if indices is None:
        LOGGER.warning(
            "missing-indicator column identities are unavailable; naming them "
            "positionally"
        )
        n_extra = int(getattr(imputer, "n_features_in_", 0))
        return names + [f"__missing_indicator_{i}" for i in range(n_extra)]
    return names + [f"{names[int(i)]}__was_missing" for i in indices]
