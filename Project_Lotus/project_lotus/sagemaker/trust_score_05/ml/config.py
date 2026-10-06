"""Configuration for the ML pipeline: one typed object, built from YAML.

The notebook carried a ``PipelineConfig`` dataclass with hard-coded S3 buckets in
its field defaults, and then mutated it — one cell monkey-patched
``versioned_models_root`` into a different property, several read overrides out of
environment variables and assigned them back onto the frozen-in-name-only
instance. The result was that no single place stated what a run's configuration
was, and the effective value of ``models_version_id`` depended on the order the
cells had been executed in.

Here, configuration comes from ``conf/ml/base.yaml`` plus overlays, through the
same loader the lineage jobs use, and lands in one frozen :class:`MLConfig`. The
loader is imported from :mod:`trust_score_05.lineage.config` rather than
reimplemented: deep merge, ``--set`` overrides and ``${ENV_VAR}`` interpolation
are not lineage-specific and having two subtly different mergers in one repo is
how a config key comes to mean different things in two stages.

Two conventions carried over from the lineage config and worth stating:

*Zero disables a limit.* ``max_eval_rows_per_scenario: 0`` means no cap, not
"score nothing". Every ``max_*`` field in here reads that way.

*Business rules live in the config, not in literals.* The lookback windows, the
*k* grids, the nine preprocessing variants and the feature counts to sweep are
all data. A run that wants a cheaper sweep narrows them in an overlay rather than
editing code, and the overlay is then the record of what that run did.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from trust_score_05.lineage.config import Config, ConfigError, load_config
from trust_score_05.ml.taxonomy import FRAUD_SOURCES, resolve_fraud_source

__all__ = [
    "DEFAULT_PREPROCESSING_IDS",
    "MLConfig",
    "default_base_path",
    "load_ml_config",
]

#: The nine preprocessing variants the pipeline sweeps, in the order they are
#: reported. Each identifier is parsed by
#: :mod:`trust_score_05.ml.preprocessing`; the names are not free text.
DEFAULT_PREPROCESSING_IDS: Tuple[str, ...] = (
    "raw_numeric_zero_impute_no_scaling",
    "domain_impute_no_scaling",
    "domain_impute_standard_scaler",
    "domain_impute_robust_scaler",
    "domain_impute_minmax_scaler",
    "domain_impute_quantile_normal",
    "median_impute_standard_scaler",
    "median_impute_robust_scaler",
    "zero_impute_with_missing_indicators_robust_scaler",
)

#: ``version_id=`` is a partition-style segment, so a bare version string gets
#: the prefix added. Accepting both spellings is what the notebook's
#: ``_normalize_model_output_version`` did, and it is genuinely useful: an
#: operator setting the version by hand types ``v3``, while a value read back out
#: of a path arrives as ``version_id=v3``.
_VERSION_SEGMENT = "version_id="

_TIME_WINDOW_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def default_base_path() -> Path:
    """``<repo>/conf/ml/base.yaml``.

    Derived from ``__file__``, which is a directory in a checkout and a path
    *inside a zip* when the package is shipped with ``spark-submit --py-files``.
    :func:`load_ml_config` therefore treats an absent base as fatal only when no
    overlay was named — the same accommodation
    :func:`trust_score_05.lineage.config.load_config` makes, and for the same
    reason.
    """
    return Path(__file__).resolve().parents[2] / "conf" / "ml" / "base.yaml"


def _as_str_tuple(values: Any, key: str) -> Tuple[str, ...]:
    """Coerce a YAML list (or a lone scalar) into a tuple of strings."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        return (str(values),)
    if not isinstance(values, Iterable):
        raise ConfigError(f"config key {key!r} must be a list, got {type(values).__name__}")
    return tuple(str(value) for value in values)


def _as_int_tuple(values: Any, key: str) -> Tuple[int, ...]:
    """Coerce a YAML list into a tuple of ints, refusing non-numeric entries."""
    out = []
    for value in _as_str_tuple(values, key):
        try:
            out.append(int(value))
        except ValueError as exc:
            raise ConfigError(f"config key {key!r} contains a non-integer: {value!r}") from exc
    return tuple(out)


@dataclass(frozen=True)
class MLConfig:
    """Everything a run of the ML pipeline needs to know, resolved and validated.

    Frozen. A stage that needs a variant calls :meth:`with_overrides`, which
    returns a new object — so the configuration a stage ran under is always the
    one it was handed, and can be written into its manifest without wondering
    whether something mutated it since.
    """

    # -- identity ----------------------------------------------------------
    #: ``full`` or ``pilot``. Selects between the two training-sample sizes and
    #: the two profiling sample sizes; nothing else branches on it.
    mode: str = "full"
    #: Groups every artifact of one experiment under a single prefix. Set it to
    #: the same value across the jobs of one sweep or their outputs scatter.
    models_version: str = "v3"
    #: Identifies one execution within a version. Defaults to a UTC timestamp,
    #: which makes reruns distinguishable without an operator inventing names.
    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    seed: int = 42

    # -- data locations ----------------------------------------------------
    training_root: str = ""
    testing_root: str = ""
    feature_selection_root: str = ""
    models_root: str = ""

    #: Monthly partitions of the training non-fraud population, as
    #: ``time_window=`` values.
    training_months: Tuple[str, ...] = ()
    #: Monthly partitions of the evaluation non-fraud population. Each becomes
    #: one *scenario*.
    testing_windows: Tuple[str, ...] = ()
    fraud_sources: Tuple[str, ...] = ()
    #: Fraud events before this instant are dropped. It exists because the fraud
    #: feed's early history is incomplete, not because early fraud is
    #: uninteresting — so it is a data-quality bound and belongs in config where
    #: it can be moved when the feed is backfilled.
    fraud_min_timestamp: str = "2024-10-01 00:00:00"

    # -- sampling ----------------------------------------------------------
    train_rows_per_month_full: int = 130_000
    train_rows_per_month_pilot: int = 25_000
    #: 0 = no cap. A cap here changes the *reported metrics*, because it changes
    #: the population they are computed over, so it should stay 0 outside a smoke
    #: test.
    max_eval_rows_per_scenario: int = 0
    #: 0 = no cap. Only affects plots, via
    #: :func:`trust_score_05.common.splitting.stratified_sample`.
    max_plot_rows: int = 200_000
    #: Rows drawn to profile feature null-rate and variance during selection.
    profile_rows_full: int = 200_000
    profile_rows_pilot: int = 50_000

    # -- sweep shape -------------------------------------------------------
    top_n_feature_counts: Tuple[int, ...] = (25, 50, 75, 100, 125, 150, 200)
    preprocessing_ids: Tuple[str, ...] = DEFAULT_PREPROCESSING_IDS
    #: 0 = the full grid. A positive value truncates it *deterministically* (see
    #: :func:`trust_score_05.ml.grids.limit_grid`), because a randomly truncated
    #: grid is not reproducible from the config alone.
    max_hyperparam_configs: int = 0

    # -- selection thresholds ---------------------------------------------
    #: A feature null in more than this fraction of profiled rows is dropped.
    #: 0.999 rather than something like 0.5 because a feature that is null 99% of
    #: the time can still be the sharpest signal available — a device change in
    #: the last hour is rare and that is the point. The bound is there to drop
    #: features that are *entirely* absent, which happens when a feature family
    #: was not computed for a scope.
    max_null_rate: float = 0.999
    #: A feature with variance at or below this is constant and carries nothing.
    min_variance: float = 1e-12

    # -- best-config selection --------------------------------------------
    #: The three together define "best": the recall a queue of this size achieves
    #: at this aggregation level. Held in config because it is a statement about
    #: the review capacity of the team that will use the model, and that is not a
    #: modelling decision.
    best_config_level: str = "customer"
    best_config_metric: str = "recall_at_k"
    best_config_k_percent: float = 1.0
    #: How scenario values are combined before configurations are compared. The
    #: mean, not the maximum: a configuration that is excellent on one month and
    #: useless on the others is worse than one that is good on all of them, and
    #: the notebook's ``max`` said the opposite. See
    #: :func:`trust_score_05.ml.evaluation.leaderboard`.
    best_config_aggregate: str = "mean"

    # -- drift thresholds --------------------------------------------------
    #: Flag a cluster whose share of the population changed by more than this
    #: *relative* to its training share. Relative rather than absolute, because
    #: ``k`` is a swept dimension and an absolute threshold means something
    #: different at every value of it: at ``k=2`` the average cluster holds half
    #: the population and a 0.05 absolute shift is a 10% move, while at ``k=200``
    #: the average cluster holds 0.005 and 0.05 is a shift no cluster can
    #: reach. The notebook used a fixed 0.05 absolute difference across a grid
    #: spanning ``k=2`` to ``k=200``, so its drift flag fired constantly at the
    #: bottom of the ladder and could not fire at all at the top.
    cluster_drift_proportion_ratio: float = 0.25
    #: Flag a cluster whose mean distance to its centroid grew by more than this
    #: multiple of the training mean.
    cluster_drift_distance_ratio: float = 1.25
    #: Flag the population when the score distribution's stability index exceeds
    #: this. 0.25 is the conventional "significant shift" line for a PSI.
    score_drift_psi_threshold: float = 0.25

    # -- behaviour ---------------------------------------------------------
    #: Fail rather than impute when a selected feature is missing from a
    #: population being scored. Default off, because the selection is made on
    #: training data and a feature legitimately absent from one month's fraud
    #: extract should not end the run — but see
    #: :mod:`trust_score_05.ml.datasets` for why the *count* of such features is
    #: recorded and why a high count invalidates the metrics.
    strict_missing_feature_check: bool = False
    #: Write fraud rows with unparseable or out-of-range timestamps to a
    #: quarantine table instead of dropping them silently.
    quarantine_bad_fraud_timestamps: bool = True
    #: Write the full scored population as CSV as well as parquet. Off: at a
    #: million rows per scenario per configuration the CSVs dominate the run's
    #: storage and nothing reads them.
    write_scores_csv: bool = False
    #: Raise when a plot cannot be written. See
    #: :mod:`trust_score_05.common.viz.plots` for why this defaults to on.
    strict_plots: bool = True

    # -- Spark ------------------------------------------------------------
    #: The session timezone the ML jobs pin. It must agree with
    #: ``spark.session_timezone`` in ``conf/lineage/base.yaml``, because the
    #: feature tables this pipeline reads were written by that pipeline and
    #: their timestamps are wall-clock time in that zone. ``fraud_min_timestamp``
    #: is a string parsed by ``to_timestamp`` in the *session's* zone, so a
    #: session five hours away moves the fraud cutoff five hours and changes
    #: which labelled rows are evaluated — silently, since the filter still
    #: succeeds. It is a field here rather than a constant so that one value
    #: governs and it appears in every run manifest.
    spark_session_timezone: str = "EST"

    #: The config sources this object was built from, for the run manifest.
    sources: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.mode not in ("full", "pilot"):
            raise ConfigError(f"mode must be 'full' or 'pilot', got {self.mode!r}")
        if not 0.0 <= self.max_null_rate <= 1.0:
            raise ConfigError(f"max_null_rate must be in [0, 1], got {self.max_null_rate}")
        if self.min_variance < 0:
            raise ConfigError(f"min_variance must be non-negative, got {self.min_variance}")
        if not 0.0 < self.best_config_k_percent <= 100.0:
            raise ConfigError(
                "best_config_k_percent is a percentage of the scored population "
                f"and must be in (0, 100]; got {self.best_config_k_percent}"
            )
        if any(count <= 0 for count in self.top_n_feature_counts):
            raise ConfigError(
                f"top_n_feature_counts must all be positive, got {self.top_n_feature_counts}"
            )
        if not self.preprocessing_ids:
            raise ConfigError("preprocessing_ids is empty; there would be nothing to sweep")
        if self.best_config_aggregate not in ("mean", "median", "min", "max"):
            raise ConfigError(
                "best_config_aggregate must be one of 'mean', 'median', 'min', "
                f"'max'; got {self.best_config_aggregate!r}"
            )
        for name in (
            "cluster_drift_proportion_ratio",
            "cluster_drift_distance_ratio",
            "score_drift_psi_threshold",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")

        # Time windows are partition *values*, and a malformed one produces a
        # path that does not exist — which the discovery step reports as "no
        # data for this month" rather than "this month is misspelled". Checking
        # the shape here turns a silently short run into a startup error.
        for label, windows in (
            ("training_months", self.training_months),
            ("testing_windows", self.testing_windows),
        ):
            bad = [w for w in windows if not _TIME_WINDOW_PATTERN.match(w)]
            if bad:
                raise ConfigError(
                    f"{label} entries must look like 'YYYY-MM-DD_HH-MM-SS'; got {bad}"
                )

        # A fraud source is used twice: as the directory name under
        # `<testing_root>/fraud/` that its partitions are listed from, and as the
        # feed identity that `taxonomy.category_for` maps fraud types with. Both
        # uses must work, so a name that resolves to neither feed is refused
        # here. The notebook checked neither, configured `("phase1", "nbc")`
        # against directories actually named `Phase1-v1.0` and `NBC-v1.0`, and so
        # discovered no fraud at all — see `taxonomy.resolve_fraud_source` for
        # what that did to the training population.
        unresolvable = [
            source for source in self.fraud_sources if resolve_fraud_source(source) is None
        ]
        if unresolvable:
            raise ConfigError(
                f"fraud_sources entries {unresolvable} name no known feed. Each entry "
                "is both a directory under <testing_root>/fraud/ and the feed identity "
                f"the fraud taxonomy maps types with, so it must contain exactly one of "
                f"{list(FRAUD_SOURCES)} (for example 'Phase1-v1.0' or 'nbc')."
            )

        overlap = set(self.training_months) & set(self.testing_windows)
        if overlap:
            raise ConfigError(
                "a time window appears in both training_months and "
                f"testing_windows: {sorted(overlap)}. The evaluation population "
                "would then include rows the model was fitted on."
            )

    # ------------------------------------------------------------------
    # derived values
    # ------------------------------------------------------------------
    @property
    def is_pilot(self) -> bool:
        return self.mode == "pilot"

    @property
    def train_rows_per_month(self) -> int:
        """Rows sampled per training month, per :attr:`mode`."""
        return self.train_rows_per_month_pilot if self.is_pilot else self.train_rows_per_month_full

    @property
    def profile_rows(self) -> int:
        """Rows sampled to profile features during selection, per :attr:`mode`."""
        return self.profile_rows_pilot if self.is_pilot else self.profile_rows_full

    @property
    def version_segment(self) -> str:
        """The models version as a path segment, e.g. ``version_id=v3``.

        Accepts ``v3`` or ``version_id=v3`` in the config and normalises both,
        so that a value copied out of an existing S3 path round-trips.
        """
        value = str(self.models_version).strip().strip("/")
        if value.startswith(_VERSION_SEGMENT):
            value = value[len(_VERSION_SEGMENT) :]
        if not value:
            raise ConfigError("models_version must not be empty")
        return _VERSION_SEGMENT + value

    @property
    def versioned_models_root(self) -> str:
        """``<models_root>/version_id=<models_version>``."""
        if not self.models_root:
            raise ConfigError("models_root is not configured")
        return f"{self.models_root.rstrip('/')}/{self.version_segment}"

    def scenario_names(self) -> Tuple[str, ...]:
        """The scenario identifiers, one per testing window: ``2024-11``, ….

        The first seven characters of the window, which is its year and month.
        Used as a path segment and as a column value, so it must be short and
        must not contain a character that needs escaping.
        """
        return tuple(window[:7] for window in self.testing_windows)

    def with_overrides(self, **changes: Any) -> "MLConfig":
        """A copy with fields replaced. Re-runs validation."""
        return replace(self, **changes)

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable view, for the run manifest.

        The derived values are included alongside the fields, because the
        question a manifest is read to answer is usually "where did this write
        to" rather than "what were the inputs to the path calculation".
        """
        from dataclasses import fields as dataclass_fields

        payload: Dict[str, Any] = {}
        for spec in dataclass_fields(self):
            value = getattr(self, spec.name)
            payload[spec.name] = list(value) if isinstance(value, tuple) else value
        payload["derived"] = {
            "train_rows_per_month": self.train_rows_per_month,
            "profile_rows": self.profile_rows,
            "version_segment": self.version_segment,
            "versioned_models_root": self.versioned_models_root if self.models_root else None,
            "scenario_names": list(self.scenario_names()),
        }
        return payload


def _config_value(config: Config, key: str, default: Any) -> Any:
    """``config.get`` with the default applied to an explicit YAML ``null`` too.

    A key present with a null value means "use the default" — that is how an
    overlay un-sets something a lower layer had set — and ``Config.get`` returns
    the ``None`` rather than the default in that case.
    """
    value = config.get(key, default)
    return default if value is None else value


def config_to_ml_config(config: Config) -> MLConfig:
    """Project a merged config tree onto :class:`MLConfig`.

    Every key is read from an explicit path, and the dataclass default supplies
    the fallback — so ``conf/ml/base.yaml`` and this file cannot silently
    disagree about a default, since only one of them is consulted per key.
    """
    defaults = MLConfig(
        training_months=(),
        testing_windows=(),
    )

    def get(key: str, attribute: Optional[str] = None) -> Any:
        return _config_value(config, key, getattr(defaults, attribute or key.split(".")[-1]))

    return MLConfig(
        mode=str(get("run.mode", "mode")),
        models_version=str(get("run.models_version", "models_version")),
        run_id=str(_config_value(config, "run.run_id", defaults.run_id)),
        seed=int(get("run.seed", "seed")),
        training_root=str(_config_value(config, "data.training_root", "")),
        testing_root=str(_config_value(config, "data.testing_root", "")),
        feature_selection_root=str(_config_value(config, "data.feature_selection_root", "")),
        models_root=str(_config_value(config, "data.models_root", "")),
        training_months=_as_str_tuple(
            _config_value(config, "data.training_months", ()), "data.training_months"
        ),
        testing_windows=_as_str_tuple(
            _config_value(config, "data.testing_windows", ()), "data.testing_windows"
        ),
        fraud_sources=_as_str_tuple(
            _config_value(config, "data.fraud_sources", ()), "data.fraud_sources"
        ),
        fraud_min_timestamp=str(get("data.fraud_min_timestamp", "fraud_min_timestamp")),
        train_rows_per_month_full=int(
            get("sampling.train_rows_per_month_full", "train_rows_per_month_full")
        ),
        train_rows_per_month_pilot=int(
            get("sampling.train_rows_per_month_pilot", "train_rows_per_month_pilot")
        ),
        max_eval_rows_per_scenario=int(
            get("sampling.max_eval_rows_per_scenario", "max_eval_rows_per_scenario")
        ),
        max_plot_rows=int(get("sampling.max_plot_rows", "max_plot_rows")),
        profile_rows_full=int(get("sampling.profile_rows_full", "profile_rows_full")),
        profile_rows_pilot=int(get("sampling.profile_rows_pilot", "profile_rows_pilot")),
        top_n_feature_counts=_as_int_tuple(
            _config_value(config, "sweep.top_n_feature_counts", defaults.top_n_feature_counts),
            "sweep.top_n_feature_counts",
        ),
        preprocessing_ids=_as_str_tuple(
            _config_value(config, "sweep.preprocessing_ids", defaults.preprocessing_ids),
            "sweep.preprocessing_ids",
        ),
        max_hyperparam_configs=int(
            get("sweep.max_hyperparam_configs", "max_hyperparam_configs")
        ),
        max_null_rate=float(get("selection.max_null_rate", "max_null_rate")),
        min_variance=float(get("selection.min_variance", "min_variance")),
        best_config_level=str(get("best_config.level", "best_config_level")),
        best_config_metric=str(get("best_config.metric", "best_config_metric")),
        best_config_k_percent=float(get("best_config.k_percent", "best_config_k_percent")),
        best_config_aggregate=str(get("best_config.aggregate", "best_config_aggregate")),
        cluster_drift_proportion_ratio=float(
            get("drift.cluster_proportion_ratio", "cluster_drift_proportion_ratio")
        ),
        cluster_drift_distance_ratio=float(
            get("drift.cluster_distance_ratio", "cluster_drift_distance_ratio")
        ),
        score_drift_psi_threshold=float(get("drift.score_psi", "score_drift_psi_threshold")),
        strict_missing_feature_check=bool(
            get("behaviour.strict_missing_feature_check", "strict_missing_feature_check")
        ),
        quarantine_bad_fraud_timestamps=bool(
            get("behaviour.quarantine_bad_fraud_timestamps", "quarantine_bad_fraud_timestamps")
        ),
        write_scores_csv=bool(get("behaviour.write_scores_csv", "write_scores_csv")),
        strict_plots=bool(get("behaviour.strict_plots", "strict_plots")),
        spark_session_timezone=str(
            get("spark.session_timezone", "spark_session_timezone")
        ),
        sources=tuple(config.sources),
    )


def load_ml_config(
    config_paths: Optional[Sequence[str]] = None,
    overrides: Optional[Sequence[str]] = None,
    base_path: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> MLConfig:
    """Load ``conf/ml/base.yaml`` plus overlays into a validated :class:`MLConfig`.

    Args:
        config_paths: Overlay files, applied in order. YAML or JSON, local or
            ``s3://``.
        overrides: ``key.path=value`` strings, applied after the files.
        base_path: Defaults to :func:`default_base_path`.
        env: Environment mapping for ``${VAR}`` interpolation.

    The one environment variable read outside interpolation is ``TS05_RUN_ID``,
    and only when the config does not set ``run.run_id``. It exists because a
    scheduler that fans one sweep out across many jobs has to give them all the
    same run ID, and it does that by setting an environment variable on each —
    there is no config file at that point.
    """
    tree = load_config(
        config_paths=config_paths,
        overrides=overrides,
        base_path=base_path if base_path is not None else default_base_path(),
        env=env,
    )
    resolved = config_to_ml_config(tree)

    if tree.get("run.run_id", None) is None:
        environ = os.environ if env is None else env
        from_env = environ.get("TS05_RUN_ID")
        if from_env:
            resolved = resolved.with_overrides(run_id=str(from_env))

    return resolved
