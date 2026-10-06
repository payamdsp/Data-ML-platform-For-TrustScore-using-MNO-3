"""Load and validate silver DQ check definitions (checks.yaml).

A check definition is table-agnostic: it names the dataset, the timestamp column(s),
the uniqueness keys, the per-MNO breakdown column, and the per-family parameters.
The engine in ``metrics.py`` / ``runner.py`` consumes this; no DQ logic lives per table.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CheckConfig:
    """One silver table's DQ contract, parsed from that table's checks.yaml."""
    dataset: str
    primary_keys: list = field(default_factory=list)        # [] -> full-row
    partition_by: Optional[str] = None
    event_timestamp_col: str = "event_timestamp"

    timeliness_enabled: bool = False                        # set by presence of `timeliness:` block
    timeliness_mode: str = "batch"                          # batch = event-time | streaming = ingestion gap
    future_grace_seconds: int = 0                           # batch mode
    max_recency_seconds: Optional[int] = None               # batch mode, optional gate
    ingestion_timestamp_col: str = "ingestion_ts"           # streaming mode
    max_delay_seconds: Optional[int] = None                 # streaming mode, optional gate

    required_columns: list = field(default_factory=list)
    allowed_values: dict = field(default_factory=dict)              # {col:[vals]}
    allowed_values_by_mno: dict = field(default_factory=dict)       # {col:{mno:[vals]}}
    validity_formats: dict = field(default_factory=dict)            # {col: java-regex} (S12 Validity)
    victim_exclusion_col: Optional[str] = None                      # S12 Victim exclusion guard
    victim_allowed_values: list = field(default_factory=list)       #   values that mean "confirmed bad actor"
    currency_timestamp_col: str = "ingestion_ts"                    # S12 Currency: per-row age basis
    currency_max_age_seconds: Optional[int] = None                  #   review cycle; governance-provided

    def __post_init__(self):
        if not self.dataset:
            raise ValueError("checks.yaml must set 'dataset'")
        if self.allowed_values_by_mno and self.partition_by is None:
            raise ValueError("allowed_values_by_mno set but 'partition_by' (MNO col) is not")
        if self.timeliness_mode not in ("batch", "streaming"):
            raise ValueError(f"timeliness.mode must be 'batch' or 'streaming', got {self.timeliness_mode!r}")
        if (self.victim_exclusion_col is None) != (not self.victim_allowed_values):
            raise ValueError("victim_exclusion needs both 'column' and a non-empty 'allowed_values'")
        for col, pattern in (self.validity_formats or {}).items():
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"validity.formats.{col} must be a non-empty regex string")
        if self.currency_max_age_seconds is not None and int(self.currency_max_age_seconds) <= 0:
            raise ValueError("timeliness.currency.max_age_seconds must be positive")


def parse_checks(raw: dict) -> CheckConfig:
    c = raw or {}
    completeness = c.get("completeness") or {}
    consistency  = c.get("consistency") or {}
    validity     = c.get("validity") or {}
    victim       = c.get("victim_exclusion") or {}
    tl           = c.get("timeliness") or {}
    currency     = tl.get("currency") or {}
    return CheckConfig(
        dataset=c.get("dataset"),
        primary_keys=list(c.get("primary_keys") or []),
        partition_by=c.get("partition_by"),
        event_timestamp_col=c.get("event_timestamp_col", "event_timestamp"),
        timeliness_enabled=("timeliness" in c),
        timeliness_mode=tl.get("mode", "batch"),
        future_grace_seconds=int(tl.get("future_grace_seconds", 0)),
        max_recency_seconds=tl.get("max_recency_seconds"),
        ingestion_timestamp_col=tl.get("ingestion_timestamp_col", "ingestion_ts"),
        max_delay_seconds=tl.get("max_delay_seconds"),
        required_columns=list(completeness.get("required_columns") or []),
        allowed_values=dict(consistency.get("allowed_values") or {}),
        allowed_values_by_mno=dict(consistency.get("allowed_values_by_mno") or {}),
        validity_formats=dict(validity.get("formats") or {}),
        victim_exclusion_col=victim.get("column"),
        victim_allowed_values=list(victim.get("allowed_values") or []),
        currency_timestamp_col=currency.get("timestamp_col", "ingestion_ts"),
        currency_max_age_seconds=currency.get("max_age_seconds"),
    )


def load_checks(path: str) -> CheckConfig:
    """Read checks.yaml from a local path or s3:// uri."""
    import yaml
    if path.startswith("s3://"):
        import boto3  # lazy: local tests don't need boto3/creds
        bucket, key = path[len("s3://"):].split("/", 1)
        raw = yaml.safe_load(boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read())
    else:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
    return parse_checks(raw)
