"""The four §6.1.2 DQ metric families, expressed as Spark aggregation columns.

Every family contributes a list of aggregation ``Column`` expressions plus a parser
that turns a collected ``Row`` into a JSON-ready dict. ``runner._scan`` concatenates
all families into ONE ``.agg(...)``, so a DQ run scans the silver table at most twice
regardless of how many checks the yaml configures.

Null discipline: silver has already mapped ``\\N`` / "" / ``MISSING_DATA`` to true nulls
upstream, so completeness counts SQL nulls. Consistency violations are counted on
NON-NULL values only (a null is a completeness signal, not a domain signal).
"""

from pyspark.sql import Column
from pyspark.sql import functions as F

_NULL_SENTINEL = "\u0000NULL\u0000"
_KEY_SEP = "\u0001"


def alias_safe_name(column_name: str) -> str:
    """Strip characters that can't appear in an aggregation alias."""
    return column_name.replace(".", "_").replace("`", "")


def resolve_column(name, columns):
    """The table column matching `name` case-insensitively, or None.

    Same identifier rule Spark itself uses. Every family that takes a column name
    from the yaml resolves through here; see parse_completeness for the live
    incident that made case-blind matching mandatory.
    """
    fold = name.casefold()
    for c in columns:
        if c.casefold() == fold:
            return c
    return None


# completeness #
def completeness_exprs(columns):
    return [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"null_rows_{alias_safe_name(c)}")
            for c in columns]


def parse_completeness(row, columns, row_count, required):
    per_column, rules = {}, []
    # Yaml names must match table names the way Spark resolves columns: case-
    # insensitively. Proven live 2026-07-17 on account_changes_batch: Iceberg
    # preserves column case, the table has phone_number_AC_hash, the yaml says
    # phone_number_ac_hash a case-sensitive `c in required` produced 6 rules
    # from 7 required columns and the missing gate left no trace.
    required_by_fold = {r.casefold(): r for r in sorted(required)}
    matched_folds = set()
    for c in columns:
        nulls = int(row[f"null_rows_{alias_safe_name(c)}"] or 0)
        rate = round(nulls / row_count, 6) if row_count else 0.0
        per_column[c] = {"null_count": nulls, "null_rate": rate}
        if c.casefold() in required_by_fold:
            matched_folds.add(c.casefold())
            rules.append({"family": "completeness", "rule": f"{c} is not null",
                          "column": c, "violations": nulls, "passed": nulls == 0})
    # A required column the table lacks under ANY casing is a failed rule, not a
    # silent no-op otherwise a typo or a renamed column turns its gate off.
    for fold in sorted(set(required_by_fold) - matched_folds):
        name = required_by_fold[fold]
        rules.append({"family": "completeness", "rule": f"{name} exists in the table",
                      "column": name, "violations": 1, "passed": False})
    return {"per_column": per_column, "rules": rules}


# consistency #
def consistency_exprs(allowed_values, allowed_values_by_mno, mno_col):
    exprs = []
    for col, allowed in allowed_values.items():
        viol = F.col(col).isNotNull() & ~F.col(col).isin(list(allowed))
        exprs.append(F.sum(F.when(viol, 1).otherwise(0))
                     .alias(f"domain_violation_rows_{alias_safe_name(col)}"))
    for col, by_mno in allowed_values_by_mno.items():
        cond = F.lit(False)
        for mno, allowed in by_mno.items():
            cond = cond | ((F.col(mno_col) == mno) & F.col(col).isNotNull()
                           & ~F.col(col).isin(list(allowed)))
        exprs.append(F.sum(F.when(cond, 1).otherwise(0))
                     .alias(f"mno_domain_violation_rows_{alias_safe_name(col)}"))
    return exprs


def parse_consistency(row, allowed_values, allowed_values_by_mno, row_count):
    per_column, rules = {}, []
    for col, allowed in allowed_values.items():
        v = int(row[f"domain_violation_rows_{alias_safe_name(col)}"] or 0)
        per_column[col] = {"violation_count": v,
                           "violation_rate": round(v / row_count, 6) if row_count else 0.0,
                           "allowed_values": list(allowed)}
        rules.append({"family": "consistency", "rule": f"{col} in {sorted(map(str, allowed))}",
                      "column": col, "violations": v, "passed": v == 0})
    for col, by_mno in allowed_values_by_mno.items():
        v = int(row[f"mno_domain_violation_rows_{alias_safe_name(col)}"] or 0)
        per_column[col] = {"violation_count": v,
                           "violation_rate": round(v / row_count, 6) if row_count else 0.0,
                           "allowed_values_by_mno": {m: list(a) for m, a in by_mno.items()}}
        rules.append({"family": "consistency", "rule": f"{col} in per-MNO allowed set",
                      "column": col, "violations": v, "passed": v == 0})
    return {"per_column": per_column, "rules": rules}


# validity #
def validity_exprs(formats, columns):
    """One mismatch counter per configured format whose column exists (any casing).

    NULLs are completeness's job; a format check counts only non-null values that
    fail the pattern. Patterns are Java regex (Spark rlike) matched ANYWHERE in
    the value; anchor with ^...$ to require a full-string match.
    """
    exprs = []
    for name, pattern in sorted(formats.items()):
        c = resolve_column(name, columns)
        if c is not None:
            exprs.append(
                F.sum(F.when(F.col(c).isNotNull() & ~F.col(c).rlike(pattern), 1).otherwise(0))
                 .alias(f"format_violation_rows_{alias_safe_name(c)}"))
    return exprs


def parse_validity(row, formats, columns, row_count):
    per_column, rules = {}, []
    for name, pattern in sorted(formats.items()):
        c = resolve_column(name, columns)
        if c is None:
            # Fail closed, same as completeness: a format on a column the table
            # lacks under any casing must go red, not silently vanish.
            rules.append({"family": "validity", "rule": f"{name} exists in the table",
                          "column": name, "violations": 1, "passed": False})
            continue
        bad = int(row[f"format_violation_rows_{alias_safe_name(c)}"] or 0)
        per_column[c] = {"pattern": pattern, "violation_count": bad,
                         "violation_rate": round(bad / row_count, 6) if row_count else 0.0}
        rules.append({"family": "validity", "rule": f"{c} matches format",
                      "column": c, "violations": bad, "passed": bad == 0})
    return {"per_column": per_column, "rules": rules}


# uniqueness (null-safe key) #
def uniqueness_exprs(primary_keys, all_columns):
    keys = primary_keys or all_columns
    # coalesce each key part to a sentinel BEFORE concat (concat_ws drops nulls),
    # hash to fixed length, count distinct. A raw countDistinct over the key columns
    # would skip any row with a null key part -> wrong dedup rate on tables with
    # nullable keys (e.g. audit_trail full-row dedup where msisdn is 100% null).
    key = F.sha2(F.concat_ws(
        _KEY_SEP,
        *[F.coalesce(F.col(k).cast("string"), F.lit(_NULL_SENTINEL)) for k in keys],
    ), 256)
    return [F.countDistinct(key).alias("distinct_key_rows")]


def parse_uniqueness(row, primary_keys, row_count):
    distinct = int(row["distinct_key_rows"] or 0)
    dupes = row_count - distinct
    keys = primary_keys or ["<full row>"]
    return {"metrics": {"primary_keys": keys, "distinct_keys": distinct,
                        "duplicate_count": dupes,
                        "duplicate_rate": round(dupes / row_count, 6) if row_count else 0.0,
                        "distinct_rate": round(distinct / row_count, 6) if row_count else 0.0},
            "rules": [{"family": "uniqueness", "rule": f"unique on {keys}",
                       "column": ",".join(keys), "violations": dupes, "passed": dupes == 0}]}


# timeliness #
# Two modes, chosen by `timeliness.mode` in the yaml:
#
# batch event-time only. Gate = no event dated after the run. Use while the
# table is a one-shot historical backfill: every row's ingestion_ts is
# ~the same instant, so an ingestion gap measures "how old is this
# event" (0-3 years), not "how slow is the pipeline" no threshold
# is correct. Today all 4 tables are here, and 3 of them write
# ingestion_ts as an F.lit placeholder anyway.
#
# streaming ingestion gap = ingestion_ts - event_timestamp, the AC's definition. Gate =
# no row later than max_delay_seconds. Only meaningful once the table
# is fed incrementally AND ingestion_ts is stamped at write time.
#
# Both emit min/max_event_time so coverage is comparable across modes.
def timeliness_exprs(cfg, run_date, columns):
    ev = F.col(cfg.event_timestamp_col)
    exprs = [
        F.min(ev).cast("string").alias("min_event_time"),
        F.max(ev).cast("string").alias("max_event_time"),
        F.sum(F.when(ev.isNull(), 1).otherwise(0)).alias("null_event_time_rows"),
    ]
    if cfg.timeliness_mode == "streaming":
        ing = F.col(cfg.ingestion_timestamp_col)
        gap = F.unix_timestamp(ing) - F.unix_timestamp(ev)
        exprs += [
            F.min(ing).cast("string").alias("min_ingestion_time"),
            F.max(ing).cast("string").alias("max_ingestion_time"),
            F.max(gap).alias("max_gap_seconds"),
            F.sum(F.when(gap < 0, 1).otherwise(0)).alias("negative_gap_rows"),
            F.sum(F.when(ing.isNull(), 1).otherwise(0)).alias("null_ingestion_time_rows"),
            F.sum(F.when(F.lit(cfg.max_delay_seconds is not None)
                         & (gap > F.lit(cfg.max_delay_seconds or 0)), 1).otherwise(0))
             .alias("delay_violation_rows"),
        ]
    else:
        ev_unix = F.unix_timestamp(ev)
        as_of_unix = F.unix_timestamp(F.date_add(F.to_date(F.lit(run_date)), 1))  # midnight after run_date
        future_cutoff = as_of_unix + F.lit(int(cfg.future_grace_seconds))
        exprs += [
            F.min(as_of_unix - ev_unix).alias("recency_seconds"),   # == as_of - max(event_time)
            F.sum(F.when(ev_unix > future_cutoff, 1).otherwise(0)).alias("future_event_rows"),
        ]
    # S12 Currency: per-row review-cycle gate, valid in either mode. Age basis
    # defaults to ingestion_ts; as-of = midnight after run_date, the same instant
    # batch recency uses.
    if cfg.currency_max_age_seconds is not None:
        cur = resolve_column(cfg.currency_timestamp_col, columns)
        if cur is not None:
            cur_as_of = F.unix_timestamp(F.date_add(F.to_date(F.lit(run_date)), 1))
            age = cur_as_of - F.unix_timestamp(F.col(cur))
            exprs.append(
                F.sum(F.when(age > F.lit(int(cfg.currency_max_age_seconds)), 1).otherwise(0))
                 .alias("stale_rows"))
    return exprs


def parse_timeliness(row, cfg, columns, row_count):
    _rate = lambda n: round(n / row_count, 6) if row_count else 0.0
    metrics = {
        "mode": cfg.timeliness_mode,
        "event_timestamp_col": cfg.event_timestamp_col,
        "min_event_time": row["min_event_time"],
        "max_event_time": row["max_event_time"],
        "null_event_time_count": int(row["null_event_time_rows"] or 0),
    }
    if cfg.timeliness_mode == "streaming":
        delayed = int(row["delay_violation_rows"] or 0)
        negative = int(row["negative_gap_rows"] or 0)
        metrics.update({
            "ingestion_timestamp_col": cfg.ingestion_timestamp_col,
            "min_ingestion_time": row["min_ingestion_time"],
            "max_ingestion_time": row["max_ingestion_time"],
            "max_gap_seconds": int(row["max_gap_seconds"]) if row["max_gap_seconds"] is not None else None,
            "negative_gap_count": negative,
            "negative_gap_rate": _rate(negative),
            "null_ingestion_time_count": int(row["null_ingestion_time_rows"] or 0),
            "delay_violation_count": delayed,
            "delay_violation_rate": _rate(delayed),
            "max_delay_seconds": cfg.max_delay_seconds,
        })
        rules = [{"family": "timeliness", "rule": "no event_time after ingestion_time",
                  "column": cfg.event_timestamp_col, "violations": negative, "passed": negative == 0}]
        if cfg.max_delay_seconds is not None:
            rules.append({"family": "timeliness",
                          "rule": f"ingestion gap <= {cfg.max_delay_seconds}s",
                          "column": cfg.event_timestamp_col,
                          "violations": delayed, "passed": delayed == 0})
    else:
        future = int(row["future_event_rows"] or 0)
        recency = int(row["recency_seconds"]) if row["recency_seconds"] is not None else None
        metrics.update({
            "recency_seconds": recency,
            "future_count": future,
            "future_rate": _rate(future),
            "max_recency_seconds": cfg.max_recency_seconds,
        })
        rules = [{"family": "timeliness", "rule": "no future event_time",
                  "column": cfg.event_timestamp_col, "violations": future, "passed": future == 0}]
        if cfg.max_recency_seconds is not None:
            over = 1 if (recency is not None and recency > cfg.max_recency_seconds) else 0
            rules.append({"family": "timeliness", "rule": f"recency <= {cfg.max_recency_seconds}s",
                          "column": cfg.event_timestamp_col, "violations": over, "passed": over == 0})
    if cfg.currency_max_age_seconds is not None:
        cur = resolve_column(cfg.currency_timestamp_col, columns)
        if cur is None:
            rules.append({"family": "timeliness",
                          "rule": f"{cfg.currency_timestamp_col} exists in the table",
                          "column": cfg.currency_timestamp_col, "violations": 1, "passed": False})
        else:
            stale = int(row["stale_rows"] or 0)
            metrics["currency"] = {"timestamp_col": cur,
                                   "max_age_seconds": int(cfg.currency_max_age_seconds),
                                   "stale_count": stale, "stale_rate": _rate(stale)}
            rules.append({"family": "timeliness",
                          "rule": f"rows within review cycle ({cur} age <= {int(cfg.currency_max_age_seconds)}s)",
                          "column": cur, "violations": stale, "passed": stale == 0})
    return {"metrics": metrics, "rules": rules}


# victim exclusion #
def victim_exclusion_exprs(col_name, allowed, columns):
    """One counter: rows NOT attested as confirmed bad actor.

    NULL is a violation here, unlike consistency: an unattested record must not
    sit in the matchable population (governance framework S7.3 makes both
    attestations mandatory at ingestion), so a missing value fails the same as a
    wrong one.
    """
    c = resolve_column(col_name, columns)
    if c is None:
        return []
    return [F.sum(F.when(F.col(c).isin(list(allowed)), 0).otherwise(1))
             .alias("victim_violation_rows")]


def parse_victim_exclusion(row, col_name, allowed, columns, row_count):
    c = resolve_column(col_name, columns)
    if c is None:
        # Fail closed: a guard pointed at a column the table lacks is OFF, and an
        # off guard must be loud (see parse_completeness for the live incident).
        return {"metrics": {"column": col_name, "allowed_values": sorted(allowed),
                            "violation_count": None, "violation_rate": None},
                "rules": [{"family": "victim_exclusion",
                           "rule": f"{col_name} exists in the table",
                           "column": col_name, "violations": 1, "passed": False}]}
    bad = int(row["victim_violation_rows"] or 0)
    return {"metrics": {"column": c, "allowed_values": sorted(allowed),
                        "violation_count": bad,
                        "violation_rate": round(bad / row_count, 6) if row_count else 0.0},
            "rules": [{"family": "victim_exclusion",
                       "rule": f"{c} attests confirmed bad actor (nulls fail)",
                       "column": c, "violations": bad, "passed": bad == 0}]}

