# Known issues in the previous feature library

The v23 feature-preparation notebook that produced the training tables for the current Trust Score
models was read end to end. This document records what that code actually does, where it is wrong,
and where it is not wrong but is arranged in a way this package deliberately departs from.

It exists for two audiences. For anyone reading `trust_score_05/features/`, it is the list of
defects the port sets out to fix, and each finding names the function it came from so the fix can be
traced. For anyone who has to interpret a model that was trained on the old features, it is the list
of columns that were not measuring what their names claim, which matters because several of them
were not empty — they were plausible and wrong.

The findings are split three ways:

- **Correctness** — the output disagrees with the column's name or with itself. These are bugs.
- **Structural** — the logic is defensible but the arrangement makes a class of mistake easy, and
  the mistake was in fact made. These need a change of shape, not a patch.
- **Gaps** — features the design asks for that no code produces.

Section 4 is separate from all three and is about this package rather than the old one: what the
replacement got wrong on the way, and what remains unverified.

One property is shared by every finding in section 1, and it is the reason the list is as long as it
is: **not one of them raised.** Each produced a column of nulls, a wrong number, or a duplicated
row, and the notebook ran to completion and wrote a table. A feature table has no schema anyone
checks against, so a column that is entirely null looks exactly like a column describing a
population that mostly did not do the thing. Several of these survived into production models.

Sources reviewed: `01_prepare_v23_BNS_2024-09.ipynb`, the whole of it — the event normalisation
cells, the eleven feature calculators, the EnStream weighting cells, and the two assembly functions.

---

## 1. Correctness findings

### 1.1 A missing comma silently deleted two feature families

*`assemble_scope_features`, the `generic_event_families` literal.*

The list of families that receive the generic feature set was written out by hand:

```python
generic_event_families = [
    "device_change", "sim_change", "account_reactivation",
    "account_suspension", "account_cancellation"      # <- no comma
    "phone_number_change", "mno_port", "enstream_api_call",
]
```

Python concatenates adjacent string literals, so the fifth and sixth entries became the single
string `"account_cancellationphone_number_change"`. That value matches no row of `event_family`.

The consequences are three, and the third is the expensive one:

1. `account_cancellation` got no features at all. Cancellation is the end of an account, and the
   burst of activity immediately before a cancellation is one of the patterns the model exists to
   find.
2. `phone_number_change` got no features at all. Number change is the derived event that the whole
   MSISDN lineage pipeline exists to produce.
3. A full generic feature set — around forty columns per scope, so a hundred and twenty in total —
   was computed for the concatenated family and joined onto the output. Every one of those columns
   is null for every row, under names like
   `cust_account_cancellationphone_number_change_cnt_7d`. They passed through feature selection,
   where a constant column is dropped by a variance filter if there is one and carried if there is
   not, and they are in the shipped feature list.

The fix is not to add the comma. `GENERIC_EVENT_FAMILIES` in `config.py` is now derived from
`CANONICAL_EVENT_FAMILY`:

```python
GENERIC_EVENT_FAMILIES: Tuple[str, ...] = tuple(sorted(set(CANONICAL_EVENT_FAMILY.values())))
```

A family that exists in the event table now cannot be absent from the feature set, and a typo can no
longer produce a family name that is not a family.

### 1.2 `STATUS_CHANGE_C` was spelled with an underscore

*`DEFAULT_STATE_TRANSITIONS`, via the `state_transition_event_types` literal.*

Every status change type in the carrier feed is hyphenated — `STATUS_CHANGE-A`, `STATUS_CHANGE-S`.
The state transition list spelled the cancellation type `STATUS_CHANGE_C`, with an underscore. The
calculator tests membership with `F.col("event_type") == target`, so the comparison was false for
every row.

Cancellation appears in the transition matrix as both a source and a target, so with nine event
types this removed sixteen of the seventy-two ordered pairs: eight of the form "how long after a
cancellation until *X*" and eight of the form "how long after *X* until a cancellation". Each pair
contributes a days-to-transition column and a velocity column, at three scopes — around ninety-six
columns of nulls.

The pair that matters most is `(DEVICE_CHANGE, STATUS_CHANGE-C)`: a device change followed shortly
by a cancellation is the signature of a device being taken and the account being abandoned. It was
among the columns that did not exist.

Note that this defect and 1.1 interact. `account_cancellation` had no frequency or recency features
because of the comma, *and* no transition features because of the underscore. The cancellation event
is in the event table, is counted by the lineage pipeline, and contributed nothing at all to the
model.

### 1.3 `account_activation` was never in the feature set

*`assemble_scope_features`, same literal as 1.1.*

`STATUS_CHANGE-AA` maps to the family `account_activation` and the family is present in the event
table, but the hand-written list simply does not name it — this is independent of the missing comma
and would have remained after fixing it.

Activation is the start of an account. Its absence means the feature set can see that an account was
cancelled or suspended but not that it was recently opened, and account age at the reference moment
is one of the strongest priors in fraud. Some of that signal is recovered by
`cust_days_since_first_seen` in the stability features, which is computed from the lineage interval
rather than from an event, which is presumably why the gap was not noticed.

Deriving the family list from the mapping, as in 1.1, closes this too.

### 1.4 The partner-volume weight join fanned out the EnStream event rows

*`add_partner_volume_weights`.*

EnStream events carry the partner that made the call, and partners differ in volume by orders of
magnitude, so each event is weighted by roughly the inverse square root of its partner's volume:

```python
weight = clamp(sqrt(avg_partner_volume / partner_volume), min_weight, max_weight)
```

The weights are computed per `(reference, service_provider_id, use_case)` — correctly. They were
then joined back onto the event rows on `reference_cols` alone:

```python
weighted = events.join(partner_weights, reference_cols, "left")
```

The right side has one row per partner grain per reference, not one row per reference. The join is
therefore not a lookup but a cross product: an event belonging to a reference that has *n* distinct
partner grains comes out as *n* rows, each carrying a different partner's weight.

This is worse than a wrong weight. Every weighted count feature is a `sum` over these rows, so each
count was multiplied by the reference's partner breadth. The feature intended to *remove*
partner-volume bias became the single largest source of partner-volume bias in the table, and it did
so multiplicatively, so the effect is strongest exactly where the correction was supposed to matter
most. The unweighted counts computed from the same frame are inflated by the same factor.

`with_partner_volume_weights` in `enstream.py` joins on the full partner grain:

```python
partner_group_cols = [*reference_cols, *partner_cols]
weighted = weighted.join(
    partner_weights.select(*partner_group_cols, weight_col),
    partner_group_cols,
    "left",
).withColumn(weight_col, F.coalesce(F.col(weight_col), F.lit(1.0)))
```

and the row count of the frame is asserted unchanged across the join in the tests. The coalesce is
there because a partner grain that appears in an event but not in the volume table — possible when
the volume table is computed over a narrower window — should weight as one rather than as null,
which would erase the event from every sum.

### 1.5 The state-transition tie-break was inverted, losing simultaneous events

*`calculate_state_transition_features`, the `next_event_window` definition.*

"How long until the next *target* event" is computed with a window ordered by timestamp descending
over `[unboundedPreceding, currentRow]`, so the frame from any row covers every event at or after
that row's timestamp, and the minimum target timestamp in the frame is the next target.

At equal timestamps the ordering within the frame decides whether a simultaneous target event is
inside it. The secondary sort key was

```python
F.when(F.col("event_type") == target, 1).otherwise(0).asc()
```

Ascending puts the non-target source row *first*, so a target event at the same instant sorts after
it and falls outside the frame. The transition is then either missed entirely, or — if the same
target occurs again later — reported at that later gap, which is the more damaging outcome because
it is a plausible number rather than a null.

Simultaneous events are not a hypothetical here. A device change and a SIM change arriving in the
same carrier batch share a timestamp, because the carrier stamps the batch rather than the event.
That pair — a new device and a new SIM at the same moment — is one of the sharpest signals in the
feature set, and it was recorded as either "never" or "some number of days later".

`next_event_window` in `expressions.py` orders the flag `.desc()`, so a target at the same instant
sorts before the source, lands in the frame, and the transition is recorded as taking zero days.

### 1.6 The EnStream wrapper passed the scope name where the prefix belonged

*`calculate_enstream_partner_only_features`.*

Every feature name is prefixed `cust`, `acct` or `msisdn`. This function took a `prefix` argument
and then passed `scope` to all three calculators beneath it, so the EnStream-only columns came out
named `customer_enstream_api_call_*` while every other column at the same scope was
`cust_enstream_api_call_*`.

The columns exist and hold correct values, so the effect is not a null column. It is worse in a
subtler way: any downstream code that selects features by prefix — and the preprocessing stage
selects by prefix — silently omits the EnStream features at the customer and account scopes.
Whether they reached a model depends on which selection path ran. At the MSISDN scope the scope name
and the prefix happen to be the same string, so the bug is invisible there, which is presumably why
it survived.

`scope_prefix()` in `config.py` is now the only source of these three strings and every caller goes
through it.

### 1.7 `cfg` was not forwarded to nested calculators

*`calculate_generic_event_feature_set` calling `calculate_time_to_k_events`;
`calculate_burstness_features` calling `calculate_event_frequency_features`.*

Several calculators call others, and some of those calls omitted `cfg`, so the callee fell back to
its default argument. A caller who narrowed `lookback_days` — which the sweep code does, to build
cheaper feature sets for model selection — got the narrowed windows in most families and the
defaults in the rest. The resulting table has columns for windows the caller did not ask for and, in
the reverse direction, a `KeyError`-free silent mismatch between what the run was labelled as and
what it computed.

Every call in this package forwards `cfg`. The burstness ratios are also now taken from
`cfg.ratio_windows` rather than hardcoded as `(1, 3)`, `(7, 30)`, `(30, 90)`, and
`FeatureConfig.__post_init__` raises if a ratio names a window that `lookback_days` does not
contain. A narrowed config now fails at construction with a message naming the offending ratio,
rather than deep inside a helper with an unresolved-column error.

### 1.8 `build_reference_df` raised `UnboundLocalError` on its own error path

*`build_reference_df`.*

The function accepts either a single `reference_ts` for the whole population or a per-MSISDN
reference frame, and dispatches on which was given. Given neither, it fell through both branches to
`return reference_df`, where `reference_df` was never assigned — so the diagnostic a caller received
for the most likely misuse of the function was `UnboundLocalError: local variable 'reference_df'
referenced before assignment`.

It now raises a `ValueError` that names both arguments and says what each one is for.

---

## 2. Structural findings

### 2.1 The scope-to-event join ran once per family

*`assemble_scope_features`.*

The old code attached events to the scope snapshot inside the per-family loop, so the range join
between the snapshot and the event table ran once for each of the eight families it looped over,
each time to produce a disjoint subset of the same result. The join is the expensive step in the
whole pipeline — it is a range join, so Spark plans it as a broadcast nested loop or a sort-merge
with a filter, neither of which is cheap.

`assemble_scope_features` now calls `attach_events_to_scope` once per scope and each calculator
filters the attached frame by family. Three joins instead of twenty-four.

### 2.2 The EnStream weights were a dict of one frame per window

*`add_partner_volume_weights` returned `Dict[int, DataFrame]`.*

One frame per lookback window, each a full copy of the scoped event rows with one weight column.
Five windows meant five passes over the events and five joins to reassemble. `with_partner_volume_weights`
returns one frame carrying every window's weight as a separate column, computed in a single pass.

This is a performance finding, not a correctness one — but it is also what made 1.4 hard to see,
because the fan-out happened five times in five places that looked like five separate correct
lookups.

### 2.3 Ratios were computed against counts the caller could not see

*`calculate_burstness_features`.*

The burstness ratios need the count features as inputs, so the function called the frequency
calculator itself and left its columns on the output. The frequency calculator was then called again
by the assembly function, and `join_feature_frames` dropped the second copy — so the shipped counts
came from inside the burstness calculator, with whatever config that call had been given, which per
1.7 was not necessarily the caller's.

The counts are still borrowed, but they are now dropped again before returning:

```python
borrowed_cols = feature_columns(frequency, entity_cols)
...
return result.drop(*borrowed_cols)
```

so the frequency columns in the output come from exactly one call, the one the assembly function
makes.

---

## 3. Gaps

### 3.1 Nulls were filled inside the calculators

Several calculators returned `0` where the correct answer is "undefined": a seven-over-thirty ratio
for an entity with no events in thirty days, a mean inter-event gap for an entity with one event, a
slope for an entity with one point. Zero is not a neutral filler for any of these. It places the
entity at the quiet end of a scale it is not on at all, and it is indistinguishable from an entity
that genuinely had a ratio of zero.

`safe_divide` in `expressions.py` returns null on a zero denominator, and no calculator in this
package fills a null. What to do about them is a preprocessing decision — impute, indicator column,
or drop the row — and it can only be made if the null survives to the stage that makes it.

### 3.2 There is no event-weight provenance

`event_weight` is a single column, and for EnStream events it is a partner-volume correction while
for carrier events it is one. Nothing records which. A reader of the feature table cannot tell a
weighted sum from a count, and a future feed with a different weighting scheme would be
indistinguishable again. `event_source` partially covers this by naming the feed, and the
per-window EnStream weights are now separate named columns, but the generic `event_weight` remains
overloaded.

---

## 4. Defects in this package

The honest counterweight to a document that otherwise reads as a list of somebody else's mistakes.

### 4.1 A dead placeholder helper shipped in a draft of `events.py`

An early draft of `build_changes_table` contained a helper `_selected_schema_names` that returned an
empty tuple and was used in a comprehension guarded by `or True`, so it had no effect on the output
and no effect on the control flow. It was a stub that was never finished and never removed. It was
caught by reading the file back before commit, not by any tool, and replaced with a post-select
filter on `cfg.event_dedupe_keys()`.

### 4.2 The package now runs against Spark, and running it found two more defects

This section previously read "nothing in this package has been run against Spark". That is no longer
true and the correction is worth keeping visible, because the two findings numbered below — 4.4 and
4.5 — were both found by executing the code, and neither was visible to the careful reading that
produced findings 1.1 through 3.2.

A local JVM-backed Spark session is now available and `tests/features/` holds 298 tests across seven
modules — `test_config.py` (30), `test_expressions.py` (44), `test_snapshots.py` (36),
`test_events.py` (52), `test_calculators.py` (66), `test_enstream.py` (42) and `test_assemble.py`
(28). Every one of them constructs real DataFrames through a real `SparkSession`; the whole suite is
marked `pytest.mark.spark` and skips cleanly where pyspark is absent, so a machine without a JVM
still gets a green `pytest tests/`. Each of findings 1.1 to 1.8 now has at least one test that names
it in its docstring and fails if the fix is reverted, so those eight are demonstrated rather than
argued.

Two honest caveats remain. The tests run against a single local executor on frames of a few dozen
rows, so they settle *semantics* and say nothing about behaviour at scale — a shuffle that is
correct on ten rows and ruinous on ten billion looks identical here. And the mutation sweep that
would prove each regression test is load-bearing (the standard applied to the ML and common
packages, where 200 mutations across six drivers were all caught) is written as
`mutate_features.py` but has not been run to completion: at roughly thirty seconds per Spark module
the full fifty-mutation sweep is a half-hour of wall clock, and it was deferred. The driver is the
outstanding work; until it runs, the *tests* are verified to pass against the fixed code and to fail
against the two defects that were found by accident, but not verified to fail against every defect
they claim to guard.

### 4.3 The hour/day lookback split is inherited, not justified

`FeatureConfig.lookback_hour_windows` makes the one- and three-day windows exact-hour windows and
the rest calendar-day windows. This is what the old code did, and it is kept because the shipped
models were fitted against features with that boundary. It is not defended as correct. A single
convention would be easier to explain, and switching to one is a retraining, not a patch — the
config field exists so that the switch is a one-line change when there is appetite for the
retraining.

### 4.4 Every stability feature was read from the wrong grain, twice

`calculate_stability_features` in `calculators.py` is supposed to answer two questions about an
entity: how long has it existed, and how wide is it. Both answers were wrong, in two separate
drafts, for the same reason — the function read `snapshot_scope[scope]` when it needed
`snapshot_scope["base"]`.

`build_scope_snapshots` in `snapshots.py` produces both. The `"base"` entry is one row per lineage
interval: every `(msisdn, account, customer)` assignment the entity has ever held, each with its own
`from_ts` and `to_ts`. The scope entries are aggregates over that — `snapshots.py:297` collapses
each entity to a single row carrying `min(from_ts)` and `max(to_ts)`, and the grouping keys of the
scope are the only entity columns that survive.

The first draft read breadth — `{prefix}_distinct_account_cnt` and `{prefix}_distinct_msisdn_cnt` —
from the scope entry. At customer scope `acct_id` and `msisdn` are not columns of that frame at all,
so the `has_col` fallback emitted a typed null and every customer in production reported an unknown
number of accounts. At account scope `acct_id` *is* the grouping key, so `countDistinct` over a
single row returned the constant 1, and the feature that was meant to distinguish a one-line account
from a fifty-line account was the literal 1 for both. The same held for `msisdn` at msisdn scope.

The second draft fixed breadth and left tenure behind. `{prefix}_days_since_first_seen` is
`datediff(reference_ts, min(from_ts))` and `{prefix}_days_since_current_assignment_start` is
`datediff(reference_ts, max(from_ts))`. Against the `"base"` intervals those are different numbers.
Against the scope entry, where the aggregation has already reduced the intervals to the single
surviving `min(from_ts)`, `max` can only re-derive the value `min` produced — so the two columns
were bit-identical for every entity at every scope. A three-year customer who added a line last week
scored the same on assignment recency as a three-year customer who had never changed anything, which
is precisely the signal the feature exists to carry.

Both halves now read `snapshot_scope["base"]`; the scope entry supplies only the entity skeleton
that the features are joined onto, so a silent entity still gets a row. A missing `"base"` entry
raises `KeyError` rather than falling back, on the explicit grounds that a silent fallback is how
this defect got in twice. The regression test
`test_days_since_first_seen_is_the_earliest_interval_and_the_assignment_the_latest` asserts the two
values are 1096 and 121 *and* asserts they differ, so a version that reads the aggregated snapshot
cannot pass by coincidence.

The general lesson, which is why this is written up at length rather than noted as a typo: a
DataFrame that has the right *columns* is not a DataFrame at the right *grain*, and Spark will not
tell you the difference. Both drafts ran, produced output of the correct shape, and were wrong.

### 4.5 Zero and null mean different things in the count features, deliberately

An entity with no events of a family in the window gets `null` for that family's counts, not `0`.
An entity that has such an event but outside the window gets `0`. This asymmetry is a consequence of
the join order and it is being recorded rather than repaired.

`filtered_events` in `calculators.py` restricts the scoped frame to one family. The scoped frame is
produced by `attach_events_to_scope`, which is a left join, so an entity with no events survives
with a null `event_family` — and a null does not equal the family, so the row is dropped by the
filter. The entity then forms no group in the aggregate, and `safe_join` fills its features from the
miss side of the final join: null. An entity that *does* have an event of the family, but older than
the window, forms a group whose `sum(when(condition, 1))` is 0, and gets 0.

Three separate tests hit this against three different code paths before it was understood, which is
itself the evidence that it is a property of the design and not one oversight. It is left alone
because either meaning is defensible — "we have never seen this" and "we have seen this, just not
lately" are different facts, and collapsing them loses information — and because the downstream
imputation in `trust_score_05.ml.preprocessing` treats a null count as zero anyway, so the
distinction costs nothing at training time. The tests now assert `is None` explicitly and carry a
non-null control on a neighbouring entity, so unifying the two is a deliberate change that arrives
with a failing test attached rather than a silent shift in the feature distribution.

One related inconsistency in the same area: `calculate_temporal_entropy_features` bases its output
frame on the *filtered* events, while the frequency and recency calculators base theirs on the whole
scoped frame. The practical effect is that entropy columns are absent for a silent entity where the
frequency columns are present-and-null. It is harmless downstream for the same imputation reason,
and it is noted here so the next person to touch that function knows the difference is not
load-bearing.
