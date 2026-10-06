"""Trust Score 0.5 Silver -> Gold incremental pipelines (EMR / PySpark).

Two entrypoints:

    trust_score_05.lineage.jobs.lineage_job     phone -> lifecycle -> account -> customer
    trust_score_05.lineage.jobs.imei_map_job    phone <-> device associations

Both work the same way. A run picks up the Silver rows written since the last
successful run, works out which phone numbers those rows can possibly affect
(the *slice*), rebuilds every Gold row for exactly those phone numbers from
their complete history, and merges the result.

That is the one structural difference from Bronze -> Silver, and everything
else follows from it. Silver appends immutable events, so a batch that arrives
twice can simply be dropped. Gold holds *states* - a lifecycle's end, an
account's shape, a device segment's interval - and a single late event can move
a boundary that was written months ago. So Gold merges rather than appends, and
"idempotent" means a second run converges on the same rows rather than writing
nothing at all.

The full design, including the scenario catalogue every test is drawn from,
is in ``docs/lineage/incremental_design.md``.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
