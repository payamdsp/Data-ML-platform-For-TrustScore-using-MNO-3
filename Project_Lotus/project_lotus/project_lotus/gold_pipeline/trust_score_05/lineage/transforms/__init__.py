"""The Gold stages, plus what they share.

Reading order is the pipeline order, and each stage's docstring explains the one
decision that shaped it:

    canonical_events  carrier vocabulary -> one vocabulary, with the five types
                      no carrier sends worked out from the shape of the history
    lifecycle         events -> one continuous stretch of a phone number being
                      alive at one carrier
    boundaries        the enrichment feeds -> starts and ends the carrier feeds
                      could only guess at
    tu_lifecycle      TransUnion's porting history -> the stretches of a number's
                      life spent at a carrier we hold no feed for
    accounts          lifecycles -> chains joined by number changes
    customers         accounts -> chains joined by ports
    normalized        all of the above flattened back to one row per event

    imei_map          the second lineage, independent of the first except for
                      the optional close-on-lifecycle-end coupling

``common`` holds what a difference between stages would make a bug - surrogate
keys above all - and ``chains`` holds the traversal that accounts and customers
both need, because a chain of lifecycles and a chain of accounts are the same
problem one level apart.

Two of the stages are written as plain Python over one phone number's history
rather than as Spark expressions: :func:`lifecycle.walk_phone` and
:func:`imei_map.walk_device_history`. Both are sequential by nature - what an
event does depends on what is currently open - and both have edge cases that a
chain of window functions would express in a way nobody could check against the
design document. They are callable from a test with a handful of dictionaries
and no SparkSession, which is the point.
"""

from . import (  # noqa: F401
    accounts,
    boundaries,
    canonical_events,
    chains,
    common,
    customers,
    imei_map,
    lifecycle,
    normalized,
    tu_lifecycle,
)

__all__ = [
    "accounts",
    "boundaries",
    "canonical_events",
    "chains",
    "common",
    "customers",
    "imei_map",
    "lifecycle",
    "normalized",
    "tu_lifecycle",
]
