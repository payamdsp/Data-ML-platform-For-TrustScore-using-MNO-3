"""The fraud taxonomy: ten categories, and the feed labels that map onto them.

Two upstream feeds label fraud, and neither uses the other's vocabulary. The
Phase 1 feed uses short internal codes (``FA``, ``TW21``) mixed with prose
("First party fraud"). The NBC feed uses upper-case descriptions, several of them
compound (``CREDIT_CARD_BUST OUT``, ``INTERAC_FRAUD_RECIPIENT_PH``). Neither
vocabulary is stable and neither is a taxonomy: ``FA`` and ``FRCH`` and ``TW21``
all describe first-party fraud, and no consumer of the data should have to know
that.

So this module owns one mapping per feed onto a single ten-category taxonomy, and
the pipeline works in categories from the point of ingest onward. The categories
are not a hierarchy and are not ordered by severity; :data:`CATEGORY_ORDER` is a
*display* order, fixed so that a chart's legend does not reshuffle between runs
when a rare category happens to be absent.

Matching is case- and whitespace-insensitive, which the notebook's version was
not. It compared raw feed strings against dict keys directly, so ``"first party
fraud"`` and ``"First Party Fraud"`` both missed the key ``"First party fraud"``
and fell through to the unmapped bucket. Since the unmapped bucket was simply
labelled with the raw string and never audited, the effect was a long tail of
one-row "categories" that were really casing variants of four real ones.

The feeds' *own* names are a separate problem from their vocabularies, and
:func:`resolve_fraud_source` is the answer to it. A feed is called ``phase1``
here, ``Phase1-v1.0`` in the prefix its integrated table is written to, and
``Phase1-v1.0`` again in that table's ``fraud_source`` column. The notebook
treated the three as one string, which cost it the entire fraud population; the
docstring of that function has the detail, because it is the most expensive
defect in this rebuild's inheritance.

Nothing here decides what a *label* is — every row in the fraud feed is fraud,
whatever its category. The categories exist for stratification and for reporting:
"the model finds synthetic-identity fraud and misses mule accounts" is the
sentence this taxonomy is here to let someone write.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

__all__ = [
    "CATEGORIES",
    "CATEGORY_ORDER",
    "CATEGORY_SHORT_LABEL",
    "FRAUD_SOURCES",
    "NBC_FRAUD_TYPE_TO_CATEGORY",
    "PHASE1_FRAUD_TYPE_TO_CATEGORY",
    "UNMAPPED_CATEGORY",
    "category_for",
    "normalize_fraud_type",
    "resolve_fraud_source",
    "short_label",
]

CATEGORY_1 = "category_1_synthetic_id_fraud"
CATEGORY_2 = "category_2_true_name_fraud"
CATEGORY_3 = "category_3_first_party_fraud"
CATEGORY_4 = "category_4_impersonation_scam_otvc"
CATEGORY_5 = "category_5_uncategorized_nbc"
CATEGORY_6 = "category_6_other_nbc"
CATEGORY_7 = "category_7_mule_nbc"
CATEGORY_8 = "category_8_ato_nbc"
CATEGORY_9 = "category_9_phishing_nbc"
CATEGORY_10 = "category_10_scam_nbc"

#: Assigned when a feed emits a fraud type nothing maps. Distinct from
#: ``category_5_uncategorized_nbc``, which is a real NBC category meaning "the
#: NBC analyst did not categorise this"; this one means "our mapping does not
#: know this string", which is a defect in this module and must be visible as
#: such rather than hidden inside a legitimate bucket.
UNMAPPED_CATEGORY = "category_unmapped"

#: Display order. Numeric, matching the category names, which is also the order
#: they were defined in by the fraud team.
CATEGORY_ORDER: Tuple[str, ...] = (
    CATEGORY_1,
    CATEGORY_2,
    CATEGORY_3,
    CATEGORY_4,
    CATEGORY_5,
    CATEGORY_6,
    CATEGORY_7,
    CATEGORY_8,
    CATEGORY_9,
    CATEGORY_10,
)

CATEGORIES = frozenset(CATEGORY_ORDER) | {UNMAPPED_CATEGORY}

CATEGORY_SHORT_LABEL: Mapping[str, str] = {
    CATEGORY_1: "Category 1 - Synthetic ID",
    CATEGORY_2: "Category 2 - True Name",
    CATEGORY_3: "Category 3 - First Party Fraud",
    CATEGORY_4: "Category 4 - Impersonation / Scam-OTVC",
    CATEGORY_5: "Category 5 - Uncategorized (NBC)",
    CATEGORY_6: "Category 6 - Other (NBC)",
    CATEGORY_7: "Category 7 - Mule (NBC)",
    CATEGORY_8: "Category 8 - ATO (NBC)",
    CATEGORY_9: "Category 9 - Phishing (NBC)",
    CATEGORY_10: "Category 10 - Scam (NBC)",
    UNMAPPED_CATEGORY: "Unmapped fraud type",
}

#: The feed identifiers, which are also the S3 path segments the fraud data is
#: partitioned by.
FRAUD_SOURCES: Tuple[str, ...] = ("phase1", "nbc")


def normalize_fraud_type(raw: Optional[str]) -> str:
    """Canonical form of a feed's fraud-type string, for lookup.

    Upper-cased, with runs of whitespace collapsed to a single space and the ends
    trimmed. Punctuation is left alone: ``CREDIT_CARD_BUST OUT`` and
    ``P2P/INTERAC`` carry their underscores and slashes meaningfully, and
    stripping them would merge distinct NBC types.
    """
    if raw is None:
        return ""
    return " ".join(str(raw).split()).upper()


def _normalized_map(pairs: Mapping[str, str]) -> Dict[str, str]:
    """Re-key a mapping by :func:`normalize_fraud_type`, refusing collisions.

    A collision means two source spellings normalise to the same key with
    *different* categories — which is a real disagreement about what a fraud type
    means, not something to resolve by whichever entry came last in the dict
    literal.
    """
    out: Dict[str, str] = {}
    for raw_key, category in pairs.items():
        key = normalize_fraud_type(raw_key)
        existing = out.get(key)
        if existing is not None and existing != category:
            raise ValueError(
                f"fraud type {raw_key!r} normalises to {key!r}, which is already "
                f"mapped to {existing!r}; it cannot also mean {category!r}."
            )
        out[key] = category
    return out


PHASE1_FRAUD_TYPE_TO_CATEGORY: Mapping[str, str] = _normalized_map(
    {
        "FA": CATEGORY_3,
        "FIDV": CATEGORY_3,
        "FPNP": CATEGORY_3,
        "FRCH": CATEGORY_3,
        "First party fraud": CATEGORY_3,
        "TW21": CATEGORY_3,
        "Impersonation": CATEGORY_4,
        "Scam-OTVC Deception": CATEGORY_4,
        "Synthetic Identity Fraud": CATEGORY_1,
        "True Name Fraud": CATEGORY_2,
    }
)

NBC_FRAUD_TYPE_TO_CATEGORY: Mapping[str, str] = _normalized_map(
    {
        "TRUE IDENTITY FRAUD": CATEGORY_2,
        "OTHER": CATEGORY_6,
        "SYNTHETIC ID FRAUD": CATEGORY_1,
        "FALSIFIED DOCUMENTS": CATEGORY_1,
        "BUST-OUT": CATEGORY_3,
        "CREDIT_CARD_FIRST PARTY ABUSE": CATEGORY_3,
        "CREDIT_CARD_FALSE APPLICATION": CATEGORY_1,
        "CREDIT_CARD_BUST OUT": CATEGORY_3,
        "BANKING_DEPOSIT FRAUD": CATEGORY_5,
        "BANKING_ACCOUNT OPENING FRAUD": CATEGORY_7,
        "BANKING_P2P/INTERAC - COMPLICIT CUSTOMER": CATEGORY_5,
        "IDENTITY_THEFT_RECIPIENT_PH": CATEGORY_8,
        "FAMILY_FRAUD_RECIPIENT_PH": CATEGORY_3,
        "INTERAC_FRAUD_RECIPIENT_PH": CATEGORY_7,
        "PHISHING_RECIPIENT_PH": CATEGORY_9,
        "SCAM_RECIPIENT_PH": CATEGORY_10,
        "VIRUS_TROJAN_RECIPIENT_PH": CATEGORY_5,
        "INTERCEPTED_FUNDS_RECIPIENT_PH": CATEGORY_7,
        "OTHER_RECIPIENT_PH": CATEGORY_6,
    }
)

_MAPPINGS: Mapping[str, Mapping[str, str]] = {
    "phase1": PHASE1_FRAUD_TYPE_TO_CATEGORY,
    "nbc": NBC_FRAUD_TYPE_TO_CATEGORY,
}


def resolve_fraud_source(raw: Optional[str]) -> Optional[str]:
    """Which of the two feeds an arbitrary feed identifier refers to, or ``None``.

    The two feeds are called different things in the three places their identity
    appears, and this function is what reconciles them. The taxonomy keys are
    ``phase1`` and ``nbc``. The prefixes the integrated fraud tables are written
    to are ``…/fraud/Phase1-v1.0/`` and ``…/fraud/NBC-v1.0/``. The
    ``fraud_source`` column inside those tables carries the same versioned
    strings, ``"Phase1-v1.0"`` and ``"NBC-v1.0"``.

    The notebook treated all three as one vocabulary. Its config declared
    ``fraud_sources = ("phase1", "nbc")`` and its discovery built
    ``<testing_root>/fraud/<source>/`` from exactly those values, so it listed
    two prefixes that do not exist. That listing raised, the ``except Exception``
    around it downgraded the failure to a warning, and ``discover_fraud_data_paths``
    returned an empty list — which meant *no* fraud customers were removed from
    the training population and the model was fitted on a sample containing the
    thing it was being trained to find as an anomaly. Nothing in the output said
    so. The same mismatch hit the category lookup from the other end: the
    ``fraud_source`` column's ``"Phase1-v1.0"``, lower-cased, is not the key
    ``phase1`` either, so every row's category resolved against a missing
    mapping.

    So a feed name is *resolved*, not compared. A name is attributed to the one
    taxonomy key it contains, which makes it robust to the version suffix the
    feed will bump; a name containing both keys, or neither, resolves to
    ``None`` rather than to a guess, and :func:`category_for` then raises naming
    the value. Two feeds cannot be told apart by a rule that matched both.
    """
    text = normalize_fraud_type(raw).lower()
    if not text:
        return None
    matched = [source for source in FRAUD_SOURCES if source in text]
    if len(matched) != 1:
        return None
    return matched[0]


def category_for(fraud_source: Optional[str], fraud_type: Optional[str]) -> str:
    """The taxonomy category for one feed's fraud type.

    Returns :data:`UNMAPPED_CATEGORY` for a type this module does not know and
    for a row with no type at all. It does *not* raise, because a single
    unrecognised fraud type must not end a training run — but every caller that
    assigns categories in bulk is expected to count the unmapped rows and put
    the count in its manifest, which is how the mapping gets extended.

    An unknown ``fraud_source`` does raise. There are two feeds, the source is a
    partition value in the path the row was read from, and a third value means
    the reader is looking at data it does not understand.

    The source goes through :func:`resolve_fraud_source`, so the feed's own
    versioned spelling (``"NBC-v1.0"``) is accepted alongside the taxonomy key.
    """
    source = resolve_fraud_source(fraud_source)
    if source is None:
        raise KeyError(
            f"unknown fraud source {fraud_source!r}; expected one of {list(_MAPPINGS)}"
        )
    return _MAPPINGS[source].get(normalize_fraud_type(fraud_type), UNMAPPED_CATEGORY)


def short_label(category: Optional[str]) -> str:
    """A human-readable label for a category, for chart legends and reports."""
    if not category:
        return CATEGORY_SHORT_LABEL[UNMAPPED_CATEGORY]
    return CATEGORY_SHORT_LABEL.get(str(category), str(category))
