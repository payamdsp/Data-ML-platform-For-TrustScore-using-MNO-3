# What was wrong with the ML pipeline, and what is still open

This document is the counterpart to
`trust_score_05/features/docs/feature_library_findings.md`. That one is about
the Spark feature library; this one is about everything downstream of it —
dataset discovery, data-quality screening, feature selection, preprocessing,
the model zoo, the hyperparameter grids, the sweep, evaluation, and drift.

It exists because the previous version of this pipeline lived in notebooks, and
notebooks do not have a place to record "we know about this". The findings below
were all found while porting that code into `trust_score_05.ml` and writing
tests for it. Every one of them is fixed in the package. They are written down
anyway, at length, for two reasons: several of them will be reintroduced by
anyone who ports the same logic again from the same notebooks, and every one of
them was a *silent* failure — the pipeline ran, produced numbers, and the
numbers were wrong. Those are the expensive kind, and the only defence is
knowing they exist.

Sections 1 and 2 are the defects. Section 3 is what is not finished. Section 4
is the things that look like defects, are not, and should be left alone.

---

## 1. Defects that produced wrong numbers without failing

### 1.1 The fraud population was empty, so the model was trained on fraud

Fixed in `6e0e2df`. This is the worst one in the document and it is worth
reading even if you skip the rest.

The notebook's configuration declared its fraud feeds as
`fraud_sources = ("phase1", "nbc")`, and `discover_fraud_data_paths` built the
prefix to list as `<testing_root>/fraud/<source>/` from exactly those strings.
The prefixes the integrated fraud tables are actually written to are
`.../fraud/Phase1-v1.0/` and `.../fraud/NBC-v1.0/` — see
`02_prepare_testing_fraud_integrated_all_features_v4.ipynb`, which sets
`source_name` to those versioned strings and uses it both as the output
directory and as the literal that fills the `fraud_source` column.

So discovery listed two prefixes that do not exist. An `except Exception` around
the listing downgraded both failures to a warning, and the function returned an
empty list.

Follow what an empty fraud list does. The training population is built by
subtracting known fraud customers from the general population. Subtract nothing,
and the training sample contains the very thing the model is being fitted to
find as an anomaly. An unsupervised detector trained on contaminated data learns
that fraud is normal. Nothing in the run's output said so; "removed 0 customers"
is an unremarkable line in a manifest.

The same mismatch broke the taxonomy from the other end. The `fraud_source`
column contains `"Phase1-v1.0"`, and lower-casing that does not produce the key
`phase1` either, so every fraud row's category resolved against a mapping that
was not there and landed in the unmapped bucket.

The fix is to *resolve* a feed name rather than compare it.
`taxonomy.resolve_fraud_source` attributes a name to the one taxonomy key it
contains, which survives the version suffix the feed will certainly bump again.
A name containing both keys, or neither, resolves to `None` rather than to a
guess. `category_for` routes its `fraud_source` argument through the resolver,
so the feed's own spelling works alongside the taxonomy key, and an unresolvable
value raises and names itself. `datasets.fraud_source_from_path` resolves the
`/fraud/<feed>/` path segment the same way, and `_source_from_path_expression`
— the Spark-side counterpart that fills the same column during the read — has an
explicit `& ~contains(other)` uniqueness guard so the two sides cannot disagree
about a directory that names both feeds.

Configuration was where this was cheapest to catch, so
`MLConfig.__post_init__` now refuses a `fraud_sources` entry naming no known
feed, with an error that says what the entry is used for and gives a value that
works. The three `conf/ml` overlays carry the feeds' real directory names.

**The general lesson, which applies well beyond this function:** an `except
Exception` that logs a warning and returns an empty collection converts a
configuration error into a plausible-looking result. If a stage can legitimately
find nothing, it needs to distinguish "found nothing" from "could not look" —
they are different facts and only one of them is safe to continue from.

### 1.2 A multi-path parquet read silently discarded the partition columns

Fixed in `2dff7c3`, where the module docstring was corrected as well as the
code.

The docstring claimed that handing several `.../time_window=.../data` prefixes
to one `spark.read.parquet` call raises `"Conflicting directory structures
detected"`. Probing Spark directly shows that is half true, and the omitted half
is the dangerous one.

Without a `basePath`, the multi-path read does not fail at all. Each supplied
path becomes its own table root, nothing beneath it is treated as a partition
directory, and the read succeeds having dropped every partition column. The
training frame comes back with no `time_window` and therefore no record of which
month any row belongs to. Every temporal analysis downstream is then quietly
computed over an undated pile of rows.

The loud failure — the one the docstring described — appears only when a
`basePath` spans two directory depths, which `training_nonfraud_candidates`
makes reachable because the two layouts it probes sit at different depths under
one root. Both docstrings now state both failure modes, with a test for each and
a third for the union that avoids them. Spark's own advice in the loud case
("please load them separately and then union them") is exactly what
`read_parquet_paths` does.

### 1.3 `customer_id` typed as a number on one side of an anti-join

Pinned by a test in `2dff7c3`; the cast itself was carried over deliberately but
had nothing proving it was load-bearing.

The fraud feeds and the non-fraud tables are written by different extracts, and
one of them types `customer_id` as a number. An equi-join between a `long` and a
`string` matches nothing. An anti-join that matches nothing removes nothing. The
result is a training set containing the fraud it was supposed to exclude, with
no error anywhere — and, again, `"removed 0 customers"` is an ordinary thing for
a manifest to say. The new test writes a numeric feed against a string training
population and asserts the row is still removed.

Note the shape this shares with §1.1: two different roads to the same wrong
outcome, both arriving there without raising.

### 1.4 `NUMERIC_DTYPE_PATTERN` matched as a substring, so complex types profiled as numeric

Fixed in `0cba7f1`.

Spark spells its complex types by naming their element type. A substring test
for `int` therefore admits `array<int>`, `map<string,bigint>` and
`struct<retries:int>`. Each of those reached `F.col(name).cast("double")` in
`profile_features`, which for an array or a struct yields all nulls rather than
raising. The column profiled as 100% null, was dropped for its null rate, and
appeared in the data-quality report as a broken feature — a feature that had
never existed in the first place, described as damaged.

The pattern is now anchored, with an optional parenthesised tail for
`decimal(18,4)` and an optional width suffix for the pandas spellings a
round-tripped schema inventory comes back in. All three call sites go through a
new `is_numeric_dtype`, because an anchored pattern is unforgiving about
coercion and whitespace in a way a substring match was not.

### 1.5 Account features were imputed with the customer's age

Fixed in `0139f38`.

`DomainImputer._tenure_source` built its scope-matched source column by
interpolating the scope key that `_scope_of` returns. `_scope_of` returns a word
— `customer`, `account` — while the feature library prefixes its columns with
abbreviations — `cust_`, `acct_`. The lookup therefore asked for
`customer_node_tenure_days`, a column present in no frame, so the scope match
never fired and every elapsed-time feature fell through to the customer-level
candidates.

An account-scoped recency feature was consequently filled with the age of the
*customer* rather than of the account — precisely the thing the module docstring
says the scope match exists to prevent. Nothing raised, because the fallback
column is present and numeric.

The scope-to-column mapping is now spelled out per scope, and
`cust_node_tenure_days` remains a general fallback so that a scopeless or
msisdn-prefixed feature still reaches a tenure column rather than dropping to
its own median.

### 1.6 `join_uri` used `str.rstrip()` with no argument

Fixed in `f9c5c5b`.

The notebook wrote `prefix.rstrip() + "/name"` in roughly twenty places.
`str.rstrip()` with no argument strips *whitespace*, not the trailing slash that
was meant. Prefixes here are built by concatenation and mostly end in `/`, so
the result was `.../run_id=x//_READY.json` — a different S3 key from the one the
reader looks at. Every resumable stage therefore rebuilt work that was already
finished, on every run, and reported success. A second test pins the scheme's
own `//` against the obvious over-correction, a blanket
`replace("//", "/")` that turns `s3://bucket` into `s3:/bucket`.

### 1.7 `s3_exists` reported absence for every kind of failure

Fixed in `f9c5c5b`.

The notebook caught every exception from `list_objects_v2` and returned `False`,
so an expired credential, a throttle, and a typo'd bucket name all reported
"this prefix does not exist". For a resumable sweep that is not a cosmetic
problem: "does not exist" means "do the work again".

Absence and failure now have different outcomes. A 403 or a 404 from the head
probe is absence — a bucket policy denying `HeadObject` on a missing key returns
403 in place of 404, so both must count. Everything else raises. A listing has
no status code meaning "empty", because an empty prefix answers with
`KeyCount: 0`, so any exception from the listing raises.

### 1.8 `write_pandas` treated "not .parquet" as "CSV"

Fixed in `f9c5c5b`. The notebook wrote comma-separated text to URIs ending in
`.pq` and to several ending in no extension at all. Whatever read those back got
a single column named after the entire header row. Both parquet extensions are
now parquet, and an unrecognised extension is refused rather than guessed at.

### 1.9 The persisted model was a bare estimator

Fixed in `f9c5c5b`.

The notebook persisted the estimator alone, losing the feature order, the
learned impute values, the fitted preprocessor and the hyperparameters. Silently
— the file loads, scoring runs, and only the numbers are wrong. Feature order
alone is enough: a scaler applied to columns in a different order than it was
fitted on produces finite, plausible, meaningless output.

`load_bundle` now refuses a bare estimator by name, refuses a format version
from the future rather than reading the subset of fields it recognises, refuses
a payload missing any of the five required keys (the notebook's two write paths
produced five keys and nine respectively, and nothing checked either), and
validates what it deserialised rather than merely constructing it.

---

## 2. Defects that misdescribed the run rather than corrupting it

These did not change any number. They changed what the run *said about itself*,
which matters more than it sounds: a manifest is what a colleague reads six
months later instead of re-running the job.

### 2.1 `GRID_DIMENSIONS['weighted_kmeans']` omitted `score_variant`

Fixed in `7407518`. `_weighted_kmeans_grid` emits `score_variant` fixed at
`cluster_z_distance` on all 3,024 full-mode arms. `grid_shape` iterates
`GRID_DIMENSIONS` to describe the search, so the manifest reported a
five-dimension grid and said nothing about which of the four variants those arms
used. A reader who knew the unweighted grid sweeps all four had every reason to
assume this one did too.

The dimension is now declared, so `grid_shape` reports it with a count of one —
which states plainly that it is pinned rather than omitting it. `GRID_DIMENSIONS`
is documented as *every key a builder emits*, not every key it varies, and
`test_grids` asserts the two agree in both directions.

### 2.2 `_download_to_temp` raised `FileNotFoundError` naming the wrong path

Fixed in `f9c5c5b`. For a missing local file, `shutil.copy2` raised
`FileNotFoundError` naming the scratch *destination* — a path the caller never
asked about and cannot act on, which reads as a bug in the loader rather than a
missing artifact. Both backends now raise `ObjectStoreError`, so a caller has
one exception type to catch for "the artifact could not be fetched".

### 2.3 `safe_plot_name` stripped separators before creating them

Fixed in `7216e57`. Two bugs in four lines. Separators were stripped before the
substitution that manufactures them, so `"roc (test)"` kept a trailing
underscore; and the empty-name guard tested the wrong thing, so an
all-punctuation name normalised to `"_"` rather than to nothing and was accepted
as a filename. The guard now requires an alphanumeric, which is what it meant.

### 2.4 `_partition_base_path` took list order for path order

Fixed in `6e0e2df`, exposed by the mutation run rather than by reading.

It took the first entry of `_PARTITION_MARKERS` *present in* the path rather
than the marker occurring *earliest in* the path. The two differ for a `dt=`
partition under a root that sits below a `time_window=` segment, and taking list
order there names a `basePath` below a partition column — which is the exact
condition `basePath` exists to prevent. It also carried a dead `rstrip("/")` on
its input: the markers include their own leading separator, so trimming the
input never changed the slice. The surviving output `rstrip` is load-bearing
only for the doubled separator a hand-assembled prefix can contain, and the test
now says so.

### 2.5 `hyperparam_id` truncation without a hash

Fixed while porting; pinned by `test_paths.py` in `0cba7f1`. Two properties hold
a resumable sweep up, and both are now tested. `hyperparam_id` must be
independent of dict insertion order, or a resumed sweep re-fits everything and
double-lists each configuration in the manifest. And a truncated id must retain
a hash of the full parameter set, or two configurations agreeing on their first
180 characters collide — the second finds the first's `_READY.json` and is
skipped without a word.

---

## 3. What is not finished

Read this section before planning any work. It is the honest state of things,
not a wish list.

### 3.1 `trust_score_05/ml/discovery.py` has no dedicated test module

659 lines. The `Check` and `DataQualityReport` pair is exercised indirectly
through `tests/ml/test_jobs.py`, which is not the same as being tested. The
schema inventory, the null-rate and cardinality profiling, the dtype-agreement
comparison and the report assembly have no direct coverage.

This is the largest single gap in the ML package and the most likely place for
the next silent failure, because discovery's whole job is to describe data — and
a description that is wrong is indistinguishable from one that is right until
somebody acts on it. §1.4 above was a discovery defect and was found by
accident.

**If you are picking up one thing from this document, pick this.** The module is
Spark-dependent only in the readers; the report logic takes frames and is
testable on pandas fixtures in the style of `tests/ml/test_selection.py`.

### 3.2 `tests/unit/`, `tests/integration/` and `tests/fixtures/` are empty

They are cookiecutter leftovers. Everything real lives in `tests/ml/`,
`tests/common/`, `tests/features/` and `tests/lineage/`. Either populate them or
delete them; leaving three empty directories named after test categories invites
somebody to put a file in one and wonder why nothing runs it.

### 3.3 The feature library's mutation sweep has not been run

Recorded in §4.2 of `feature_library_findings.md` and repeated here because it
is easy to miss. `mutate_features.py` is written — 50 mutations across the seven
feature modules — and has not been run to completion. The 298 tests in
`tests/features/` are verified to pass against the fixed code and to fail
against the two defects that were found by accident, but they are not verified
to fail against every defect they claim to guard. Mutation testing is the
standard everywhere else in this repo; this is the one place the standard has
not been met.

### 3.4 No temporal validation anywhere

Every split in both the package and the notebooks is random. Nothing in this
repo measures whether a model degrades over time, which for a fraud model is the
question that matters most, because fraud adapts and the historical window is
not the deployment window. `trust_score_05/common/splitting.py` has the
time-based helpers; nothing calls them from the sweep.

### 3.5 `pyod` and `hdbscan` are not installed in the development environment

Four of the eleven models in the zoo — `ecod`, `copod`, `hbos` and `hdbscan` —
cannot be exercised locally. `tests/ml/test_models.py` skips them. They are
covered by construction only: the registry is tested, the grids are tested, and
the fit path is shared, but no test has ever fitted one. Install the two
packages before trusting a sweep that includes them.

---

## 4. Things that look wrong and are not

Left here so the next person does not "fix" them.

### 4.1 The 0-versus-null asymmetry in the count features

Deliberate, and explained at length in §4.5 of `feature_library_findings.md`. A
count of zero and a count that could not be computed are different facts and the
feature library keeps them different. Do not coalesce nulls to zero at the
feature layer to make a profile look tidier; `DomainImputer` is the layer that
makes that decision, and it makes it with a rule per feature family.

### 4.2 `require_all_scenarios` defaults to on

A configuration that wins on one evaluation scenario and loses on another has
not won. The default forces a champion to hold up across every scenario defined
in the config, which will occasionally mean no configuration qualifies. That is
the intended answer in that situation, not a bug. Turning it off to get a
champion out of a sweep is how you ship a model that works on one month.

### 4.3 Evaluation ranks on precision at k, not AUC

The base rate is on the order of 1%, and analysts review a fixed-size queue.
AUC integrates over operating points nobody will ever use and is dominated by
the ranking of the 99% of customers who will never be looked at. Two models can
differ by 0.01 in AUC and by a factor of three in precision at the k that
matters. Rank on the metric that describes the decision being made.

### 4.4 The grids exclude `max_samples='auto'`

Not an oversight. `'auto'` is 256 rows, which for this population is a
meaningless subsample, and it silently overrides any `max_samples` reasoning a
reader would assume applies. `test_grids` asserts its absence, along with the
other absences the module exists to guarantee: no contamination in any grid, no
PCA rank at or above the feature width, no row-count rung above the population,
`hyperparam_id` injective on every grid, and `k` independent of the feature
count.
