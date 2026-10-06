# Handover

Written for whoever picks this up while Moein is away. It assumes you know
Python and roughly what an anomaly detector is, and assumes nothing else. Read
it top to bottom once; it is ordered so that each section only depends on the
ones above it.

---

## 1. What this repo is

Trust Score 0.5 scores telecom customers for fraud risk. It has two halves that
meet at one artifact:

**The lineage half** (`trust_score_05/lineage/`) turns raw carrier feeds into a
Gold-layer table describing every MSISDN's history: which account and customer
it belonged to over which interval, which device, and every lifecycle event
along the way. This half was ported first and is the larger of the two.

**The ML half** (`trust_score_05/features/` and `trust_score_05/ml/`) turns that
history into a feature matrix, selects features, fits a zoo of unsupervised
anomaly detectors, and ranks them. Fraud is rare — on the order of 1% — and the
labels we have are confirmed cases only, never confirmed negatives. That is why
the models are unsupervised: we can measure a candidate model against known
fraud, but we cannot train a classifier on a population where the negatives are
"nobody has looked yet".

They meet at the Gold lineage table. The feature library reads it; nothing in
the ML half knows how it was built.

---

## 2. The five-minute version of how to run something

Everything below assumes you are at the repo root.

```bash
# tests, no cluster needed
PYTHONPATH=. python3 -m pytest tests/ml tests/common -q

# tests that need a local Spark session
PYTHONPATH=. python3 -m pytest tests/features tests/lineage -q

# lint (note the explicit paths -- see §6)
python3 -m ruff check --select E4,E7,E9,F,I,E501 \
    tests/ml/ tests/common/ tests/features/ \
    trust_score_05/ml/ trust_score_05/common/ trust_score_05/features/
```

The three ML jobs, in the order they must run:

```bash
python3 -m trust_score_05.ml.jobs.discovery_job --config conf/ml/dev.yaml
python3 -m trust_score_05.ml.jobs.selection_job --config conf/ml/dev.yaml
python3 -m trust_score_05.ml.jobs.sweep_job     --config conf/ml/dev.yaml
```

`conf/ml/base.yaml` is always applied underneath and does not need to be passed.
`--config` is repeatable and later files overlay earlier ones. `--set
key.path=value` overrides anything after the files. Every job takes `--run-id`,
`--log-level` and `--keep-spark`.

The order is a real dependency and nothing enforces it beyond markers on the
object store. Discovery writes `_READY.json` only when no data-quality check is
at `error` severity, and the two stages after it read that marker. The sweep
refuses to start if the selection run it was pointed at is absent.

**If you want to understand the pipeline rather than run it, do not start here.**
Start with §3.

---

## 3. Where to start reading, depending on what you need

### You are new and need to understand the ML pipeline

Open `notebooks/0.0-standalone/` and run the five notebooks in order. They
implement the entire pipeline — synthetic data, exploration, selection,
preprocessing, training, evaluation, reporting — in plain numpy, pandas and
scikit-learn, importing nothing from this package. No cluster, no credentials,
about two minutes end to end.

They exist because the package is correct and also opaque. A working sweep is
one call to `run_sweep(...)`, and reading that call teaches you nothing about
what a sweep is. The standalone notebooks spell every step out longhand so you
can break things and watch what breaks.

The synthetic data is built to teach. Fraud is planted as a *conjunction* —
recent device change AND recent SIM change AND an identity-lookup burst AND
short tenure — so no single threshold separates the classes. Five deliberate
traps are planted too (noise columns, a 95%-null column, a zero-variance
column, a numeric identifier, a label leak) and notebook `02` asserts that all
five were caught. If you change the selection logic and that assertion fires,
you broke something real.

`notebooks/README.md` explains both tracks and their order. Read it first.

### You have cluster access and need real numbers

Use the package track: `notebooks/1.0-exploration/` through
`notebooks/3.0-modeling/`. Every cell that writes anything is gated behind a
flag set to `False` at the top of that cell, so nothing overwrites a table
because you held down shift-enter.

### You are about to change the code

Read the two findings documents before you touch anything:

- `trust_score_05/features/docs/feature_library_findings.md` — the Spark feature
  library.
- `trust_score_05/ml/docs/ml_pipeline_findings.md` — everything downstream of
  it.

They are not changelogs. They record defects that were found in the previous
notebook implementation, all now fixed, all of which failed *silently* — the
pipeline ran, produced numbers, and the numbers were wrong. Several of them will
be reintroduced by anyone porting the same logic from the same notebooks. Each
document also has a section listing what is deliberately the way it is, so you
do not "fix" it back.

### You want to know what is left to do

§5 of this document, and §3 of `ml_pipeline_findings.md`.

---

## 4. The map of the code

```
trust_score_05/
├── lineage/                 Gold-layer construction (ported first, largest)
│   ├── transforms/          the per-entity walks: accounts, customers,
│   │                        lifecycle, chains, boundaries, canonical events
│   ├── dq/                  the data-quality rule framework and its rule sets
│   ├── io/                  readers, the merge sink, run-control markers
│   ├── jobs/                spark-submit entry points
│   ├── schemas.py           every table's schema, in one place
│   └── slice.py, staging.py, spark.py, timezone.py, progress.py
│
├── features/                Spark feature engineering (five stages)
│   ├── config.py            FeatureConfig -- every window, threshold and rule
│   ├── snapshots.py         entity snapshots at a reference timestamp
│   ├── events.py            event counts and recencies per lookback window
│   ├── calculators.py       intervals, ratios, entropies, decay features
│   ├── enstream.py          identity-lookup (EnStream) features
│   ├── expressions.py       shared column expressions and naming
│   ├── assemble.py          runs the stages and joins them into one matrix
│   └── docs/feature_library_findings.md
│
├── ml/                      everything downstream of the feature matrix
│   ├── config.py            MLConfig, and the YAML -> dataclass loader
│   ├── paths.py             every prefix and every run/hyperparam id
│   ├── datasets.py          path discovery, the readers, population loaders
│   ├── discovery.py         schema inventory and the data-quality gates
│   ├── taxonomy.py          fraud feeds, categories, scenario definitions
│   ├── selection.py         candidate features, profiling, ranking, top-n
│   ├── preprocessing.py     DomainImputer and the scaler recipes
│   ├── models.py            the eleven-model zoo and the fit/score interface
│   ├── grids.py             hyperparameter grids and GRID_DIMENSIONS
│   ├── sweep.py             arm enumeration, resumable execution, manifests
│   ├── evaluation.py        top-k metrics, scenarios, champion selection
│   ├── drift.py             PSI and score-distribution monitoring
│   ├── jobs/                the three spark-submit entry points
│   └── docs/ml_pipeline_findings.md
│
└── common/                  shared by both halves
    ├── io/s3.py             object store; accepts filesystem paths too
    ├── io/serialization.py  the model bundle format
    ├── metrics/             classification and ranking metrics
    ├── splitting.py         train/test splitting, including time-based
    └── viz/plots.py         plotting helpers

conf/lineage/, conf/ml/       YAML overlays: base + dev/sandbox/local/s3tables
tests/                        mirrors the package; ~20k lines
notebooks/                    two tracks; see notebooks/README.md
```

### Four things about this codebase that are not obvious

**Business rules live in config, never as literals.** Windows, thresholds, feed
names, scenario definitions — all in `FeatureConfig` or `MLConfig` and the YAML
overlays. If you find yourself typing `30` into a calculator, look for the
config key that already exists.

**`common/io/s3.py` accepts a filesystem path wherever it accepts a URI.** This
is why most of the test suite needs no object store: `tmp_path` *is* a real
object store, and the artifact round-trips are tested against the same code that
runs on a cluster.

**Docstrings here carry the reasoning, not the signature.** Every non-obvious
claim cites a file or symbol and says *why*, usually naming the specific defect
in the previous notebook implementation that the code exists to prevent. They
are long on purpose. If you change behaviour, change the docstring in the same
commit — a stale explanation is worse than none, because it is believed.

**Mutation testing is the standard for regression tests.** A test that claims to
pin a defect is not trusted until the defect has been reintroduced in the source
and the named test confirmed to fail. The drivers used for this live outside the
repo and are disposable; what matters is that you do it. Commit messages state
the mutation count, e.g. "27/27 mutations caught".

---

## 5. State of the work, and what to do next

### Done

The lineage half is ported with its known defects fixed and has ~19k lines of
tests. The feature library is ported, with 298 tests over seven modules against
a real local Spark session. The ML half is complete — config, paths, datasets,
discovery, taxonomy, selection, preprocessing, the eleven-model zoo, the grids,
the resumable sweep, evaluation, drift, and three job entry points — with tests
for everything except `discovery.py`. Both notebook tracks exist. The YAML
overlays for dev, sandbox and local are written.

### The four things worth doing next, in priority order

**1. Test `trust_score_05/ml/discovery.py`.** 659 lines, no dedicated test
module. This is the largest gap and the most likely place for the next silent
failure, because discovery's entire job is to *describe* data, and a wrong
description is indistinguishable from a right one until somebody acts on it.
One of the defects in §1.4 of `ml_pipeline_findings.md` was a discovery defect
and it was found by accident. The report logic takes frames and is testable on
pandas fixtures; copy the style of `tests/ml/test_selection.py`.

**2. Run the feature library's mutation sweep.** 50 mutations across the seven
feature modules, written and dry-run-verified but not executed. Until it runs,
the 298 feature tests are known to pass against the fixed code but not known to
fail against every defect they claim to guard. See §3.3 of
`ml_pipeline_findings.md`. Expect roughly 30 seconds per Spark module.

**3. Add temporal validation.** Every split in the repo is random. For a fraud
model the question that matters most is whether it degrades over time, because
fraud adapts and the historical window is not the deployment window.
`common/splitting.py` has the time-based helpers; nothing calls them from the
sweep.

**4. Install `pyod` and `hdbscan`.** Four of the eleven models — `ecod`,
`copod`, `hbos`, `hdbscan` — have never been fitted by any test in any
environment, because the packages are absent. They are covered by construction
only. Do not trust a sweep that includes them until this is fixed.

### Known gaps that are smaller but should not be forgotten

`tests/unit/`, `tests/integration/` and `tests/fixtures/` are empty cookiecutter
leftovers — populate or delete them, but do not leave three directories named
after test categories that contain nothing. And the sweep has never been run at
full scale on real data; the resumability and marker logic are tested, but
tested is not the same as exercised across a hundred submissions.

---

## 6. Practical notes that will otherwise cost you an afternoon

**Lint with the explicit path list.** Running ruff against `trust_score_05/` as
a whole reports 14 pre-existing violations in the committed `lineage/` package.
Use the path list in §2. `ruff format` is never run on this repo. Settings live
in `pyproject.toml`: line length 99, `extend-select = ["I"]`,
`known-first-party = ["trust_score_05"]`, `force-sort-within-sections = true`.

**`python3 -m pytest`, not `pytest`.** The console script is not always on PATH
in the environments this has been developed in.

**Spark tests need `SPARK_LOCAL_IP` set** in most sandboxes, and are slow enough
that `tests/features/test_assemble.py` may need running in `-k` chunks rather
than as a whole module.

**`snapshot_scope["base"]` is not the same grain as `snapshot_scope[scope]`.**
This one has cost two people a day each, in this codebase, twice.
`build_scope_snapshots` returns a dict. The `"base"` entry is one row per
lineage interval and carries every entity column. Every other entry is
aggregated to one row per entity, and only the grouping keys survive. A frame
with the right *columns* is not a frame at the right *grain*, and Spark will not
tell you the difference — you get a join that silently multiplies rows. Written
up at length in §4.4 of `feature_library_findings.md`.

**`LABEL_LEAKAGE_COLUMNS` is not optional.** It is the only thing between you
and a model that has learned to read the fraud case file. Removing it makes the
metrics go up. That is the tell, not the reward.

**Rank models on precision at k, not AUC.** The base rate is around 1% and
analysts review a fixed-size queue. AUC integrates over operating points nobody
will use and is dominated by the ranking of the 99% who will never be looked at.
Two models can differ by 0.01 in AUC and by 3x in precision at the k that
matters.

**Fix your operating point before you look at the numbers.** Choosing k after
seeing the leaderboard is how you convince yourself of a result that is not
there. Notebook `04` in the standalone track does it in the right order on
purpose.

---

## 7. Working conventions

Branch: `feature/DAT-trust-score-05-rebuild`. Commits are incremental and are
**not pushed** — that was the standing instruction and it should stay that way
until someone with the context decides otherwise.

Commit messages are long prose. A message names every defect the commit pins,
says what the wrong behaviour actually produced, and states the mutation count
where regression tests are involved. This is deliberate: the commit log is the
only place that records *why* a line of defensive code exists, and the git
history is the one document that cannot drift out of sync with the code.

The word "Tulip" does not appear in ported code. The reference repo it came from
used that name; this one does not.

---

## 8. If you only remember one thing

Nearly every defect recorded in the two findings documents shares a shape: the
code caught an exception, or coerced a type, or fell back to a default, and
carried on producing plausible output. Not one of them crashed. The training set
that contained the fraud it was meant to exclude, the account feature imputed
with the customer's age, the resumable stage that redid its work every run, the
model bundle that lost its feature order — all of these ran clean and reported
success.

So when you add a fallback, ask what it looks like from the outside when it
fires. If the answer is "the same as success", the fallback is a bug.
