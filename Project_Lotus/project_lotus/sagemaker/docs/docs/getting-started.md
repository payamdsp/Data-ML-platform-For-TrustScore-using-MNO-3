# Getting started

From a clean machine to a green test suite, then to something running. Every
command assumes the repo root as the working directory.

## 1. Python and the package

Python 3.10 or later. Install the package in editable mode with its development
extras:

```bash
python3 -m pip install -e ".[dev]"
```

If that fails or the extras are not defined in your checkout, the direct route
is enough to run everything except the four optional models:

```bash
python3 -m pip install pandas numpy scikit-learn joblib pyyaml matplotlib \
                       fastparquet pytest ruff
```

## 2. Spark

The lineage half and the feature library are PySpark. The ML package is
deliberately Spark-free except for `ml/datasets.py` and `ml/discovery.py`, which
is why most of the test suite needs no cluster at all.

```bash
python3 -m pip install "pyspark==3.5.1"
```

Spark needs a JVM — Java 11 or 17. Check with `java -version`. On many sandboxes
you also need to pin the driver's address or the session will hang looking for
one:

```bash
export SPARK_LOCAL_IP=127.0.0.1
```

## 3. The four optional models

`ecod`, `copod` and `hbos` come from `pyod`; `hdbscan` is its own package.
Neither is installed in the environment this was developed in, so those four
models have never actually been fitted by a test — the registry, the grids and
the shared fit path are covered, but the estimators are not.

```bash
python3 -m pip install pyod hdbscan
```

Install them before trusting a sweep that includes those models.
`tests/ml/test_models.py` will stop skipping once they are present.

## 4. Run the tests

Start with the suites that need no cluster. This is the fast loop you will use
while developing, and it covers the whole ML package and `common/`:

```bash
PYTHONPATH=. python3 -m pytest tests/ml tests/common -q
```

Then the Spark suites. These are slower — `tests/features/test_assemble.py` in
particular may need running in `-k` chunks rather than as a whole module if your
environment has a per-command time limit:

```bash
SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=. python3 -m pytest tests/features -q
SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=. python3 -m pytest tests/lineage  -q
```

Use `python3 -m pytest`, not the bare `pytest` console script — it is not
reliably on PATH.

## 5. Lint

```bash
python3 -m ruff check --select E4,E7,E9,F,I,E501 \
    tests/ml/ tests/common/ tests/features/ \
    trust_score_05/ml/ trust_score_05/common/ trust_score_05/features/
```

Pass the explicit path list. Running ruff against `trust_score_05/` as a whole
reports 14 pre-existing violations in the committed `lineage/` package, which
will bury anything you actually introduced. `ruff format` is not run on this
repo; the configuration in `pyproject.toml` sets line length 99,
`extend-select = ["I"]`, `known-first-party = ["trust_score_05"]` and
`force-sort-within-sections = true`.

## 6. Run something, with no data and no credentials

This is the recommended first thing to do on a new machine, because it proves
the install and teaches the pipeline at the same time:

```bash
export TS05_RUN_DIR=./_run
jupyter lab notebooks/0.0-standalone/
```

Run `00` through `04` in order. They generate their own synthetic data, import
nothing from `trust_score_05`, and take about two minutes end to end. Each one
writes to `TS05_RUN_DIR` and asserts that its predecessor finished, so you
cannot accidentally run them out of order.

Notebook `02` ends with an assertion that all five deliberate traps in the
synthetic data were caught by the selection logic. If that assertion passes,
your environment is working.

## 7. Configure access to real data

The three ML jobs read from an object store. Configuration composes in layers:
`conf/ml/base.yaml` is always applied underneath, then any `--config` files in
order, then `--set key.path=value` overrides.

```bash
python3 -m trust_score_05.ml.jobs.discovery_job --config conf/ml/dev.yaml
```

Run discovery first, always. It inventories the schema of every configured
prefix, evaluates the data-quality gates, and writes `_READY.json` only when no
check is at `error` severity — and the two stages after it read that marker.
Discovery is metadata-only and finishes in minutes, so there is no reason to
skip it.

The available overlays are `dev.yaml`, `sandbox.yaml` and `local.yaml`. If
discovery reports that a prefix does not exist, check the overlay's roots before
assuming the data is missing; a fraud-feed name mismatch of exactly this kind
once emptied the fraud population silently and trained a model on the thing it
was meant to detect. That story is §1.1 of
`trust_score_05/ml/docs/ml_pipeline_findings.md` and it is worth the five
minutes.

## 8. Then read the handover

[Handover](handover.md) is the document that explains what everything is, what
state the work is in, and what to do next. Read it before making changes.
