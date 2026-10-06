# Trust Score 0.5

Trust Score 0.5 scores telecom customers for fraud risk. It has two halves that
meet at one artifact: a lineage half that builds a Gold-layer table of every
MSISDN's history from raw carrier feeds, and an ML half that turns that history
into features and fits a zoo of unsupervised anomaly detectors against it.

Fraud here is rare — on the order of 1% — and the labels are confirmed cases
only, never confirmed negatives. That is why the models are unsupervised: a
candidate model can be *measured* against known fraud, but a classifier cannot
be *trained* on a population whose negatives are really "nobody has looked yet".

## Start here

| If you are | Read |
|---|---|
| New to the project | [Handover](handover.md), then run `notebooks/0.0-standalone/` in order |
| Setting up a machine | [Getting started](getting-started.md) |
| About to change the code | The two findings documents, listed below |
| Looking for what to work on next | [Handover](handover.md) §5 |

**[Handover](handover.md)** is the main document. It covers what the repo is,
how to run each stage, a map of every module, the state of the work, the four
things worth doing next, and the practical traps that otherwise cost an
afternoon each. It was written to be read top to bottom once.

## The findings documents

Two documents record the defects found in the previous notebook implementation
of this pipeline. All of them are fixed in the package. They are written down
because several will be reintroduced by anyone porting the same logic from the
same notebooks, and because every one of them failed *silently* — the pipeline
ran, produced numbers, and the numbers were wrong.

- `trust_score_05/features/docs/feature_library_findings.md` — the Spark feature
  library: correctness defects, structural issues, gaps, and the defects found
  in this package while testing it.
- `trust_score_05/ml/docs/ml_pipeline_findings.md` — everything downstream of
  the feature matrix: discovery, selection, preprocessing, models, grids, sweep,
  evaluation, drift. Also lists what is unfinished, and what looks like a defect
  but is deliberate.

Read the "deliberate" sections. They exist so nobody helpfully undoes a decision.

## The notebooks

There are two tracks and they are not alternatives. `notebooks/README.md`
explains which to open first.

`notebooks/0.0-standalone/` is five notebooks that implement the whole pipeline
in plain numpy, pandas and scikit-learn, importing nothing from this package. No
cluster, no credentials, about two minutes end to end. This is the documentation
that runs, and it is where a newcomer should begin.

`notebooks/1.0-exploration/` through `notebooks/3.0-modeling/` are eight
notebooks that drive the package against real data. Every cell that writes
anything is gated behind a flag set to `False`.

## Running the pipeline

Three jobs, in this order, because the order is a real dependency:

```bash
python3 -m trust_score_05.ml.jobs.discovery_job --config conf/ml/dev.yaml
python3 -m trust_score_05.ml.jobs.selection_job --config conf/ml/dev.yaml
python3 -m trust_score_05.ml.jobs.sweep_job     --config conf/ml/dev.yaml
```

Discovery inventories the schemas and evaluates the data-quality gates, writing
`_READY.json` only when nothing is at `error` severity. Selection profiles the
candidate features once and writes each model's ranked list. The sweep is the
expensive, fanned-out, resumable one. See [Handover](handover.md) §2 for the
flags and how config overlays compose.
