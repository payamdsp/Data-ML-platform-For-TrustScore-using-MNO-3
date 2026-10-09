# Project Lotus

Project Lotus is a data platform and ML pipeline for computing a Trust Score from mobile network operator (MNO) data. It provides:

- ETL and preprocessing code to move raw/bronze data to curated silver tables (entity transforms, data quality checks, quarantine handling).
- Gold-stage model training and scoring code to produce Trust Score outputs and explainability artifacts.
- CI/infra separation and artifacts that run in the data-sandbox account.

## Table of contents

- What this is
- Stack
- Repository layout (top-level)
- How it fits together (runtime/dataflow)
- How to run (shortest path)
- Detailed module runbooks and run commands
  - Preprocessing
  - Silver pipeline
  - Transforms
  - Data-quality and quarantine
  - Publishing
  - Gold pipeline and Trust Score
- Model lifecycle: feature engineering → selection → training → evaluation → deployment
- Detailed operational chapters
  - Preparing a local dev environment
  - Running the full Silver pipeline on a sample partition
  - Training / model experimentation
  - Productionize a model
- Observability, metrics, and lineage
- Security & infra notes
- Testing & CI
- Troubleshooting
- Housekeeping and conventions
- Where to look next
- Try asking
- Contact / escalation
- Appendix: Precise file list

---

What this is
------------
Project Lotus is a data platform and ML pipeline for computing a Trust Score from mobile network operator (MNO) data. It provides a structured set of scripts, transforms, and orchestration to move raw inputs through Bronze → Silver → Gold stages, producing Trust Score outputs with explainability and lineage.

### Stack
- **Language(s):** Python (primary)
- **Framework / runtime:** Python packages with scripts and Docker for containerized preprocessing jobs; integration points for SageMaker and EMR
- **Notable files that shape runtime:** pyproject.toml, preprocessing/requirements.txt, build_and_push.sh, runner.py (silver pipeline), trust_score_05 (gold)

## Repository layout (top-level)
Annotated canonical tree (paths are relative to repository root):

```
Project_Lotus/
  .github/                        GitHub actions/workflows (project-level CI in this directory)
  accounts/                       Terraform for data-sandbox (account infrastructure)
  gitignore.txt                   repo ignore for Project_Lotus
  README.md                       (this file)
  project_lotus/                  Python project - package sources and orchestration
    Makefile.txt
    README.md                     package-local README (detailed inside package)
    pyproject.toml                packaging and dependency pins
    gitignore.txt
    docs/                         documentation and runbooks
    notebooks/                    notebooks used for analysis/model development
    sagemaker/                    SageMaker wrappers and scripts
    lambdas/                      lambda code (if used in orchestration)
    reports/                      report templates and outputs
    references/                   reference artifacts, data dictionaries
    tests/                        test suite
    project_lotus/                Python package; key subpackages below
      __init__.py
      preprocessing/              Bronze->Silver ETL code
        Dockerfile.txt
        build_and_push.sh
        preprocess.py
        run_processing_job.py
        requirements.txt
      silver_pipeline/            Silver-stage orchestration and transforms
        README.md
        __init__.py
        artifacts.py
        common.py
        config.py
        dq.py
        publish.py
        quarantine.py
        references.py
        runner.py
        transforms/               per-entity transforms and feature extraction
          __init__.py
          account.py
          audit.py
          device.py
        configs/
        yaml/
      gold_pipeline/              Gold-stage model & scoring code
        yaml/
        licenses/
        trust_score_05/           Trust score implementation (models / lineage)
          __init__.py
          lineage/
```

## How it fits together (runtime/dataflow)
High-level flow:

1. Bronze (raw data, external ingestion) — NOT stored in this repo; assumed to be placed into source S3 or equivalent by upstream ingestion.
2. Preprocessing (project_lotus/preprocessing): prepares raw inputs, harmonizes schemas, converts to parquet or partitioned formats used by silver pipeline.
3. Silver pipeline (project_lotus/silver_pipeline): orchestrates entity transforms (transforms/), runs data-quality (dq.py), quarantines bad records (quarantine.py), and publishes curated silver artifacts (publish.py).
4. Gold pipeline (project_lotus/gold_pipeline/trust_score_05): consumes silver artifacts, trains or loads models, performs scoring, and emits trust score + explainability artifacts. Notebooks and sagemaker/ contain experiment and training glue.
5. Downstream reporting and lineage: reports/, references/, and trust_score_05/lineage contain outputs used for audit and explainability.

## How to run (shortest path)
Prerequisites:
- Python 3.9+ (as pinned in pyproject.toml)
- Access to data-sandbox AWS account and appropriate credentials for infra operations
- For local development: virtualenv or Poetry

Quick start (local dev):

1. Create virtualenv and install deps:

```
cd Project_Lotus/project_lotus
python -m venv .venv && source .venv/bin/activate
pip install -r project_lotus/preprocessing/requirements.txt
pip install -e .
```

2. Run a preprocessing job on a small sample:

```
python -m project_lotus.preprocessing.preprocess --input <sample.csv> --output ./out
# or
python -m project_lotus.preprocessing.run_processing_job --config <yaml> --run-id local-test
```

3. Dry-run the silver pipeline runner:

```
python -m project_lotus.silver_pipeline.runner --config project_lotus/silver_pipeline/yaml/sample.yaml --dry-run
```

4. Run full silver pipeline (example):

```
python -m project_lotus.silver_pipeline.runner --config project_lotus/silver_pipeline/yaml/sample.yaml --run-id 20261009-local
```

5. Train or score model (notebooks or sagemaker wrappers):
- Use notebooks/ for interactive experiments.
- Use sagemaker/ scripts for managed training and tuning.

## Detailed module runbooks and run commands

1) Preprocessing (project_lotus/project_lotus/preprocessing/)

Purpose:
- Prepare raw/bronze data for the silver pipeline: schema harmonization, lightweight validation, splitting, partitioning.

Files:
- Dockerfile.txt: base container instructions (rename to Dockerfile for building).
- build_and_push.sh: helper to build and push the Docker image to a registry.
- preprocess.py: main preprocessing logic.
- run_processing_job.py: wrapper to run processing jobs with configs and run-id.
- requirements.txt: runtime dependencies for preprocessing container.

How it runs:
- Locally: run preprocess.py or run_processing_job.py against a sample input and output folder.
- Container: build_and_push.sh builds an image (Dockerfile.txt) and pushes to a registry; the image is used on EMR, ECS, or in CI to run large-scale jobs.

Typical CLI:
```
# Local
python project_lotus/preprocessing/preprocess.py --input <input-path> --output <output-path> --log-level INFO
python project_lotus/preprocessing/run_processing_job.py --config <yaml> --run-id <id>

# Container
cd project_lotus/project_lotus/preprocessing
./build_and_push.sh <registry> <tag>
```

Inputs & outputs:
- Inputs: CSV/Parquet raw extracts or S3 locations (configured via YAML/args).
- Outputs: Partitioned Parquet / Glue table seeds for silver pipeline.

Logging & errors:
- Logs to stdout; exceptions indicate malformed inputs. For EMR jobs, check EMR logs and S3 staging location.

Troubleshooting checklist:
- Ensure deps installed (requirements.txt).
- Check config values (paths, partitions).
- If EMR job fails, fetch stderr/stdout from cluster for stack trace; reproduce locally with minimal sample.

2) Silver pipeline (project_lotus/project_lotus/silver_pipeline/)

Purpose:
- Orchestrates transforms that turn preprocessed data into curated silver entity tables, performs data-quality checks, quarantines bad records, and publishes silver artifacts for the gold pipeline.

Key modules and roles:
- runner.py: main orchestration. Reads YAML config(s), sequences transforms, runs DQ checks, calls quarantine or publish workflows.
- transforms/*.py (account.py, audit.py, device.py): entity-specific transformations.
- dq.py: data-quality rules and assertions.
- quarantine.py: isolates records failing checks into quarantine sink.
- publish.py: writes curated silver artifacts to configured storage.
- artifacts.py: central naming/versioning utilities.

How to run:
- Dry-run:
```
python -m project_lotus.silver_pipeline.runner --config project_lotus/silver_pipeline/yaml/<config>.yaml --dry-run
```
- Full run:
```
python -m project_lotus.silver_pipeline.runner --config project_lotus/silver_pipeline/yaml/<config>.yaml --run-id <yyyymmdd-HHMM>
```

Inputs & outputs:
- Inputs: preprocessed parquet and lookups via config.
- Outputs: curated silver tables (S3/Glue), DQ metrics, quarantine buckets.

Observability & artifacts:
- Runner emits DQ metrics and run metadata to configured sinks. artifacts.py controls naming.

Troubleshooting & common errors:
- "couldn't find resource": security/ root not applied; apply security resources first.
- Authorization error: wrong resource location (infra vs security).
- Schema mismatch: inspect transform unit tests and sample input.

3) Transforms (project_lotus/silver_pipeline/transforms/*.py)

Purpose:
- Per-entity feature extraction and normalization. Each transform returns normalized entity rows and feature columns.

Files:
- account.py, audit.py, device.py

Runbook:
- Unit-testable functions: import and run with sample DataFrames.
- Use pytest to validate behavior.

Implementation notes:
- Derived feature logic (aggregations, time windows) is implemented here and should be reflected in model feature metadata.

4) Data-quality and quarantine (dq.py, quarantine.py)

Purpose:
- dq.py contains rules and thresholds (null thresholds, cardinality checks, distribution checks).
- quarantine.py stores failing records for manual inspection.

Runbook:
- Runner executes dq.py after transforms and decides publish vs quarantine.
- To reproduce quarantine: run runner on a partition and inspect S3 quarantine path configured in YAML.

5) Publishing (publish.py, artifacts.py)

Purpose:
- Write curated silver artifacts (file naming and versioning), seed gold pipeline inputs.

Runbook:
- Confirm artifact naming follows var.project prefix (lotus).
- Run publish only after DQ passes.

6) Gold pipeline and Trust Score (project_lotus/project_lotus/gold_pipeline/trust_score_05/)

Purpose:
- trust_score_05 contains gold-stage code for Trust Score computation and lineage tracking. It consumes silver artifacts and produces scored outputs and explainability data.

Files and locations for model code:
- project_lotus/gold_pipeline/trust_score_05/
- project_lotus/sagemaker/
- project_lotus/notebooks/

Runbook (developer workflow):
1. Recreate silver dataset locally or via a subset from S3.
2. Re-run experiment notebooks to produce baseline metrics.
3. Use sagemaker/ scripts to launch training/tuning, or run local training scripts.
4. Evaluate and record model artifacts (feature set, hyperparameters, metrics) in artifacts/lineage.
5. Promote model to production by publishing model artifact and updating configuration.

## Model lifecycle: feature engineering → selection → training → evaluation → deployment
1. Feature extraction / engineering (transforms)
2. Feature preprocessing (preprocessing + transforms)
3. Feature selection / subsetting (notebooks / training scripts record selected sets)
4. Model selection (notebooks, sagemaker experiments)
5. Hyper-parameter tuning and calibration (use sagemaker tuning or local grid/search; apply Platt or isotonic calibration as post-processing)
6. Retraining / online updates (scheduled or drift-triggered; write artifacts with lineage metadata)
7. Evaluation & scoring logic (notebooks and trust_score code produce evaluation metrics and scoring outputs)
8. Explainability & reason codes (SHAP/LIME or rule-based contributions persisted alongside scores)

## Detailed operational chapters

Preparing a local dev environment
- Create venv, install requirements, run unit tests.
- Use small input samples to exercise preprocess.py and silver runner.

Running the full Silver pipeline on a sample partition
1. Prepare sample input in S3 or local folder.
2. Edit YAML config from project_lotus/silver_pipeline/yaml/ to point at sample input and local output.
3. Dry-run: python -m project_lotus.silver_pipeline.runner --config project_lotus/silver_pipeline/yaml/sample.yaml --dry-run
4. Run: python -m project_lotus.silver_pipeline.runner --config project_lotus/silver_pipeline/yaml/sample.yaml --run-id local-test
5. Inspect outputs in configured output path and quarantine if any.

Training / model experimentation
- Use notebooks/ to explore feature sets and candidate models.
- Persist candidate models with: training data version, feature set, hyperparameters, evaluation metrics.

Productionize a model
1. Finalize model artifact and push to artifact store (S3) with metadata.
2. Update gold pipeline config to point to new model artifact.
3. Run scoring job in staging and validate explainability artifacts.
4. Promote to production by updating pipeline config and publishing lineage.

## Observability, metrics, and lineage
- Runner and publish stages emit DQ metrics and run metadata to configured sinks (S3/metrics). Check artifacts.py and runner for keys/prefixes.
- Lineage for gold models is kept in trust_score_05/lineage. Ensure model training writes JSON/YAML metadata with commit, run id, and metrics.

## Security & infra notes
- Project runs only in data-sandbox (162591926854).
- Confidential/Restricted data are banned — synthetic data only.
- Infrastructure split: accounts/data-sandbox/infra defines data resources; accounts/data-sandbox/security defines IAM/KMS, Secrets Manager and all IAM roles/policies.
- Prefix resource names with var.project (lotus) and use var.environment for environment-specific names.

## Testing & CI
- Unit tests under project_lotus/tests. Run via pytest from project_lotus directory.
- CI workflows are in Project_Lotus/.github. The infra repo's CI enforces fmt and terraform validation.

## Troubleshooting (common failures)
- Missing resource error: apply security/ root first; resource may be defined in security.
- Authorization error on role: resource mislocated; split infra/security per repo conventions.
- DQ failures: run runner in dry-run and inspect dq.py rules; quarantined rows are persisted in S3.
- EMR job failures: check cluster logs and ensure the container image used (if any) is accessible.

## Housekeeping and conventions
- Prefix resources with var.project (lotus).
- Build names from var.environment.
- No hardcoded account ids or ARNs — resolve by name or read /contract/_account/* from SSM.
- required_version = "~> 1.15.0" for Terraform; commit .terraform.lock.hcl.
- Run terraform fmt -recursive before committing.
- Never commit TF state, .terraform/, saved plans, or .tfvars with real values.

## Where to look next (quick pointers)
- Preprocessing code: Project_Lotus/project_lotus/project_lotus/preprocessing/
- Silver orchestration and transforms: Project_Lotus/project_lotus/project_lotus/silver_pipeline/
- Gold/trust score code and lineage: Project_Lotus/project_lotus/project_lotus/gold_pipeline/trust_score_05/
- Notebooks: Project_Lotus/project_lotus/notebooks/
- Packaging: Project_Lotus/project_lotus/pyproject.toml
- Docker build: Project_Lotus/project_lotus/project_lotus/preprocessing/build_and_push.sh and Dockerfile.txt

## Try asking
- "Which transforms produce the device-level features used by the model?"
- "Where does artifacts.py build artifact names and how do I register a new model there?"
- "How do I run the runner.py for a specific date partition, and where are its DQ metric outputs written?"

## Contact / escalation
If production infra or account permission issues arise, raise a Linear ticket on the Project Lotus team and tag InfraSec:
https://linear.app/enstream-workspace/team/LOTUS/overview

## Appendix: Precise file list (top-level entries and package areas inspected)
- Project_Lotus/
  - .github/
  - accounts/
  - gitignore.txt
  - README.md (this file)
  - project_lotus/
    - Makefile.txt
    - README.md
    - pyproject.toml
    - gitignore.txt
    - docs/
    - notebooks/
    - sagemaker/
    - lambdas/
    - reports/
    - references/
    - tests/
    - project_lotus/
      - __init__.py
      - preprocessing/
        - Dockerfile.txt
        - build_and_push.sh
        - preprocess.py
        - run_processing_job.py
        - requirements.txt
      - silver_pipeline/
        - README.md
        - __init__.py
        - artifacts.py
        - common.py
        - config.py
        - dq.py
        - publish.py
        - quarantine.py
        - references.py
        - runner.py
        - transforms/
          - __init__.py
          - account.py
          - audit.py
          - device.py
        - configs/
        - yaml/
      - gold_pipeline/
        - licenses/
        - yaml/
        - trust_score_05/
          - __init__.py
          - lineage/


---

Notes and limitations
---------------------
- This README is grounded in the repository structure and files found under Project_Lotus. For file-level, line-by-line code explanation (exact feature formulas, model hyperparameters, or internal function signatures), I can expand the README into per-module documentation by extracting and documenting each transform and model file. If you want that, tell me which file(s) to expand first (for example, project_lotus/silver_pipeline/transforms/account.py or the trust_score_05 model files) and I will create detailed runbooks and docblocks for them.
