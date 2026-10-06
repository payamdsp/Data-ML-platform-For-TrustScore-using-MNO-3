"""The helper Lambda the state machines call (terraform/lambda/pipeline_helper.py).

    local\\.venv\\Scripts\\python.exe -m pytest sagemaker/tests/test_pipeline_helper.py -q

Every job request the pipeline builds is fed to the parser that will read it
inside the image, so a Lambda and an entry point that disagree fail here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import boto3
from moto import mock_aws
import pytest

SAGEMAKER_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SAGEMAKER_DIR), str(SAGEMAKER_DIR / "terraform" / "lambda")]

import inference  # noqa: E402
import pipeline_helper as helper  # noqa: E402
import sm_common  # noqa: E402
import train  # noqa: E402

PARAM = "/lotus-sandbox-ts05/last-training-start"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for key, value in {
        "JOB_NAME_PREFIX": "ts05",
        "CONFIG_FILES": "conf/ml/base.yaml,conf/ml/lotus_sandbox.yaml",
        "OVERRIDES": "[]",
        "ARMS_PER_MODEL": "63",
        "SLICE_COUNT": "6",
        "ALLOW_FAILED_CHECKS": "false",
        "TOP_K": "500",
        "OUTPUT_BUCKET": "lotus-sandbox-sagemaker-data",
        "OUTPUT_PREFIX": "demo_model_results/",
        "RETRAIN_COOLDOWN_DAYS": "7",
        "LAST_TRAINING_PARAMETER": PARAM,
        "AWS_DEFAULT_REGION": "ca-central-1",
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MODELS", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def ssm():
    with mock_aws():
        client = boto3.client("ssm")
        client.put_parameter(Name=PARAM, Value="1970-01-01T00:00:00+00:00", Type="String")
        yield client


def _train_args(hyperparameters):
    return train.build_parser().parse_args(sm_common.hyperparameters_to_argv(hyperparameters))


# --------------------------------------------------------------------------
# training_context
# --------------------------------------------------------------------------

def test_training_context_builds_prepare_slices_and_finalize_for_one_run(ssm):
    ctx = helper.handler({"action": "training_context",
                          "execution_name": "0b1c2d3e-aaaa-bbbb-cccc-123456789abc",
                          "start_time": "2026-09-29T02:20:20.123Z"})
    assert ctx["run_id"] == "20260929T022020Z"
    assert ctx["total_arms"] == 630 and len(ctx["slices"]) == 6

    prepare = _train_args(ctx["prepare"]["hyperparameters"])
    assert prepare.mode == "prepare" and not prepare.allow_failed_checks
    assert prepare.config == ["conf/ml/base.yaml", "conf/ml/lotus_sandbox.yaml"]

    covered = []
    for item in ctx["slices"]:
        args = _train_args(item["hyperparameters"])
        assert args.mode == "sweep" and args.selection_run_id == "20260929T022020Z"
        covered += list(range(args.arm_start, args.arm_start + args.arm_count))
    assert covered == list(range(630))  # every arm exactly once

    final = _train_args(ctx["finalize"]["hyperparameters"])
    assert final.mode == "finalize" and final.selection_run_id == "20260929T022020Z"

    names = [ctx["prepare"]["job_name"], ctx["finalize"]["job_name"]] + \
        [s["job_name"] for s in ctx["slices"]]
    assert len(set(names)) == len(names)
    for name in names:
        assert len(name) <= 63 and name.startswith("ts05-20260929t022020z-")
        assert all(c.isalnum() or c == "-" for c in name)
    # the cooldown clock was started
    assert ssm.get_parameter(Name=PARAM)["Parameter"]["Value"].startswith("2026-09-29T02:20:20")


def test_fewer_models_and_overrides_flow_through(ssm, monkeypatch):
    monkeypatch.setenv("MODELS", "ecod,isolation_forest")
    monkeypatch.setenv("SLICE_COUNT", "2")
    monkeypatch.setenv("OVERRIDES", json.dumps(["run.mode=pilot", "sweep.max_hyperparam_configs=3"]))
    monkeypatch.setenv("ALLOW_FAILED_CHECKS", "true")
    ctx = helper.training_context({"start_time": "2026-09-29T02:20:20Z"})
    assert ctx["total_arms"] == 126
    assert [(s["hyperparameters"]["arm-start"], s["hyperparameters"]["arm-count"])
            for s in ctx["slices"]] == [("0", "63"), ("63", "63")]
    args = _train_args(ctx["prepare"]["hyperparameters"])
    assert args.overrides == ["run.mode=pilot", "sweep.max_hyperparam_configs=3"]
    assert args.allow_failed_checks and args.models == "ecod,isolation_forest"


def test_two_runs_in_the_same_second_get_different_job_names(ssm):
    a = helper.training_context({"start_time": "2026-09-29T02:20:20Z", "execution_name": "a"})
    b = helper.training_context({"start_time": "2026-09-29T02:20:20Z", "execution_name": "b"})
    assert a["prepare"]["job_name"] != b["prepare"]["job_name"]


# --------------------------------------------------------------------------
# scoring_context
# --------------------------------------------------------------------------

def _s3_event(key):
    return {"action": "scoring_context", "execution_name": "exec-1",
            "start_time": "2026-09-30T01:05:09Z",
            "detail": {"bucket": {"name": "lotus-sandbox-gold-curated-data"},
                       "object": {"key": key}}}


def test_scoring_context_scores_the_markers_folder_and_names_the_batch():
    ctx = helper.handler(_s3_event("mock-data/version1/test/batch_date=2026-09-29/_SUCCESS"))
    assert ctx["batch_date"] == "2026-09-29"
    assert ctx["input_uri"] == "s3://lotus-sandbox-gold-curated-data/mock-data/version1/test/batch_date=2026-09-29/"
    args = inference.build_parser().parse_args(ctx["container_arguments"])
    assert args.input == ctx["input_uri"] and args.batch_date == "2026-09-29"
    assert args.top_k == 500 and args.config == ["conf/ml/base.yaml", "conf/ml/lotus_sandbox.yaml"]
    assert ctx["summary_s3_prefix"] == f"s3://{ctx['summary_bucket']}/{ctx['summary_key'][:-len('summary.json')]}"
    assert ctx["summary_key"].startswith("demo_model_results/scoring-summaries/batch_date=2026-09-29/")
    assert len(ctx["job_name"]) <= 63 and ctx["job_name"].startswith("ts05-score-2026-09-29-")


def test_a_marker_without_a_date_folder_uses_the_execution_date():
    ctx = helper.scoring_context(_s3_event("mock-data/version1/test/_SUCCESS"))
    assert ctx["batch_date"] == "2026-09-30"
    assert ctx["input_uri"].endswith("/mock-data/version1/test/")


def test_a_non_s3_event_is_refused():
    with pytest.raises(ValueError, match="not an S3 Object Created event"):
        helper.scoring_context({"detail": {}})


# --------------------------------------------------------------------------
# retrain_gate
# --------------------------------------------------------------------------

def test_retrain_is_allowed_when_no_training_ran_recently(ssm):
    assert helper.retrain_gate({"start_time": "2026-09-30T00:00:00Z"})["allowed"] is True


def test_retrain_is_blocked_inside_the_cooldown(ssm):
    ssm.put_parameter(Name=PARAM, Value="2026-09-27T00:00:00+00:00", Type="String", Overwrite=True)
    gate = helper.retrain_gate({"start_time": "2026-09-30T00:00:00Z"})
    assert gate["allowed"] is False
    assert gate["next_allowed_after"].startswith("2026-10-04")


def test_the_gate_does_not_move_the_clock(ssm):
    helper.retrain_gate({"start_time": "2026-09-30T00:00:00Z"})
    assert ssm.get_parameter(Name=PARAM)["Parameter"]["Value"].startswith("1970")


def test_unknown_action_is_refused():
    with pytest.raises(ValueError, match="unknown action"):
        helper.handler({"action": "nope"})


# --------------------------------------------------------------------------
# the lotus overlay itself
# --------------------------------------------------------------------------

def test_the_lotus_overlay_loads_and_points_at_the_lotus_buckets():
    from trust_score_05.ml.config import load_ml_config

    cfg = load_ml_config(config_paths=sm_common.resolve_configs(
        ["conf/ml/base.yaml", "conf/ml/lotus_sandbox.yaml"]))
    assert cfg.training_root.startswith("s3://lotus-sandbox-gold-curated-data/")
    assert cfg.models_root.startswith("s3://lotus-sandbox-sagemaker-data/demo_model_results/")
    assert len(cfg.top_n_feature_counts) * len(cfg.preprocessing_ids) == 63  # ARMS_PER_MODEL
