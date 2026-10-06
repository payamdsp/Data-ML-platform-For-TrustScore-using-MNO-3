"""The contract between the launchers and the containers they start.

    local\\.venv\\Scripts\\python.exe -m pytest sagemaker/tests/test_launchers.py -q

No AWS: every request is built exactly as ``--dry-run`` builds it and then fed
to the parser that will read it inside the image. A launcher and an entry point
that disagree on a flag name fail here instead of in a paid job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

SAGEMAKER_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SAGEMAKER_DIR), str(SAGEMAKER_DIR / "scripts")]

import inference  # noqa: E402
import launch  # noqa: E402
import sm_common  # noqa: E402
import submit_emr_serverless as emrs  # noqa: E402
import train  # noqa: E402

COMMON = ["--image-uri", "123.dkr.ecr.ca-central-1.amazonaws.com/ts05:1",
          "--role-arn", "arn:aws:iam::123:role/sm", "--output-s3", "s3://bucket/sm/"]


# --------------------------------------------------------------------------
# slicing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("total,slices", [(63, 4), (7, 7), (10, 3), (5, 9), (1, 1)])
def test_slices_partition_the_arms_exactly(total, slices):
    ranges = launch.slice_arms(total, slices)
    covered = [i for r in ranges for i in range(r["start"], r["start"] + r["count"])]
    assert covered == list(range(total))          # disjoint, contiguous, complete
    assert all(r["count"] > 0 for r in ranges)     # no empty (wasted) job
    assert max(r["count"] for r in ranges) - min(r["count"] for r in ranges) <= 1


N_MODELS = len(launch._model_names())
FULL_SWEEP = N_MODELS * 7 * 9  # models x 7 feature counts x 9 preprocessing variants


def test_the_sandbox_sweep_has_the_expected_number_of_arms():
    assert launch.count_arms(None, launch._model_names()) == FULL_SWEEP


# --------------------------------------------------------------------------
# launch.py -> train.py
# --------------------------------------------------------------------------

def _requests(capsys, argv):
    launch.main(argv)
    text = capsys.readouterr().out
    decoder, pos, out = json.JSONDecoder(), 0, []
    while pos < len(text):
        text_rest = text[pos:].lstrip()
        if not text_rest:
            break
        obj, used = decoder.raw_decode(text_rest)
        out.append(obj)
        pos = len(text) - len(text_rest) + used
    return out


def _parse_in_container(request) -> argparse.Namespace:
    """What train.py sees: hyperparameters.json -> argv -> its own parser."""
    argv = sm_common.hyperparameters_to_argv(request["HyperParameters"])
    return train.build_parser().parse_args(argv)


def test_every_slice_reaches_train_py_with_its_range_and_the_shared_run_id(capsys):
    requests = _requests(capsys, ["train", "--slices", "4", "--run-id", "20260929T010000Z",
                                  "--dry-run", "--then-finalize"] + COMMON)
    sweeps, finalize = requests[:-1], requests[-1]
    assert len(sweeps) == 4
    starts = []
    for request in sweeps:
        args = _parse_in_container(request)
        assert args.mode == "sweep"
        assert request["Environment"]["TS05_RUN_ID"] == "20260929T010000Z"
        assert request["EnableManagedSpotTraining"] is True
        stop = request["StoppingCondition"]
        assert stop["MaxWaitTimeInSeconds"] >= stop["MaxRuntimeInSeconds"]
        assert len(request["TrainingJobName"]) <= 63
        assert sm_common.resolve_configs(args.config)[-1].endswith("conf/ml/sandbox.yaml") or \
            sm_common.resolve_configs(args.config)[-1].endswith("conf\\ml\\sandbox.yaml")
        starts.append((args.arm_start, args.arm_count))
    assert sum(c for _, c in starts) == FULL_SWEEP
    fin = _parse_in_container(finalize)
    assert fin.mode == "finalize" and finalize["Environment"]["TS05_RUN_ID"] == "20260929T010000Z"


def test_finalize_flags_survive_the_hyperparameter_round_trip(capsys):
    (request,) = _requests(capsys, ["finalize", "--run-id", "R1", "--allow-partial",
                                    "--no-promote", "--dry-run"] + COMMON)
    args = _parse_in_container(request)
    assert args.allow_partial and args.no_promote and args.mode == "finalize"
    assert request["ResourceConfig"]["InstanceType"] == "ml.m6i.2xlarge"


def test_prepare_reaches_train_py_in_prepare_mode(capsys):
    (request,) = _requests(capsys, ["prepare", "--run-id", "R2", "--allow-failed-checks",
                                    "--dry-run"] + COMMON)
    args = _parse_in_container(request)
    assert args.mode == "prepare" and args.allow_failed_checks
    assert request["Environment"]["TS05_RUN_ID"] == "R2"


def test_sdk_style_json_encoded_hyperparameters_are_accepted():
    argv = sm_common.hyperparameters_to_argv({"mode": '"sweep"', "arm-start": '"12"',
                                              "force": '"true"', "no-promote": "false"})
    args = train.build_parser().parse_args(argv)
    assert (args.mode, args.arm_start, args.force, args.no_promote) == ("sweep", 12, True, False)


def test_the_train_token_sagemaker_passes_is_ignored():
    assert sm_common.container_argv(["train", "--mode", "finalize"]) == ["--mode", "finalize"]


# --------------------------------------------------------------------------
# launch.py -> inference.py
# --------------------------------------------------------------------------

def test_the_scoring_job_arguments_parse_in_inference_py(capsys):
    (request,) = _requests(capsys, ["score", "--input", "s3://b/batch/batch_date=2026-09-29/",
                                    "--batch-date", "2026-09-29", "--dry-run"] + COMMON)
    spec = request["AppSpecification"]
    assert spec["ContainerEntrypoint"] == ["python3", "/opt/program/inference.py"]
    args = inference.build_parser().parse_args(spec["ContainerArguments"])
    assert args.input == "s3://b/batch/batch_date=2026-09-29/"
    assert args.batch_date == "2026-09-29" and args.top_k == 500
    assert args.config == ["conf/ml/base.yaml", "conf/ml/sandbox.yaml"]
    assert "Tags" in request and request["RoleArn"].startswith("arn:aws:iam::")


# --------------------------------------------------------------------------
# submit_emr_serverless.py
# --------------------------------------------------------------------------

def _emr(argv):
    args = emrs.parse_args(["--dry-run", "--artifact-root", "s3://art/ts05/"] + argv)
    return emrs.build_request(args)


def test_lineage_gets_iceberg_and_sandbox_configs_by_basename():
    request = _emr(["lineage", "--run-id", "R9"])
    driver = request["jobDriver"]["sparkSubmit"]
    params = driver["sparkSubmitParameters"]
    assert driver["entryPoint"] == "s3://art/ts05/lineage_job.py"
    assert driver["entryPointArguments"] == ["--config", "base.yaml", "--config", "sandbox.yaml"]
    assert "--files s3://art/ts05/conf/lineage/base.yaml,s3://art/ts05/conf/lineage/sandbox.yaml" in params
    assert "spark.sql.extensions=" + emrs.ICEBERG_EXTENSIONS in params
    assert "spark.emr-serverless.driverEnv.TS05_RUN_ID=R9" in params
    assert "--py-files s3://art/ts05/trust_score_05.zip,s3://art/ts05/pydeps.zip" in params


def test_ml_jobs_get_no_iceberg_and_their_own_family():
    params = _emr(["ml-selection"])["jobDriver"]["sparkSubmit"]["sparkSubmitParameters"]
    assert "iceberg" not in params.lower()
    assert "conf/ml/sandbox.yaml" in params


def test_two_configs_with_one_basename_are_refused():
    with pytest.raises(SystemExit, match="both named base.yaml"):
        _emr(["lineage", "--config-files", "s3://a/lineage/base.yaml,s3://a/ml/base.yaml"])


def test_job_arguments_after_the_separator_reach_the_driver():
    request = _emr(["imei_map", "--", "--dry-run"])
    assert request["jobDriver"]["sparkSubmit"]["entryPointArguments"][-1] == "--dry-run"


def test_the_sweep_is_not_submittable_to_emr():
    with pytest.raises(SystemExit):
        emrs.build_parser().parse_args(["ml-sweep"])
