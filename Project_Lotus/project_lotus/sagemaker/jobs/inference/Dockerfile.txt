# =============================================================================
# Trust Score 0.5 - INFERENCE image  (inference.py: nightly batch scoring + drift)
#
#   docker build -f Dockerfile.inference -t lotus-sandbox-model_autoencoder_inference:<tag> .
#   (run from sagemaker/)
#
# Run as a SageMaker *Processing* job with
#   ContainerEntrypoint = ["python3", "/opt/program/inference.py"]
# (set by the Step Functions definition in terraform/ and by launch.py).
#
# No Java and no Spark: inference.py reads parquet with pandas and never
# imports pyspark (checked by the build step at the bottom). The model
# libraries ARE needed - a model.joblib bundle unpickles pyod / hdbscan / torch
# objects - and are pinned to the same versions as the training image, because a
# bundle only loads reliably under the versions that wrote it.
# =============================================================================
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PROGRAM_DIR=/opt/program

COPY jobs/inference/requirements-inference.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

WORKDIR /opt/program
COPY trust_score_05/ /opt/program/trust_score_05/
COPY conf/ /opt/program/conf/
COPY jobs/inference/sm_common.py jobs/inference/inference.py jobs/inference/serve.py jobs/inference/container_entry.py /opt/program/

RUN python3 -c "import sys; sys.path.insert(0, '/opt/program'); import inference, serve; \
assert not any(m.startswith('pyspark') for m in sys.modules); print('inference.py + serve.py import OK, no Spark')"

# Two uses of one image (see container_entry.py):
#   SageMaker endpoint  -> `serve`          -> gunicorn + serve.py (Flask) on :8080
#   batch Processing    -> inference.py args (and its ContainerEntrypoint is set explicitly)
EXPOSE 8080
ENTRYPOINT ["python3", "/opt/program/container_entry.py"]
