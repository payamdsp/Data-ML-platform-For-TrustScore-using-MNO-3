# =============================================================================
# Trust Score 0.5 - TRAINING image  (train.py: prepare | sweep | finalize | all)
#
#   docker build -f Dockerfile.training -t lotus-sandbox-model_autoencoder_training:<tag> .
#   (run from sagemaker/)
#
# SageMaker Training runs the image as `docker run <image> train`;
# /usr/local/bin/train below runs train.py, which reads the job's
# /opt/ml/input/config/hyperparameters.json.
#
# Needs Java + PySpark + the S3A connector because the package reads the
# feature tables through Spark (local[*] inside the container).
# =============================================================================
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    SPARK_LOCAL_IP=127.0.0.1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PROGRAM_DIR=/opt/program

# Java 17 for local Spark (PySpark 3.5.1 supports 8/11/17). procps because
# Spark's launcher scripts call `ps`; curl to fetch the S3A jars below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY jobs/training/requirements-training.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

# The S3A connector, so local Spark can read s3:// (mapped in sm_common.build_local_spark).
# Versions must match the Hadoop that PySpark 3.5.1 ships (3.3.4). Each jar is
# checked against the SHA-1 Maven Central publishes beside it.
ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_SDK_BUNDLE_VERSION=1.12.262
RUN set -eu; \
    JARS="$(python -c 'import os, pyspark; print(os.path.join(os.path.dirname(pyspark.__file__), "jars"))')"; \
    for coord in "org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar" \
                 "com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_BUNDLE_VERSION}/aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VERSION}.jar"; do \
        url="https://repo1.maven.org/maven2/${coord}"; \
        name="$(basename "${coord}")"; \
        curl -fsSL "${url}" -o "${JARS}/${name}"; \
        echo "$(curl -fsSL "${url}.sha1")  ${JARS}/${name}" | sha1sum -c -; \
    done

WORKDIR /opt/program
COPY trust_score_05/ /opt/program/trust_score_05/
COPY conf/ /opt/program/conf/
COPY jobs/training/sm_common.py jobs/training/train.py /opt/program/

RUN printf '#!/bin/sh\nexec python3 /opt/program/train.py "$@"\n' > /usr/local/bin/train \
 && chmod +x /usr/local/bin/train \
 && python3 -c "import sys; sys.path.insert(0, '/opt/program'); import train; print('train.py imports OK')"

ENTRYPOINT []
CMD ["train"]
