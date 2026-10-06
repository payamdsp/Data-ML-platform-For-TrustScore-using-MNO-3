# Dockerfile
# Custom container for the SageMaker Processing Job:
# PySpark preprocessing (fill missing values)

FROM python:3.11-slim

# Java is required for PySpark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless \
        procps && \
    rm -rf /var/lib/apt/lists/*

# Configure Java
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Working directory inside the container
WORKDIR /opt/ml/processing

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy preprocessing code
COPY preprocess.py .

# Run preprocessing when the container starts
ENTRYPOINT ["python", "preprocess.py"]