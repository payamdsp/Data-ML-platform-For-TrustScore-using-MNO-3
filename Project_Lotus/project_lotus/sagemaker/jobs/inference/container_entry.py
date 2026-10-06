"""Entry point of the inference image: one image, two ways SageMaker runs it.

    docker run <image> serve                -> the Flask inference server (endpoint)
    docker run <image> --input s3://... ... -> inference.py, the nightly batch job

SageMaker hosting (Serverless or real-time endpoint) always starts the container
with the single argument ``serve`` and expects a server on port 8080. The batch
Processing job passes inference.py's own arguments - and the Step Functions
definition and launch.py set ContainerEntrypoint explicitly anyway, so the batch
path does not depend on this file.

The server runs under gunicorn with ONE worker process and several threads:
every worker holds its own copy of the model (pyod/hdbscan/torch objects), and a
Serverless endpoint has at most 6 GB of memory, so memory - not CPU - is what
limits worker count. Tune with TS05_SERVER_WORKERS / TS05_SERVER_THREADS.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "serve":
        port = os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080")
        command = [
            "gunicorn",
            "--bind", f"0.0.0.0:{port}",
            "--workers", os.environ.get("TS05_SERVER_WORKERS", "1"),
            "--threads", os.environ.get("TS05_SERVER_THREADS", "4"),
            "--worker-class", "gthread",
            # Model load happens on the first request, inside this limit.
            "--timeout", os.environ.get("TS05_SERVER_TIMEOUT", "120"),
            "--chdir", os.path.dirname(os.path.abspath(__file__)),
            "serve:create_app()",
        ]
        os.execvp(command[0], command)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import inference

    raise SystemExit(inference.main(args))


if __name__ == "__main__":
    main()
