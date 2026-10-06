"""Write the small run summary produced by the Spark driver."""
import json
from pathlib import Path
from urllib.parse import urlparse


def write_json(path, data):
    """Write JSON to S3 in EMR, with local-path support for development."""
    body = json.dumps(data, indent=2, default=str, sort_keys=True) + "\n"
    if path.startswith("s3://"):
        # boto3 is supplied by EMR; it is deliberately not vendored in the ZIP.
        import boto3

        location = urlparse(path)
        boto3.client("s3").put_object(
            Bucket=location.netloc,
            Key=location.path.lstrip("/"),
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
