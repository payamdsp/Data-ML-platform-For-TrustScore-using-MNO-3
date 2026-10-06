"""
Preprocessing entrypoint for the SageMaker Processing Job.

SageMaker Processing convention:
  - Input data is mounted (downloaded from S3) into /opt/ml/processing/input
  - Anything written to /opt/ml/processing/output is uploaded back to S3
    automatically when the job finishes.

Uses PySpark in local mode (single container, multi-threaded) - no separate
EMR cluster involved. Good fit for small/medium files; if the real dataset
grows large enough to need a genuine distributed cluster, this same script's
transform logic can be lifted into an actual EMR Serverless job later with
minimal changes, since it's plain PySpark.
"""

import glob
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("telecom-fraud-preprocessing")
        .master("local[*]")
        .getOrCreate()
    )

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    print("Reading input files:", csv_files)
    df = spark.read.option("header", True).option("inferSchema", True).csv(csv_files)

    print("Input row count:", df.count())
    print("Schema:")
    df.printSchema()

    # --- Fill missing values ---
    # Numeric columns: fill with column median.
    # Non-numeric (string) columns: fill with a literal "UNKNOWN".
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    string_cols = [f.name for f in df.schema.fields if f.name not in numeric_cols]

    for col_name in numeric_cols:
        median_val = df.approxQuantile(col_name, [0.5], 0.01)
        median_val = median_val[0] if median_val else 0.0
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull(), F.lit(median_val)).otherwise(F.col(col_name)),
        )

    for col_name in string_cols:
        df = df.fillna({col_name: "UNKNOWN"})

    print("Row count after fill:", df.count())

    # Write single CSV file (coalesce(1) - fine for small/medium data;
    # remove for genuinely large datasets so Spark can write partitioned output).
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_DIR, "processed")
    )

    print("Preprocessing complete. Output written to:", OUTPUT_DIR)
    spark.stop()


if __name__ == "__main__":
    main()
