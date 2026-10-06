import os
from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType, LongType, StringType, StructField, StructType, TimestampType,
)

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .appName("silver_dq_tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield s
    s.stop()


@pytest.fixture
def schema_dir():
    return os.path.join(PKG_ROOT, "schemas")


@pytest.fixture
def acct_yaml():
    return os.path.join(PKG_ROOT, "checks", "dataset=account_changes_batch", "checks.yaml")


@pytest.fixture
def account_changes_df(spark):
    """A small account_changes_batch (9-col §6.1.2) frame.

    Mixed MNO, one duplicate record_id (rows 1 & 4 identical) so uniqueness has a
    known violation; hashes are real 64-hex so the now-active validity rule passes.
    Matches the account_changes_batch columns.
    """
    schema = StructType([
        StructField("record_id", StringType()),           # AWS: string
        StructField("phone_number_ac_hash", StringType()),
        StructField("event_type", StringType()),
        StructField("event_timestamp", TimestampType()),
        StructField("mno", StringType()),
        StructField("ingestion_ts", TimestampType()),
        StructField("source_name", StringType()),
        StructField("source_event_id", LongType()),        # AWS: bigint
        StructField("schema_version", IntegerType()),
    ])
    # record_id is sha256(bronze id) = 64-hex in the real table. Use valid hex64 here so the
    # record_id validity rule now committed in the yaml passes; rows 1 & 4 share a record_id
    # (the planted duplicate that drives the uniqueness violation).
    rid_1, rid_2, rid_3 = "a" * 64, "b" * 64, "c" * 64
    rows = [
        (rid_1, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "SIM_CHANGE",    datetime(2026, 6, 20, 0, 0, 0), "BELL",   datetime(2026, 6, 20, 1, 0, 0), "MYSQL", None, 1),
        (rid_2, "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210", "DEVICE_CHANGE", datetime(2026, 6, 20, 0, 0, 0), "ROGERS", datetime(2026, 6, 20, 2, 0, 0), "MYSQL", None, 1),
        (rid_3, "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff", "SIM_CHANGE",    datetime(2026, 6, 19, 0, 0, 0), "TELUS",  datetime(2026, 6, 20, 0, 0, 0), "MYSQL", None, 1),
        (rid_1, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "SIM_CHANGE",    datetime(2026, 6, 20, 0, 0, 0), "BELL",   datetime(2026, 6, 20, 1, 0, 0), "MYSQL", None, 1),
    ]
    return spark.createDataFrame(rows, schema)
