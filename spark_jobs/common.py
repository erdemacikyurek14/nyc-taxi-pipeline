import os

from pyspark.sql import SparkSession

# Delta Lake and Kafka JAR coordinates (Maven)
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.2.0"
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"


def create_spark(app_name: str, include_kafka: bool = False) -> SparkSession:
    packages = [DELTA_PACKAGE]

    if include_kafka:
        kafka_pkg = os.getenv("SPARK_KAFKA_PACKAGE", KAFKA_PACKAGE)
        packages.append(kafka_pkg)

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", ",".join(packages))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    )

    return builder.getOrCreate()


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
