FROM bitnamilegacy/spark:3.5.0

USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download Delta Lake and Kafka JARs into Spark's classpath
RUN spark-submit --packages \
      "io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1" \
      --class org.apache.spark.examples.SparkPi \
      /opt/bitnami/spark/examples/jars/spark-examples_2.12-3.5.0.jar 1 \
    && cp /root/.ivy2/jars/*.jar /opt/bitnami/spark/jars/ \
    && echo "JARs pre-loaded into /opt/bitnami/spark/jars/"

USER 1001

