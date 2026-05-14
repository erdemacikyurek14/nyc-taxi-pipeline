$ErrorActionPreference = "Stop"

Write-Host "Starting infrastructure..."
docker compose up -d zookeeper kafka mlflow spark

Write-Host "Publishing sample taxi records to Kafka..."
docker compose --profile pipeline run --rm producer

Write-Host "Writing Kafka messages to Bronze Delta..."
docker compose exec spark spark-submit spark_jobs/stream_to_bronze.py

Write-Host "Building Silver Delta layer..."
docker compose exec spark spark-submit spark_jobs/bronze_to_silver.py

Write-Host "Building Gold Delta feature layer..."
docker compose exec spark spark-submit spark_jobs/silver_to_gold.py

Write-Host "Training fare prediction models and logging to MLflow..."
docker compose exec -e MPLCONFIGDIR=/tmp/matplotlib -e GIT_PYTHON_REFRESH=quiet spark spark-submit ml_pipeline/model_training.py

Write-Host "Starting Streamlit Dashboard (Adim 7)..."
docker compose --profile dashboard up -d dashboard

Write-Host ""
Write-Host "=== PIPELINE TAMAMLANDI ==="
Write-Host "MLflow UI    : http://localhost:5000"
Write-Host "Dashboard    : http://localhost:8501"
