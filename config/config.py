import mlflow
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.dir_path = "archive/"
        self.file_path = "data.csv"
        self.file_type = "csv"
        
        # Configure MLflow
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("my_experiment")
            # End any active runs before configuration
            mlflow.end_run()
            logger.info("MLflow configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure MLflow: {e}")

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)