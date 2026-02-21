
import mlflow

class config: 
    def __init__(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("my_experiment")
        self.run = mlflow.start_run()

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def end_run(self):
        mlflow.end_run()