import logging
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = MlflowClient()

    def register_model(self, run_id, artifact_path="models"):
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_version = mlflow.register_model(model_uri, self.model_name)
            logger.info(f"Model registered: {self.model_name}, Version: {model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def transition_model_stage(self, version, stage):
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {self.model_name} v{version} transitioned to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise

    def get_model_version(self, version):
        try:
            model_version = self.client.get_model_version(self.model_name, version)
            return model_version
        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            raise

    def get_all_versions(self):
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            return versions
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            raise

    def get_latest_version_by_stage(self, stage):
        try:
            latest_version = self.client.get_latest_versions(self.model_name, stages=[stage])
            if latest_version:
                return latest_version[0]
            return None
        except Exception as e:
            logger.error(f"Error getting latest version by stage: {e}")
            raise

    def load_model_from_registry(self, version=None, stage=None):
        try:
            if stage:
                model_uri = f"models:/{self.model_name}/{stage}"
            elif version:
                model_uri = f"models:/{self.model_name}/{version}"
            else:
                raise ValueError("Either version or stage must be provided")
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded from registry: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from registry: {e}")
            raise

    def list_model_stages(self, version):
        try:
            model_version = self.client.get_model_version(self.model_name, version)
            return model_version.current_stage
        except Exception as e:
            logger.error(f"Error listing model stages: {e}")
            raise

    def get_model_description(self, version):
        try:
            model_version = self.client.get_model_version(self.model_name, version)
            return model_version.description
        except Exception as e:
            logger.error(f"Error getting model description: {e}")
            raise

    def set_model_description(self, version, description):
        try:
            self.client.update_model_version(
                name=self.model_name,
                version=version,
                description=description
            )
            logger.info(f"Description set for {self.model_name} v{version}")
        except Exception as e:
            logger.error(f"Error setting model description: {e}")
            raise

    def archive_model(self, version):
        try:
            self.transition_model_stage(version, "Archived")
            logger.info(f"Model {self.model_name} v{version} archived")
        except Exception as e:
            logger.error(f"Error archiving model: {e}")
            raise

    def get_best_model_by_metric(self, metric_name):
        try:
            experiment = mlflow.get_experiment_by_name("Default")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} DESC"]
            )
            
            if runs.empty:
                logger.warning(f"No runs found for metric {metric_name}")
                return None
            
            best_run = runs.iloc[0]
            return best_run["run_id"], best_run[f"metrics.{metric_name}"]
        except Exception as e:
            logger.error(f"Error getting best model by metric: {e}")
            raise
    
    def get_run_params(self, run_id):
        """
        Get all parameters from a specific run.
        
        Args:
            run_id: The MLflow run ID
            
        Returns:
            Dictionary of run parameters
        """
        try:
            run = self.client.get_run(run_id)
            return run.data.params
        except Exception as e:
            logger.error(f"Error getting run parameters: {e}")
            raise
    
    def get_selected_features(self, run_id):
        """
        Get the selected features used in a specific run.
        
        Args:
            run_id: The MLflow run ID
            
        Returns:
            List of selected feature names, or None if not found
        """
        try:
            params = self.get_run_params(run_id)
            selected_features_str = params.get("selected_features")
            
            if selected_features_str:
                selected_features = selected_features_str.split(",")
                logger.info(f"Retrieved {len(selected_features)} selected features from run {run_id}")
                return selected_features
            
            logger.warning(f"No selected features found in run {run_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting selected features: {e}")
            return None
