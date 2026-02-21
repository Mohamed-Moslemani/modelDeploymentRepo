import mlflow
import logging
import os

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.dir_path = "archive/"
        self.file_path = "data.csv"
        self.file_type = "csv"
        
        # Configure MLflow
        try:
            # Use local filesystem backend for artifact storage
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("my_experiment")
            # End any active runs before configuration
            mlflow.end_run()
            logger.info("MLflow configured successfully with local filesystem backend")
        except Exception as e:
            logger.error(f"Failed to configure MLflow: {e}")

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)


class ExperimentConfig:
    """Configuration for machine learning experiments"""
    
    @staticmethod
    def get_experiment_configs(X_train_shape):
        """
        Get comprehensive experiment configurations.
        
        Args:
            X_train_shape: Shape of training data (used for feature count calculations)
        
        Returns:
            List of experiment configuration dictionaries
        """
        return [
            # ---- RUN 1: Baseline with All Features, Default Split ----
            {
                'name': 'Run1_Baseline_AllFeatures_80-20Split',
                'description': 'Baseline model with all features, 80-20 train-test split',
                'hyperparams': {
                    'n_estimators': 50,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                },
                'feature_strategy': 'all',
                'test_size': 0.2,
                'feature_count': None,
            },
            
            # ---- RUN 2: Aggressive Hyperparameters with Feature Selection (f_classif) ----
            {
                'name': 'Run2_Aggressive_KBestF_70-30Split',
                'description': 'Aggressive model (more trees, deeper) with f_classif feature selection, 70-30 split',
                'hyperparams': {
                    'n_estimators': 150,
                    'max_depth': 25,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                },
                'feature_strategy': 'kbest_f',
                'test_size': 0.3,
                'feature_count': None,  # Will use 80% of features
            },
            
            # ---- RUN 3: Conservative with Mutual Information Feature Selection ----
            {
                'name': 'Run3_Conservative_KBestMI_90-10Split',
                'description': 'Conservative model (fewer trees, shallower) with mutual_info feature selection, 90-10 split',
                'hyperparams': {
                    'n_estimators': 30,
                    'max_depth': 5,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                },
                'feature_strategy': 'kbest_mi',
                'test_size': 0.1,
                'feature_count': None,
            },
            
            # ---- RUN 4: Deep Model with Limited Features (f_classif) ----
            {
                'name': 'Run4_Deep_LimitedFeatures_F_80-20Split',
                'description': 'Deep model with 50% of best features selected by f_classif',
                'hyperparams': {
                    'n_estimators': 100,
                    'max_depth': 30,
                    'min_samples_split': 3,
                    'min_samples_leaf': 2,
                },
                'feature_strategy': 'kbest_f',
                'test_size': 0.2,
                'feature_count': max(1, int(X_train_shape[1] * 0.5)),  # Use only 50% of features
            },
            
            # ---- RUN 5: Shallow Model with Mutual Info Features ----
            {
                'name': 'Run5_Shallow_LimitedFeatures_MI_75-25Split',
                'description': 'Shallow model with 60% of best features selected by mutual_info',
                'hyperparams': {
                    'n_estimators': 80,
                    'max_depth': 8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 3,
                },
                'feature_strategy': 'kbest_mi',
                'test_size': 0.25,
                'feature_count': max(1, int(X_train_shape[1] * 0.6)),  # Use 60% of features
            },
        ]