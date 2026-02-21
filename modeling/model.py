import logging
import pickle
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, random_state=42):
        """
        Initialize the ModelTrainer.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            random_state: Random state for reproducibility
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.model = None
    
    def train(self, hyperparameters):
        """
        Train the Random Forest model with given hyperparameters.
        
        Args:
            hyperparameters: Dictionary of hyperparameters for RandomForestClassifier
        """
        try:
            self.model = RandomForestClassifier(**hyperparameters, random_state=self.random_state)
            self.model.fit(self.X_train, self.y_train)
            logger.info(f"Model trained successfully with hyperparameters: {hyperparameters}")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate(self):
        """
        Evaluate the model on the test set and return metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            }
            
            logger.info(f"Model evaluated - Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self, model_path="models"):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Directory to save the model
        """
        try:
            os.makedirs(model_path, exist_ok=True)
            filepath = os.path.join(model_path, "trained_model.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def log_to_mlflow(self, hyperparameters, metrics, model_path):
        """
        Log parameters, metrics, and artifacts to MLflow.
        
        Args:
            hyperparameters: Dictionary of model hyperparameters
            metrics: Dictionary of performance metrics
            model_path: Path to the saved model file
        """
        try:
            # Log hyperparameters
            mlflow.log_params(hyperparameters)
            logger.info(f"Logged hyperparameters to MLflow: {hyperparameters}")
            
            # Log metrics
            mlflow.log_metrics(metrics)
            logger.info(f"Logged metrics to MLflow: {metrics}")
            
            # Log model artifact
            mlflow.log_artifact(model_path, artifact_path="models")
            logger.info(f"Logged model artifact to MLflow")
            
            # Log additional metadata
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("training_samples", len(self.X_train))
            mlflow.log_param("test_samples", len(self.X_test))
            mlflow.log_param("feature_count", self.X_train.shape[1])
            
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            raise
