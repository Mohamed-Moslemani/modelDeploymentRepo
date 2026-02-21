import logging
import pandas as pd
import numpy as np
from modeling.registry import ModelRegistry
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Handle model inference and predictions using registered models from MLflow.
    """
    
    def __init__(self, model_name="stroke_prediction_model", stage="Production"):
        """
        Initialize the ModelInference class.
        
        Args:
            model_name: Name of the registered model
            stage: Stage of the model to load (e.g., 'Production', 'Staging')
        """
        self.model_name = model_name
        self.stage = stage
        self.model_registry = ModelRegistry(model_name)
        self.model = None
        self.label_encoders = {}
        self.selected_features = None  # Features used during training
        
    def load_model(self, stage=None):
        """
        Load a registered model from MLflow Model Registry.
        
        Args:
            stage: Stage of the model to load. If None, uses self.stage
        """
        try:
            stage = stage or self.stage
            self.model = self.model_registry.load_model_from_registry(stage=stage)
            
            logger.info(f"Model loaded successfully from stage: {stage}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model from stage '{stage}': {e}")
            raise
    
    def set_label_encoders(self, encoders_dict):
        """
        Set label encoders for categorical features.
        
        Args:
            encoders_dict: Dictionary of LabelEncoder objects for categorical columns
        """
        self.label_encoders = encoders_dict
        logger.info(f"Label encoders set for {len(encoders_dict)} columns")
    
    def load_selected_features(self, run_id):
        """
        Load the selected features used during model training from MLflow.
        
        Args:
            run_id: The MLflow run ID of the best model
            
        Returns:
            List of selected feature names
        """
        try:
            self.selected_features = self.model_registry.get_selected_features(run_id)
            if self.selected_features:
                logger.info(f"Selected features loaded from run {run_id}: {self.selected_features}")
            return self.selected_features
        except Exception as e:
            logger.error(f"Error loading selected features: {e}")
            return None
    
    def preprocess_input(self, input_data, categorical_cols=None):
        """
        Preprocess input data for inference.
        
        Args:
            input_data: Dictionary or DataFrame with input features
            categorical_cols: List of categorical column names
        
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        try:
            # Convert to DataFrame if dictionary
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Make a copy to avoid modifying original
            data = input_data.copy()
            
            # Handle missing values
            data = data.fillna(data.median(numeric_only=True))
            
            # Encode categorical features
            if categorical_cols:
                for col in categorical_cols:
                    if col in data.columns and col in self.label_encoders:
                        # Only transform if encoder is not None
                        if self.label_encoders[col] is not None:
                            try:
                                data[col] = self.label_encoders[col].transform(data[col].astype(str))
                            except ValueError as e:
                                logger.warning(f"Could not encode {col}: {e}")
                        else:
                            # If encoder is None, create a new one from the unique values in the data
                            logger.info(f"Encoder for {col} is None, creating new encoder from data")
                            le = LabelEncoder()
                            data[col] = le.fit_transform(data[col].astype(str))
                            self.label_encoders[col] = le
            
            # Get expected features from the model
            expected_features = None
            if self.selected_features:
                expected_features = self.selected_features
                logger.info(f"Using selected features from training: {expected_features}")
            elif hasattr(self.model, 'feature_names_in_'):
                try:
                    expected_features = list(self.model.feature_names_in_)
                    logger.info(f"Model was trained on features: {expected_features}")
                except (AttributeError, TypeError):
                    logger.info("Could not get feature names from model")
            elif hasattr(self.model, 'get_feature_names_in'):
                try:
                    expected_features = list(self.model.get_feature_names_in())
                    logger.info(f"Model expects features: {expected_features}")
                except (AttributeError, TypeError):
                    logger.info("Could not call get_feature_names_in")
            
            # Filter and reorder to match expected features
            if expected_features:
                # Ensure all expected features are present
                missing_features = [f for f in expected_features if f not in data.columns]
                if missing_features:
                    raise ValueError(f"Missing required features for model: {missing_features}. Available columns: {list(data.columns)}")
                
                # Select and reorder features to match training data
                data = data[expected_features]
                logger.info(f"Filtered to {len(expected_features)} expected features in correct order")
            else:
                logger.warning("Could not determine expected features, using available columns")
            
            logger.info(f"Input data preprocessed successfully. Shape: {data.shape}, Columns: {list(data.columns)}")
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing input data: {e}")
            raise
    
    def predict(self, input_data, categorical_cols=None):
        """
        Make predictions on input data.
        
        Args:
            input_data: Dictionary or DataFrame with input features
            categorical_cols: List of categorical column names
        
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data, categorical_cols)
            
            # Make predictions based on model type
            if hasattr(self.model, 'predict'):
                # For sklearn models and MLflow pyfunc models
                predictions = self.model.predict(processed_data)
            else:
                raise ValueError(f"Model does not have predict method: {type(self.model)}")
            
            logger.info(f"Predictions made successfully. Shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba(self, input_data, categorical_cols=None):
        """
        Make probability predictions on input data.
        
        Args:
            input_data: Dictionary or DataFrame with input features
            categorical_cols: List of categorical column names
        
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data, categorical_cols)
            
            # Check if predict_proba is available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(processed_data)
                logger.info(f"Probability predictions made successfully")
                return proba
            else:
                logger.warning("Model does not support predict_proba, returning None")
                return None
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            return None
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        try:
            version = self.model_registry.get_latest_version_by_stage(self.stage)
            if version:
                info = {
                    'model_name': self.model_name,
                    'stage': self.stage,
                    'version': version.version,
                    'run_id': version.run_id,
                    'created_timestamp': version.creation_timestamp,
                    'description': version.description,
                    'status': version.status
                }
                logger.info(f"Model info retrieved: {info}")
                return info
            return None
        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            raise
