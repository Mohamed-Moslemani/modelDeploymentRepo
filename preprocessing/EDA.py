import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

logger = logging.getLogger(__name__)

class EDA:
    def __init__(self, dataframe):
        self.df = dataframe

    def show_head(self, n=5):
        return self.df.head(n)

    def show_info(self):
        return self.df.info()
    
    def show_columns(self):
        return self.df.columns

    def summary_statistics(self):
        return self.df.describe()

    def missing_values(self):
        return self.df.isnull().sum()

    def show_shape(self):
        return self.df.shape


class DataPreprocessor:
    """Handle all data preprocessing tasks including handling missing values and encoding categorical features."""
    
    def __init__(self, data, target_column):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data: DataFrame with features and target
            target_column: Name of the target column
        """
        self.data = data.copy()
        self.target_column = target_column
        self.label_encoders = {}
        
    def preprocess_data(self):
        """Preprocess data by handling missing values and encoding categorical features."""
        try:
            # Handle missing values
            self.data = self.data.replace('N/A', np.nan)
            
            # Fill numeric missing values with median
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
            
            # Encode categorical features
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != self.target_column:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.label_encoders[col] = le
            
            logger.info("Data preprocessing completed")
            return self.data
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def get_processed_data(self):
        """Return the preprocessed data."""
        return self.data
    
    def get_label_encoders(self):
        """Return the dictionary of label encoders."""
        return self.label_encoders