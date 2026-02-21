"""
Stroke Prediction UI Launcher
Launches the Gradio interface for making predictions with the registered MLflow model.
"""

import logging
from ui.gradio_ui import StrokePredictionUI
from config.env_loader import EnvLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Launch the Gradio UI for stroke prediction.
    """
    # Load environment variables
    EnvLoader.load()
    
    logger.info("="*80)
    logger.info("LAUNCHING STROKE PREDICTION UI")
    logger.info("="*80)
    
    try:
        # Get configuration from environment variables
        model_name = EnvLoader.get("MODEL_NAME", "stroke_prediction_model")
        model_stage = EnvLoader.get("MODEL_STAGE", "Production")
        gradio_server_name = EnvLoader.get("GRADIO_SERVER_NAME", "0.0.0.0")
        gradio_port = EnvLoader.get_int("GRADIO_PORT", 7860)
        gradio_share = EnvLoader.get_bool("GRADIO_SHARE", False)
        categorical_cols_str = EnvLoader.get("CATEGORICAL_COLUMNS", "ever_married,work_type,smoking_status")
        categorical_cols = [col.strip() for col in categorical_cols_str.split(",")]
        
        # Initialize the UI
        logger.info(f"Initializing Gradio UI with {model_stage} model: {model_name}...")
        ui = StrokePredictionUI(
            model_name=model_name,
            stage=model_stage
        )
        
        # Set categorical columns - MUST match the exact column names from training data
        # The model was trained on: age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status
        # Categorical columns among these are: ever_married, work_type, smoking_status
        ui.set_categorical_columns(categorical_cols)
        
        logger.info("UI initialized successfully!")
        logger.info("\n" + "="*80)
        logger.info("GRADIO INTERFACE DETAILS")
        logger.info("="*80)
        logger.info(f"Local URL: http://localhost:{gradio_port}")
        logger.info(f"Network URL: http://<your-ip>:{gradio_port}")
        logger.info("\nThe interface includes:")
        logger.info("  - Single Patient Prediction: Get stroke risk for one patient")
        logger.info("  - Batch Prediction: Upload CSV with multiple patients")
        logger.info("  - Model Information: View details about the registered model")
        logger.info("="*80 + "\n")
        
        # Launch the interface
        ui.launch(
            share=gradio_share,
            server_name=gradio_server_name,
            server_port=gradio_port
        )
        
    except Exception as e:
        logger.error(f"Error launching UI: {e}")
        raise


if __name__ == "__main__":
    main()
