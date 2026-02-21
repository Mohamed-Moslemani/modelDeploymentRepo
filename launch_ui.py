"""
Stroke Prediction UI Launcher
Launches the Gradio interface for making predictions with the registered MLflow model.
"""

import logging
from ui.gradio_ui import StrokePredictionUI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Launch the Gradio UI for stroke prediction.
    """
    logger.info("="*80)
    logger.info("LAUNCHING STROKE PREDICTION UI")
    logger.info("="*80)
    
    try:
        # Initialize the UI
        logger.info("Initializing Gradio UI with Production model...")
        ui = StrokePredictionUI(
            model_name="stroke_prediction_model",
            stage="Production"
        )
        
        # Set categorical columns (these would be determined from your data)
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        ui.set_categorical_columns(categorical_cols)
        
        logger.info("UI initialized successfully!")
        logger.info("\n" + "="*80)
        logger.info("GRADIO INTERFACE DETAILS")
        logger.info("="*80)
        logger.info("Local URL: http://localhost:7860")
        logger.info("Network URL: http://<your-ip>:7860")
        logger.info("\nThe interface includes:")
        logger.info("  - Single Patient Prediction: Get stroke risk for one patient")
        logger.info("  - Batch Prediction: Upload CSV with multiple patients")
        logger.info("  - Model Information: View details about the registered model")
        logger.info("="*80 + "\n")
        
        # Launch the interface
        ui.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860
        )
        
    except Exception as e:
        logger.error(f"Error launching UI: {e}")
        raise


if __name__ == "__main__":
    main()
