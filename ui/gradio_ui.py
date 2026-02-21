import logging
import gradio as gr
import pandas as pd
import numpy as np
from inference.inference import ModelInference
from typing import Tuple, Dict
import json

logger = logging.getLogger(__name__)


class StrokePredictionUI:
    """
    Gradio UI for stroke prediction using a registered MLflow model.
    """
    
    def __init__(self, model_name="stroke_prediction_model", stage="Production"):
        """
        Initialize the Gradio UI.
        
        Args:
            model_name: Name of the registered model
            stage: Stage of the model to load
        """
        self.model_name = model_name
        self.stage = stage
        self.inference = ModelInference(model_name, stage)
        self.categorical_cols = []
        self.model_loaded = False
        
        # Try to load the model
        try:
            self.inference.load_model(stage)
            self.model_loaded = True
            logger.info(f"Model loaded for UI: {model_name} ({stage})")
        except Exception as e:
            logger.warning(f"Could not load model at initialization: {e}")
            logger.info("Model will be loaded on first prediction attempt")
    
    def set_categorical_columns(self, categorical_cols):
        """
        Set categorical column names for preprocessing.
        
        Args:
            categorical_cols: List of categorical column names
        """
        self.categorical_cols = categorical_cols
        self.inference.set_label_encoders({col: None for col in categorical_cols})
    
    def predict_single(self, age, gender, hypertension, heart_disease, 
                       ever_married, work_type, residence_type, avg_glucose_level, 
                       bmi, smoking_status) -> Tuple[str, str]:
        """
        Make a prediction for a single sample.
        
        Args:
            age: Patient age
            gender: Gender ('Male' or 'Female')
            hypertension: Hypertension status (0 or 1)
            heart_disease: Heart disease status (0 or 1)
            ever_married: Ever married status ('Yes' or 'No')
            work_type: Work type
            residence_type: Residence type ('Urban' or 'Rural')
            avg_glucose_level: Average glucose level
            bmi: Body Mass Index
            smoking_status: Smoking status
        
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Ensure model is loaded
            if not self.model_loaded or self.inference.model is None:
                logger.info("Loading model for prediction...")
                self.inference.load_model(self.stage)
                self.model_loaded = True
            
            # Create input dictionary
            input_data = {
                'age': float(age),
                'gender': gender,
                'hypertension': int(hypertension),
                'heart_disease': int(heart_disease),
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': float(avg_glucose_level),
                'bmi': float(bmi),
                'smoking_status': smoking_status
            }
            
            # Make prediction
            prediction = self.inference.predict(input_data, self.categorical_cols)
            
            # Get probability predictions
            proba = self.inference.predict_proba(input_data, self.categorical_cols)
            
            # Format output
            pred_label = "High Risk (Stroke)" if prediction[0] == 1 else "Low Risk (No Stroke)"
            
            if proba is not None:
                confidence = f"{max(proba[0]) * 100:.2f}%"
            else:
                confidence = "N/A"
            
            logger.info(f"Prediction made: {pred_label} (Confidence: {confidence})")
            return pred_label, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return f"Error: {str(e)}", "N/A"
    
    def predict_batch(self, csv_data: str) -> Tuple[str, pd.DataFrame]:
        """
        Make predictions on a batch of samples from CSV data.
        
        Args:
            csv_data: CSV formatted string with sample data
        
        Returns:
            Tuple of (status message, predictions DataFrame)
        """
        try:
            # Ensure model is loaded
            if not self.model_loaded or self.inference.model is None:
                logger.info("Loading model for batch prediction...")
                self.inference.load_model(self.stage)
                self.model_loaded = True
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(csv_data))
            
            # Make predictions
            predictions = self.inference.predict(df, self.categorical_cols)
            
            # Create results DataFrame
            results = df.copy()
            results['prediction'] = predictions
            results['risk_level'] = results['prediction'].apply(
                lambda x: "High Risk (Stroke)" if x == 1 else "Low Risk (No Stroke)"
            )
            
            logger.info(f"Batch predictions made for {len(results)} samples")
            return f"Successfully made predictions for {len(results)} samples", results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return f"Error: {str(e)}", pd.DataFrame()
    
    def get_model_info(self) -> str:
        """
        Get information about the loaded model.
        
        Returns:
            Formatted string with model information
        """
        try:
            info = self.inference.get_model_info()
            if info:
                info_text = f"""
                **Model Information**
                
                - **Model Name**: {info['model_name']}
                - **Stage**: {info['stage']}
                - **Version**: {info['version']}
                - **Run ID**: {info['run_id']}
                - **Status**: {info['status']}
                - **Description**: {info['description'] or 'N/A'}
                """
                return info_text
            else:
                return "No model information available"
        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            return f"Error retrieving model info: {str(e)}"
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface.
        
        Returns:
            Gradio Blocks object
        """
        with gr.Blocks(title="Stroke Prediction Model", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🏥 Stroke Prediction Model
                
                This application uses a registered MLflow model to predict stroke risk based on patient health data.
                
                **Model Stage**: Production
                """
            )
            
            with gr.Tabs():
                # Tab 1: Single Prediction
                with gr.TabItem("Single Prediction"):
                    gr.Markdown("### Make a prediction for a single patient")
                    
                    with gr.Row():
                        with gr.Column():
                            age = gr.Slider(
                                minimum=0, maximum=120, value=50, 
                                label="Age", step=1
                            )
                            gender = gr.Radio(
                                choices=["Male", "Female"], 
                                value="Male", 
                                label="Gender"
                            )
                            hypertension = gr.Radio(
                                choices=[0, 1], 
                                value=0, 
                                label="Hypertension (1=Yes, 0=No)"
                            )
                            heart_disease = gr.Radio(
                                choices=[0, 1], 
                                value=0, 
                                label="Heart Disease (1=Yes, 0=No)"
                            )
                            ever_married = gr.Radio(
                                choices=["Yes", "No"], 
                                value="Yes", 
                                label="Ever Married"
                            )
                        
                        with gr.Column():
                            work_type = gr.Dropdown(
                                choices=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                                value="Private",
                                label="Work Type"
                            )
                            residence_type = gr.Radio(
                                choices=["Urban", "Rural"], 
                                value="Urban", 
                                label="Residence Type"
                            )
                            avg_glucose = gr.Slider(
                                minimum=0, maximum=300, value=100,
                                label="Average Glucose Level", step=1
                            )
                            bmi = gr.Slider(
                                minimum=10, maximum=60, value=25,
                                label="BMI (Body Mass Index)", step=0.1
                            )
                            smoking = gr.Dropdown(
                                choices=["never", "formerly", "smokes", "Unknown"],
                                value="never",
                                label="Smoking Status"
                            )
                    
                    predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
                    
                    with gr.Row():
                        prediction_output = gr.Textbox(
                            label="Prediction", 
                            interactive=False
                        )
                        confidence_output = gr.Textbox(
                            label="Confidence", 
                            interactive=False
                        )
                    
                    predict_btn.click(
                        fn=self.predict_single,
                        inputs=[age, gender, hypertension, heart_disease, 
                               ever_married, work_type, residence_type, 
                               avg_glucose, bmi, smoking],
                        outputs=[prediction_output, confidence_output]
                    )
                
                # Tab 2: Batch Prediction
                with gr.TabItem("Batch Prediction"):
                    gr.Markdown("### Make predictions on multiple patients")
                    gr.Markdown("Upload or paste CSV data with patient information")
                    
                    csv_input = gr.Textbox(
                        label="CSV Data",
                        lines=10,
                        placeholder="age,gender,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status\n50,Male,0,0,Yes,Private,Urban,100,25,never"
                    )
                    
                    batch_predict_btn = gr.Button("🔮 Predict Batch", variant="primary", size="lg")
                    
                    batch_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    
                    results_output = gr.Dataframe(
                        label="Predictions",
                        interactive=False
                    )
                    
                    batch_predict_btn.click(
                        fn=self.predict_batch,
                        inputs=[csv_input],
                        outputs=[batch_status, results_output]
                    )
                
                # Tab 3: Model Info
                with gr.TabItem("Model Information"):
                    gr.Markdown("### Registered Model Details")
                    
                    info_output = gr.Markdown(self.get_model_info())
                    
                    refresh_btn = gr.Button("🔄 Refresh Info")
                    refresh_btn.click(
                        fn=self.get_model_info,
                        outputs=[info_output]
                    )
            
            gr.Markdown(
                """
                ---
                
                **Instructions**:
                1. **Single Prediction**: Fill in patient information and click "Predict"
                2. **Batch Prediction**: Provide CSV data and click "Predict Batch"
                3. **Model Info**: View details about the registered model
                
                **Disclaimer**: This model is for educational purposes only and should not be used for medical decisions without professional consultation.
                """
            )
        
        return demo
    
    def launch(self, share=False, server_name="0.0.0.0", server_port=7860):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server name/IP
            server_port: Server port
        """
        demo = self.build_interface()
        demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )
