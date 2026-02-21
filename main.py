from preprocessing.EDA import EDA, DataPreprocessor
from data.load import LoadData
from config.config import Config, ExperimentConfig
from modeling.model import ModelTrainer
from modeling.registry import ModelRegistry
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_feature_selection(X_train, X_test, y_train, strategy='all', k=None):
    """
    Apply different feature selection strategies.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        strategy: 'all' (no selection), 'kbest_f' (SelectKBest with f_classif), 'kbest_mi' (SelectKBest with mutual_info)
        k: Number of features to select (if None, uses 80% of features)
    
    Returns:
        Transformed X_train, X_test, and feature names
    """
    if strategy == 'all':
        return X_train, X_test, list(X_train.columns)
    
    if k is None:
        k = max(1, int(X_train.shape[1] * 0.8))
    
    try:
        if strategy == 'kbest_f':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif strategy == 'kbest_mi':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            return X_train, X_test, list(X_train.columns)
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        # Convert back to DataFrame to maintain column names
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
        
        logger.info(f"Feature selection ({strategy}): {len(selected_features)} features selected out of {X_train.shape[1]}")
        return X_train_selected, X_test_selected, selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        return X_train, X_test, list(X_train.columns)

def run_experiment(X_train, X_test, y_train, y_test, config, hyperparameters, run_name, 
                   feature_strategy='all', test_size=0.2, feature_count=None):
    """
    Run a single experiment with given hyperparameters and feature selection.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        config: Config object
        hyperparameters: Dictionary of hyperparameters
        run_name: Name for this MLflow run
        feature_strategy: Feature selection strategy ('all', 'kbest_f', 'kbest_mi')
        test_size: Test set size ratio
        feature_count: Number of features to select
    
    Returns:
        Tuple of (selected_features, trainer) for later use in inference
    """
    try:
        with mlflow.start_run(run_name=run_name):
            logger.info(f"Starting run: {run_name}")
            logger.info(f"Configuration - Feature Strategy: {feature_strategy}, Test Size: {test_size}, Features: {feature_count}")
            
            # Apply feature selection
            X_train_selected, X_test_selected, selected_features = apply_feature_selection(
                X_train, X_test, y_train, strategy=feature_strategy, k=feature_count
            )
            
            # Log feature selection info
            mlflow.log_param("feature_strategy", feature_strategy)
            mlflow.log_param("feature_count", len(selected_features))
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("total_features", len(selected_features))
            mlflow.log_param("selected_features", ",".join(selected_features))
            
            # Initialize trainer with pre-prepared data
            trainer = ModelTrainer(
                X_train=X_train_selected,
                X_test=X_test_selected,
                y_train=y_train,
                y_test=y_test,
                random_state=42
            )
            
            # Train model
            logger.info(f"Training model with hyperparameters: {hyperparameters}")
            trainer.train(hyperparameters)
            
            # Evaluate model
            metrics = trainer.evaluate()
            
            # Save and log model
            model_path = trainer.save_model("models")
            trainer.log_to_mlflow(hyperparameters, metrics, model_path)
            
            logger.info(f"Run '{run_name}' completed successfully")
            logger.info(f"Performance - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
            
    except Exception as e:
        logger.error(f"Error in experiment run '{run_name}': {e}")
        mlflow.end_run()
        raise

try:
    # Initialize config and load data
    config = Config()
    data = LoadData(config.dir_path + config.file_path).load_csv(config.dir_path + config.file_path)

    # EDA for the data 
    eda = EDA(data)
    logger.info("Running EDA...")
    logger.info(f"Dataset shape: {eda.show_shape()}")
    logger.info(f"Columns: {list(eda.show_columns())}")
    logger.info(f"Missing values:\n{eda.missing_values()}")
    logger.info(f"Summary statistics:\n{eda.summary_statistics()}")
    
    # Preprocess the data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor(data, target_column='stroke')
    processed_data = preprocessor.preprocess_data()
    
    # Prepare train/test split
    X = processed_data.drop(columns=['stroke'])
    y = processed_data['stroke']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Data split - Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # ============================================
    # STEP 3: MULTIPLE RUNS WITH DIFFERENT CONFIGURATIONS
    # ============================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PERFORMING MULTIPLE RUNS WITH VARYING CONFIGURATIONS")
    logger.info("="*80 + "\n")
    
    # Load experiment configurations from config
    experiment_configs = ExperimentConfig.get_experiment_configs(X_train.shape)
    
    # Run all experiments
    logger.info(f"Total experiments to run: {len(experiment_configs)}\n")
    
    for i, exp_config in enumerate(experiment_configs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT {i}/{len(experiment_configs)}")
        logger.info(f"{'='*80}")
        logger.info(f"Run Name: {exp_config['name']}")
        logger.info(f"Description: {exp_config['description']}")
        logger.info(f"Hyperparameters: {exp_config['hyperparams']}")
        logger.info(f"Feature Selection: {exp_config['feature_strategy']}")
        logger.info(f"Train-Test Split: {int((1-exp_config['test_size'])*100)}-{int(exp_config['test_size']*100)}")
        logger.info(f"Feature Count: {exp_config['feature_count'] or 'All'}")
        logger.info(f"{'='*80}\n")
        
        run_experiment(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            config=config,
            hyperparameters=exp_config['hyperparams'],
            run_name=exp_config['name'],
            feature_strategy=exp_config['feature_strategy'],
            test_size=exp_config['test_size'],
            feature_count=exp_config['feature_count'],
        )
    
    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nNext Steps:")
    logger.info("1. View MLflow UI: mlflow ui")
    logger.info("2. Compare runs using the MLflow UI at http://localhost:5000")
    logger.info("3. Analyze metrics across different configurations")
    logger.info("\nExperiments configured:")
    for exp_config in experiment_configs:
        logger.info(f"  - {exp_config['name']}: {exp_config['description']}")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL REGISTRY AND VERSIONING")
    logger.info("="*80 + "\n")
    
    runs = mlflow.search_runs(
        order_by=["metrics.accuracy DESC"]
    )
    
    if runs is not None and not runs.empty:
        best_run = runs.iloc[0]
        best_run_id = best_run["run_id"]
        best_accuracy = best_run["metrics.accuracy"]
        
        logger.info(f"Best performing run: {best_run_id}")
        logger.info(f"Best accuracy: {best_accuracy:.4f}")
        
        model_registry = ModelRegistry("stroke_prediction_model")
        
        logger.info("\n--- Registering Best Model ---")
        model_version = model_registry.register_model(best_run_id)
        logger.info(f"Model registered successfully!")
        logger.info(f"Model Name: {model_version.name}")
        logger.info(f"Model Version: {model_version.version}")
        logger.info(f"Run ID: {model_version.run_id}")
        
        logger.info("\n--- Transitioning Model to Staging ---")
        model_registry.transition_model_stage(model_version.version, "Staging")
        logger.info(f"Model v{model_version.version} moved to Staging stage")
        
        logger.info("\n--- Setting Model Description ---")
        description = f"Best model with accuracy: {best_accuracy:.4f} trained on {len(X_train)} samples"
        model_registry.set_model_description(model_version.version, description)
        logger.info(f"Description: {description}")
        
        logger.info("\n--- Retrieving Model Version Details ---")
        version_info = model_registry.get_model_version(model_version.version)
        logger.info(f"Current Stage: {version_info.current_stage}")
        logger.info(f"Status: {version_info.status}")
        logger.info(f"Created Timestamp: {version_info.creation_timestamp}")
        
        logger.info("\n--- Listing All Model Versions ---")
        all_versions = model_registry.get_all_versions()
        logger.info(f"Total versions: {len(all_versions)}")
        for ver in all_versions:
            logger.info(f"  Version {ver.version}: Stage={ver.current_stage}, Status={ver.status}")
        
        logger.info("\n--- Transitioning Model from Staging to Production ---")
        model_registry.transition_model_stage(model_version.version, "Production")
        logger.info(f"Model v{model_version.version} moved to Production stage")
        
        logger.info("\n--- Verifying Production Model ---")
        production_model = model_registry.get_latest_version_by_stage("Production")
        if production_model:
            logger.info(f"Production Model Version: {production_model.version}")
            logger.info(f"Production Model Run ID: {production_model.run_id}")
            logger.info(f"Production Model Created: {production_model.creation_timestamp}")
        
        logger.info("\n" + "="*80)
        logger.info("MODEL REGISTRY OPERATIONS COMPLETED!")
        logger.info("="*80)
        logger.info("\nModel Registry Summary:")
        logger.info(f"Registered Model: stroke_prediction_model")
        logger.info(f"Best Version: {model_version.version}")
        logger.info(f"Current Stage: Production")
        logger.info(f"Accuracy: {best_accuracy:.4f}")
        
        logger.info("\n--- Loading Model from Registry ---")
        loaded_model = model_registry.load_model_from_registry(stage="Production")
        logger.info("Model successfully loaded from Production stage")
        
        # Get the run ID for loading selected features
        prod_model_info = model_registry.get_latest_version_by_stage("Production")
        if prod_model_info:
            prod_model_run_id = prod_model_info.run_id
        else:
            prod_model_run_id = None
    
    else:
        logger.warning("No completed runs found for model registry")
        prod_model_run_id = None
    
    logger.info("\n" + "="*80)
    logger.info("STEP 5: INFERENCE WITH REGISTERED MODEL")
    logger.info("="*80 + "\n")
    
    try:
        from inference.inference import ModelInference
        import mlflow.sklearn
        
        logger.info("Initializing Model Inference with Production model...")
        inference = ModelInference(
            model_name="stroke_prediction_model",
            stage="Production"
        )
        
        # Load the model
        inference.load_model()
        logger.info("Model loaded successfully from MLflow Registry (Production stage)")
        
        # Load selected features from the best model's run
        if prod_model_run_id:
            inference.load_selected_features(prod_model_run_id)
            logger.info("Selected features loaded from Production model run")
        
        # Get model info
        logger.info("\n--- Model Information ---")
        model_info = inference.get_model_info()
        if model_info:
            logger.info(f"Model Name: {model_info['model_name']}")
            logger.info(f"Version: {model_info['version']}")
            logger.info(f"Run ID: {model_info['run_id']}")
            logger.info(f"Stage: {model_info['stage']}")
            logger.info(f"Status: {model_info['status']}")
            if model_info['description']:
                logger.info(f"Description: {model_info['description']}")
        
        # Log the selected features
        if inference.selected_features:
            logger.info(f"\nSelected features for inference ({len(inference.selected_features)}): {inference.selected_features}")
        
        # Set the label encoders from preprocessing
        inference.set_label_encoders(preprocessor.get_label_encoders())
        logger.info("Label encoders set for inference")
        
        # Test inference on a few samples from the test set
        logger.info("\n--- Making Predictions on Test Dataset ---")
        
        # Select first 5 samples from test set
        test_sample = X_test.head(5).copy()
        actual_labels = y_test.head(5).values
        
        # Filter to only include features that the model expects (if selected features are available)
        if inference.selected_features:
            test_sample = test_sample[[col for col in inference.selected_features if col in test_sample.columns]]
        
        logger.info(f"Making predictions on {len(test_sample)} test samples...")
        logger.info(f"Test sample shape: {test_sample.shape}")
        logger.info(f"Test sample columns: {list(test_sample.columns)}")
        
        # Pass only the column names from the preprocessed features
        predictions = inference.predict(test_sample, categorical_cols=None)
        
        logger.info("\nInference Results:")
        logger.info("-" * 80)
        
        for idx, (pred, actual) in enumerate(zip(predictions, actual_labels)):
            pred_label = "Stroke Risk" if pred == 1 else "No Stroke"
            actual_label = "Stroke Risk" if actual == 1 else "No Stroke"
            match = "✓ CORRECT" if pred == actual else "✗ INCORRECT"
            
            logger.info(f"Sample {idx+1}:")
            logger.info(f"  Predicted: {pred_label}")
            logger.info(f"  Actual: {actual_label}")
            logger.info(f"  {match}")
            logger.info("")
        
        # Calculate accuracy on test set
        X_test_filtered = X_test.copy()
        if inference.selected_features:
            X_test_filtered = X_test_filtered[[col for col in inference.selected_features if col in X_test_filtered.columns]]
        
        all_predictions = inference.predict(X_test_filtered, categorical_cols=None)
        test_accuracy = (all_predictions == y_test.values).sum() / len(y_test)
        
        logger.info("-" * 80)
        logger.info(f"Inference Accuracy on Full Test Set: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Test with a single sample (dictionary input)
        logger.info("\n--- Single Sample Inference Example ---")
        
        sample_dict = {
            'age': 45.0,
            'gender': 'Male',
            'hypertension': 0,
            'heart_disease': 0,
            'ever_married': 'Yes',
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'avg_glucose_level': 100.0,
            'bmi': 25.0,
            'smoking_status': 'never'
        }
        
        logger.info("Sample Input (Dictionary):")
        for key, value in sample_dict.items():
            logger.info(f"  {key}: {value}")
        
        try:
            # Create a DataFrame from the sample
            sample_df = pd.DataFrame([sample_dict])
            
            # Apply the same preprocessing as training data
            sample_processed = sample_df.copy()
            
            # Encode categorical features using the preprocessor's encoders
            for col in preprocessor.get_label_encoders():
                if col in sample_processed.columns:
                    try:
                        sample_processed[col] = preprocessor.get_label_encoders()[col].transform(sample_processed[col].astype(str))
                    except Exception as e:
                        logger.warning(f"Could not encode {col} in sample: {e}")
            
            # Fill missing numeric values
            sample_processed = sample_processed.fillna(sample_processed.median(numeric_only=True))
            
            # Filter to only include features that the model expects (if selected features are available)
            if inference.selected_features:
                sample_processed = sample_processed[[col for col in inference.selected_features if col in sample_processed.columns]]
            
            logger.info(f"Sample processed shape: {sample_processed.shape}")
            logger.info(f"Sample processed columns: {list(sample_processed.columns)}")
            
            # Make prediction using only the features that match training data
            single_prediction = inference.predict(sample_processed, categorical_cols=None)
            single_pred_label = "Stroke Risk" if single_prediction[0] == 1 else "No Stroke"
            
            logger.info(f"\nPrediction: {single_pred_label}")
        except Exception as e:
            logger.warning(f"Could not make prediction on sample dictionary: {e}")
        
        logger.info("\n" + "="*80)
        logger.info("STEP 5 COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("\nTo use the Gradio UI for interactive predictions:")
        logger.info("  1. Run: python launch_ui.py")
        logger.info("  2. Open your browser to http://localhost:7860")
        logger.info("  3. Use the interface to make single or batch predictions")
        
    except Exception as e:
        logger.error(f"Error in Step 5 (Inference): {e}")
        raise

    
except Exception as e:
    logger.error(f"Error during execution: {e}")