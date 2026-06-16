"""Main entry point for the Mental Health Treatment Prediction project."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging, print_section_header, print_model_comparison
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


def main():
    """Run the complete ML pipeline."""
    logger = setup_logging()

    print_section_header("MENTAL HEALTH TREATMENT PREDICTION")
    logger.info("Starting ML Pipeline...")

    # Step 1: Data Preprocessing
    print_section_header("STEP 1: DATA PREPROCESSING")
    preprocessor = DataPreprocessor()
    df = preprocessor.full_preprocess()
    logger.info(f"Preprocessed data shape: {df.shape}")

    # Step 2: Feature Engineering
    print_section_header("STEP 2: FEATURE ENGINEERING")
    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    logger.info(f"Data shape after feature engineering: {df.shape}")

    # Step 3: Split Data
    print_section_header("STEP 3: TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Step 4: Model Training
    print_section_header("STEP 4: MODEL TRAINING WITH HYPERPARAMETER TUNING")
    trainer = ModelTrainer()
    results = trainer.train_with_tuning(X_train_scaled, y_train, X_test_scaled, y_test)

    # Step 5: Model Selection
    print_section_header("STEP 5: MODEL SELECTION")
    best_model, best_model_name = trainer.select_best_model()

    # Step 6: Evaluation
    print_section_header("STEP 6: EVALUATION & REPORTS")
    evaluator = ModelEvaluator()

    evaluator.generate_confusion_matrix(best_model, X_test_scaled, y_test, best_model_name)
    evaluator.generate_roc_curve(best_model, X_test_scaled, y_test, best_model_name)
    evaluator.generate_feature_importance(best_model, X_train_scaled.columns.tolist(), best_model_name)
    evaluator.generate_classification_report(best_model, X_test_scaled, y_test)

    # Step 7: Save Best Model
    print_section_header("STEP 7: SAVE BEST MODEL")
    trainer.save_best_model()

    # Print Results
    best_name = print_model_comparison(results)

    print_section_header("PIPELINE COMPLETE")
    logger.info("All artifacts generated successfully!")
    logger.info("Run 'streamlit run app/streamlit_app.py' to launch the web app.")

    return results


if __name__ == "__main__":
    results = main()
