"""Model training module for Mental Health Treatment Prediction."""

import os
import time
import tempfile
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.utils import (
    setup_logging, save_model, save_json, get_model_path, get_reports_path
)
from src.evaluate import ModelEvaluator

logger = setup_logging()


class ModelTrainer:
    """Trains and compares multiple ML models."""

    def __init__(self):
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.param_grids = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"],
            },
            "Decision Tree": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
            "XGBoost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "LightGBM": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "CatBoost": {
                "iterations": [100, 200, 300],
                "depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        }

    def _create_models(self):
        """Create fresh model instances for each name."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(
                random_state=42, eval_metric="logloss"
            ),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
            "CatBoost": CatBoostClassifier(
                random_state=42, verbose=0, train_dir=tempfile.gettempdir()
            ),
        }

    def train_with_tuning(self, X_train, y_train, X_test, y_test):
        """Train all models with hyperparameter tuning."""
        evaluator = ModelEvaluator()
        models_to_train = self._create_models()

        for name, model in models_to_train.items():
            logger.info(f"\nTraining {name}...")
            start_time = time.time()

            param_grid = self.param_grids.get(name, {})

            if param_grid:
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=10,
                    cv=5,
                    scoring="f1",
                    random_state=42,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_est = search.best_estimator_
                logger.info(f"Best params: {search.best_params_}")
            else:
                best_est = model
                best_est.fit(X_train, y_train)

            metrics = evaluator.evaluate(best_est, X_test, y_test)
            elapsed = time.time() - start_time

            metrics["training_time"] = elapsed
            self.results[name] = metrics
            self.trained_models[name] = best_est

            logger.info(
                f"{name} - F1: {metrics['f1_score']:.4f}, "
                f"Time: {elapsed:.2f}s"
            )

        return self.results

    def select_best_model(self):
        """Select the model with the highest F1 Score."""
        best_f1 = 0
        for name, metrics in self.results.items():
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                self.best_model_name = name
                self.best_model = self.trained_models[name]

        logger.info(f"\nBest model: {self.best_model_name} (F1: {best_f1:.4f})")
        return self.best_model, self.best_model_name

    def save_best_model(self):
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No model selected. Run train_with_tuning first.")

        model_path = get_model_path() / "best_model.pkl"
        save_model(self.best_model, model_path)
        logger.info(f"Best model saved to {model_path}")

        results_path = get_reports_path() / "metrics.json"
        save_json(self.results, results_path)

        return model_path
