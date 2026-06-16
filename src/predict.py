"""Prediction module for Mental Health Treatment Prediction."""

import pandas as pd
import numpy as np
from src.utils import setup_logging, load_model, get_model_path

logger = setup_logging()


class MentalHealthPredictor:
    """Makes predictions using the trained model."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = get_model_path() / "best_model.pkl"
        self.model = load_model(model_path)
        self.risk_levels = {
            0: {"label": "Low Risk", "color": "green", "message": "Unlikely to seek treatment"},
            1: {"label": "High Risk", "color": "red", "message": "Likely to seek treatment"},
        }

    def predict(self, features):
        """Make a prediction for a single instance."""
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        result = {
            "prediction": int(prediction),
            "probability_no_treatment": float(probability[0]),
            "probability_treatment": float(probability[1]),
            "risk_level": self.risk_levels[prediction]["label"],
            "message": self.risk_levels[prediction]["message"],
        }

        return result

    def predict_batch(self, df):
        """Make predictions for multiple instances."""
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)

        results = pd.DataFrame({
            "prediction": predictions,
            "probability_no_treatment": probabilities[:, 0],
            "probability_treatment": probabilities[:, 1],
        })

        return results
