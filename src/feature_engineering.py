"""Feature engineering module for Mental Health Treatment Prediction."""

import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()


class FeatureEngineer:
    """Creates meaningful features from the dataset."""

    def __init__(self):
        self.created_features = []

    def create_features(self, df):
        """Create all engineered features."""
        df = df.copy()
        df = self._work_environment_score(df)
        df = self._support_score(df)
        df = self._health_risk_score(df)
        df = self._age_group(df)
        df = self._interaction_features(df)

        logger.info(f"Created {len(self.created_features)} new features")
        return df

    def _work_environment_score(self, df):
        """Create a work environment support score."""
        cols = ["benefits", "care_options", "wellness_program", "seek_help"]
        available = [c for c in cols if c in df.columns]
        if available:
            df["work_environment_score"] = df[available].sum(axis=1)
            self.created_features.append("work_environment_score")
        return df

    def _support_score(self, df):
        """Create a support system score."""
        cols = ["coworkers", "supervisor"]
        available = [c for c in cols if c in df.columns]
        if available:
            df["support_score"] = df[available].sum(axis=1)
            self.created_features.append("support_score")
        return df

    def _health_risk_score(self, df):
        """Create a health risk score."""
        cols = ["mental_health_consequence", "phys_health_consequence"]
        available = [c for c in cols if c in df.columns]
        if available:
            df["health_risk_score"] = df[available].sum(axis=1)
            self.created_features.append("health_risk_score")
        return df

    def _age_group(self, df):
        """Create age group categories."""
        if "Age" in df.columns:
            df["age_group"] = pd.cut(
                df["Age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=[0, 1, 2, 3, 4],
            ).astype(int)
            self.created_features.append("age_group")
        return df

    def _interaction_features(self, df):
        """Create interaction features between key columns."""
        if "family_history" in df.columns and "work_interfere" in df.columns:
            df["family_work_interaction"] = df["family_history"] * df["work_interfere"]
            self.created_features.append("family_work_interaction")

        if "benefits" in df.columns and "care_options" in df.columns:
            df["benefits_care_interaction"] = df["benefits"] * df["care_options"]
            self.created_features.append("benefits_care_interaction")

        return df
