"""Model evaluation module for Mental Health Treatment Prediction."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

from src.utils import setup_logging, get_reports_path

logger = setup_logging()


class ModelEvaluator:
    """Evaluates trained models and generates reports."""

    def __init__(self):
        self.reports_path = get_reports_path()
        os.makedirs(self.reports_path, exist_ok=True)

    def evaluate(self, model, X_test, y_test):
        """Calculate all evaluation metrics."""
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        return metrics

    def generate_confusion_matrix(self, model, X_test, y_test, model_name="model"):
        """Generate and save confusion matrix plot."""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Treatment", "Treatment"],
            yticklabels=["No Treatment", "Treatment"],
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14)
        plt.ylabel("Actual", fontsize=12)
        plt.xlabel("Predicted", fontsize=12)
        plt.tight_layout()

        filepath = self.reports_path / "confusion_matrix.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved to {filepath}")

        return filepath

    def generate_roc_curve(self, model, X_test, y_test, model_name="model"):
        """Generate and save ROC curve plot."""
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_test)
        )

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"ROC Curve - {model_name}", fontsize=14)
        plt.legend(loc="lower right")
        plt.tight_layout()

        filepath = self.reports_path / "roc_curve.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curve saved to {filepath}")

        return filepath

    def generate_feature_importance(self, model, feature_names, model_name="model"):
        """Generate and save feature importance plot."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"{model_name} does not support feature importance")
            return None

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(feat_imp["feature"], feat_imp["importance"], color="steelblue")
        plt.xlabel("Importance", fontsize=12)
        plt.title(f"Feature Importance - {model_name}", fontsize=14)
        plt.tight_layout()

        filepath = self.reports_path / "feature_importance.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance saved to {filepath}")

        return filepath

    def generate_classification_report(self, model, X_test, y_test):
        """Generate classification report."""
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=["No Treatment", "Treatment"],
        )
        logger.info(f"\nClassification Report:\n{report}")
        return report
