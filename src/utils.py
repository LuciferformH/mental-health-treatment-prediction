"""Utility functions for the Mental Health Treatment Prediction project."""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_data_path(subfolder="raw"):
    """Return path to data subfolder."""
    return get_project_root() / "data" / subfolder


def get_model_path():
    """Return path to models directory."""
    return get_project_root() / "models"


def get_reports_path():
    """Return path to reports directory."""
    return get_project_root() / "reports"


def load_csv(filepath):
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")


def save_csv(df, filepath):
    """Save a DataFrame to CSV."""
    try:
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        raise Exception(f"Error saving CSV: {e}")


def save_model(model, filepath):
    """Save a model using joblib."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        return True
    except Exception as e:
        raise Exception(f"Error saving model: {e}")


def load_model(filepath):
    """Load a model using joblib."""
    try:
        return joblib.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def save_json(data, filepath):
    """Save a dictionary to JSON."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4, default=str)
        return True
    except Exception as e:
        raise Exception(f"Error saving JSON: {e}")


def load_json(filepath):
    """Load a JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading JSON: {e}")


def print_section_header(title):
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_model_comparison(results):
    """Print a formatted model comparison table."""
    print_section_header("MODEL COMPARISON RESULTS")
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}"
    print(header)
    print("-" * 65)

    best_f1 = 0
    best_model_name = ""

    for model_name, metrics in results.items():
        row = (
            f"{model_name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f}"
        )
        print(row)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name

    print("-" * 65)
    print(f"\nBest Model: {best_model_name} (F1 Score: {best_f1:.4f})")

    return best_model_name
