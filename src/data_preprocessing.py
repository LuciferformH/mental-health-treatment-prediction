"""Data preprocessing module for Mental Health Treatment Prediction."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils import setup_logging, load_csv, save_csv, get_data_path, get_project_root

logger = setup_logging()


class DataPreprocessor:
    """Handles all data preprocessing tasks."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = "treatment"

    def load_data(self, filepath=None):
        """Load the raw dataset."""
        if filepath is None:
            filepath = get_data_path("raw") / "mental_health_survey.csv"
        df = load_csv(filepath)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def initial_clean(self, df):
        """Perform initial cleaning: drop irrelevant columns, handle NAs."""
        df = df.copy()

        columns_to_drop = ["Timestamp", "comments", "state"]
        existing_cols_to_drop = [c for c in columns_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols_to_drop)

        df = df.replace("NA", np.nan)
        df = df.replace("Don't know", np.nan)

        logger.info(f"After initial clean: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        df = df.copy()

        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        logger.info(f"Missing values remaining: {df.isnull().sum().sum()}")
        return df

    def remove_duplicates(self, df):
        """Remove duplicate rows."""
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed = initial_shape - df.shape[0]
        logger.info(f"Removed {removed} duplicate rows")
        return df

    def handle_age_outliers(self, df):
        """Handle age outliers by clipping to reasonable range."""
        df = df.copy()
        if "Age" in df.columns:
            df["Age"] = df["Age"].clip(lower=18, upper=75)
        return df

    def standardize_gender(self, df):
        """Standardize gender values."""
        df = df.copy()
        if "Gender" in df.columns:
            gender_map = {
                "male": "Male", "m": "Male", "cis male": "Male",
                "cis man": "Male", "man": "Male",
                "female": "Female", "f": "Female", "cis female": "Female",
                "cis-female": "Female", "woman": "Female",
            }
            df["Gender"] = df["Gender"].str.strip().str.lower()
            df["Gender"] = df["Gender"].map(gender_map).fillna(df["Gender"])
        return df

    def encode_categorical(self, df):
        """Encode categorical variables using Label Encoding."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if col != self.target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        if self.target_column in df.columns:
            target_le = LabelEncoder()
            df[self.target_column] = target_le.fit_transform(
                df[self.target_column].astype(str)
            )
            self.label_encoders[self.target_column] = target_le

        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df

    def scale_features(self, X_train, X_test):
        """Scale numerical features using StandardScaler."""
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_cols] = self.scaler.fit_transform(
            X_train[numerical_cols]
        )
        X_test_scaled[numerical_cols] = self.scaler.transform(
            X_test[numerical_cols]
        )

        return X_train_scaled, X_test_scaled

    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Features: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    def full_preprocess(self, filepath=None):
        """Run the complete preprocessing pipeline."""
        df = self.load_data(filepath)
        df = self.initial_clean(df)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.handle_age_outliers(df)
        df = self.standardize_gender(df)
        df = self.encode_categorical(df)

        processed_path = get_data_path("processed") / "processed_data.csv"
        save_csv(df, processed_path)
        logger.info(f"Processed data saved to {processed_path}")

        return df
