"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: ml/preprocess.py
Purpose: Feature encoding, standardization, and dataset splitting pipeline.
=============================================================================

NOTE FOR STUDENTS & EXAMINERS:
Machine learning models cannot directly process text categories like "CSE" or
"Internship". This module performs two essential steps:
1. One-Hot Encoding: Converts categories into binary (0/1) indicator flags.
2. Standard Scaling: Normalizes all numeric values to have Mean = 0 and StdDev = 1,
   preventing features with large ranges (like aptitude 0-100) from dominating
   features with small ranges (like CGPA 5-10 or backlogs 0-5).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define feature columns used by the prediction models
NUMERIC_FEATURES = [
    "tenth_percent",
    "twelfth_percent",
    "cgpa",
    "backlogs",
    "num_projects",
    "num_certifications",
    "coding_skill",
    "communication_skill",
    "aptitude_score",
    "skills_count"
]

CATEGORICAL_FEATURES = [
    "branch",
    "experience_type"
]

# Specifically selected 5 intuitive dimensions for unsupervised K-Means peer clustering
KMEANS_FEATURES = [
    "cgpa",
    "coding_skill",
    "aptitude_score",
    "num_projects",
    "skills_count"
]

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads the student placement records from data/students.csv.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "students.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run data/generate_dataset.py first.")
    return pd.read_csv(filepath)

def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer that scales numeric features
    and one-hot encodes categorical branches and experience levels.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def prepare_placement_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the full dataset into train/test sets for Placement Prediction (Logistic Regression & Decision Tree).
    Target: 'placed' (0 = Not Placed, 1 = Placed).
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["placed"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def prepare_tier_and_package_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Filters for placed students only (placed == 1) to train:
    - KNN (Target: 'company_tier' in {1, 2, 3})
    - Linear Regression (Target: 'package_lpa' > 0)
    """
    placed_df = df[df["placed"] == 1].copy()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = placed_df[feature_cols].copy()
    y_tier = placed_df["company_tier"].copy()
    y_lpa = placed_df["package_lpa"].copy()

    X_train, X_test, y_tier_train, y_tier_test, y_lpa_train, y_lpa_test = train_test_split(
        X, y_tier, y_lpa, test_size=test_size, random_state=random_state, stratify=y_tier
    )
    return X_train, X_test, y_tier_train, y_tier_test, y_lpa_train, y_lpa_test

def prepare_kmeans_data(df: pd.DataFrame):
    """
    Extracts the 5 selected numeric dimensions for K-Means unsupervised clustering.
    Returns the raw slice and an independent StandardScaler fitted on these features.
    """
    X_cluster = df[KMEANS_FEATURES].copy()
    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_cluster)
    return X_cluster, X_scaled, cluster_scaler

def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extracts human-readable feature column names from the fitted preprocessor.
    """
    feature_names = []
    # Numeric features retain original names
    feature_names.extend(NUMERIC_FEATURES)
    # One-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cats = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(encoded_cats))
    return feature_names
