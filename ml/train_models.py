"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: ml/train_models.py
Purpose: Trains, tunes, evaluates, and saves 5 fundamental ML algorithms:
         1. Logistic Regression (Placement Probability)
         2. K-Nearest Neighbors (Company Tier & Peer Matching)
         3. Linear Regression (Expected Salary LPA & Range)
         4. K-Means Clustering (Student Peer Segmentation)
         5. Decision Tree (Explainability & Feature Importance)
=============================================================================
"""

import os

# Ensure Matplotlib uses a writable local cache within project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mplcache")

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)

# Import shared preprocessing pipeline
from preprocess import (
    load_data, create_preprocessor, prepare_placement_data,
    prepare_tier_and_package_data, prepare_kmeans_data,
    get_transformed_feature_names, KMEANS_FEATURES
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =============================================================================
# ALGORITHM 1: LOGISTIC REGRESSION (Binary Classification)
# =============================================================================
def train_logistic_regression(X_train_trans, X_test_trans, y_train, y_test):
    """
    Logistic Regression predicts the probability of placement (0 to 100%)
    by fitting a sigmoid curve over the weighted sum of student features.
    """
    print("\n-------------------------------------------------------------")
    print(" [1/5] Training Logistic Regression (Placement Probability)...")
    print("-------------------------------------------------------------")
    
    # max_iter=1000 ensures complete gradient convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_trans, y_train)
    
    # Predict binary outcomes and probability scores
    y_pred = model.predict(X_test_trans)
    y_proba = model.predict_proba(X_test_trans)[:, 1]
    
    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f" -> Accuracy : {acc*100:.2f}%")
    print(f" -> Precision: {prec*100:.2f}%")
    print(f" -> Recall   : {rec*100:.2f}%")
    print(f" -> F1-Score : {f1:.4f}")
    print(f" -> Confusion Matrix:\n{cm}")
    
    # Save trained model to disk
    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_model.pkl"))
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }
    return model, metrics


# =============================================================================
# ALGORITHM 2: K-NEAREST NEIGHBORS (Multi-Class Classification & Peer Matching)
# =============================================================================
def train_knn(X_train_trans, X_test_trans, y_train, y_test, raw_placed_train_df):
    """
    KNN classifies company tier (1, 2, or 3) by measuring Euclidean distance
    to find the K most similar past students and taking a majority vote.
    """
    print("\n-------------------------------------------------------------")
    print(" [2/5] Training K-Nearest Neighbors (Company Tier & Peer Match)...")
    print("-------------------------------------------------------------")
    
    # Find best K between 1 and 20 using 5-fold cross-validation
    k_range = list(range(1, 21))
    cv_scores = []
    
    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_temp, X_train_trans, y_train, cv=5, scoring="accuracy")
        cv_scores.append(scores.mean())
        
    best_k = k_range[int(np.argmax(cv_scores))]
    best_score = max(cv_scores)
    print(f" -> Optimal K found: K = {best_k} with 5-fold CV Accuracy: {best_score*100:.2f}%")
    
    # Fit final model with best K
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_trans, y_train)
    
    # Evaluate on held-out test set
    y_pred = model.predict(X_test_trans)
    test_acc = accuracy_score(y_test, y_pred)
    print(f" -> Test Set Accuracy: {test_acc*100:.2f}%")
    
    # Save trained model and tuning history
    joblib.dump(model, os.path.join(MODELS_DIR, "knn_tier_model.pkl"))
    joblib.dump({
        "best_k": best_k,
        "k_range": k_range,
        "cv_scores": cv_scores,
        "test_acc": test_acc
    }, os.path.join(MODELS_DIR, "knn_tuning.pkl"))
    
    # Save the training set features and original student rows for neighbor lookups
    joblib.dump({
        "X_train_trans": X_train_trans,
        "raw_placed_students": raw_placed_train_df.reset_index(drop=True)
    }, os.path.join(MODELS_DIR, "knn_peer_database.pkl"))
    
    metrics = {
        "best_k": best_k,
        "cv_accuracy": best_score,
        "test_accuracy": test_acc
    }
    return model, metrics


# =============================================================================
# ALGORITHM 3: LINEAR REGRESSION (Continuous Value Prediction)
# =============================================================================
def train_linear_regression(X_train_trans, X_test_trans, y_train, y_test):
    """
    Linear Regression predicts the expected salary package (in LPA)
    by learning the optimal linear slope for each student credential.
    """
    print("\n-------------------------------------------------------------")
    print(" [3/5] Training Linear Regression (Salary Package LPA)...")
    print("-------------------------------------------------------------")
    
    model = LinearRegression()
    model.fit(X_train_trans, y_train)
    
    y_pred = model.predict(X_test_trans)
    
    # Compute error metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f" -> Mean Absolute Error (MAE): {mae:.2f} LPA")
    print(f" -> Root Mean Squared Error (RMSE): {rmse:.2f} LPA")
    print(f" -> R-squared (R2 Score): {r2:.4f}")
    
    joblib.dump(model, os.path.join(MODELS_DIR, "linear_reg_model.pkl"))
    joblib.dump({"rmse": rmse, "mae": mae, "r2": r2}, os.path.join(MODELS_DIR, "linear_reg_metrics.pkl"))
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    return model, metrics


# =============================================================================
# ALGORITHM 4: K-MEANS CLUSTERING (Unsupervised Segmentation)
# =============================================================================
def train_kmeans(df):
    """
    K-Means groups students into 3 natural peer clusters without labels,
    identifying Top Performers, Average Students, and Needs Improvement.
    """
    print("\n-------------------------------------------------------------")
    print(" [4/5] Training K-Means Clustering (Student Peer Segmentation)...")
    print("-------------------------------------------------------------")
    
    X_cluster, X_scaled, cluster_scaler = prepare_kmeans_data(df)
    
    # Calculate inertia for K from 1 to 8 (for the Elbow Method plot)
    k_range = list(range(1, 9))
    inertias = []
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_temp.fit(X_scaled)
        inertias.append(km_temp.inertia_)
        
    # Fit target K = 3
    k_chosen = 3
    model = KMeans(n_clusters=k_chosen, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f" -> Silhouette Score (K=3): {sil_score:.4f}")
    
    # Compute average profile of each cluster in original scale to label personas
    cluster_profiles = {}
    for c_id in range(k_chosen):
        cluster_rows = X_cluster[cluster_labels == c_id]
        cluster_profiles[c_id] = {
            "mean_cgpa": cluster_rows["cgpa"].mean(),
            "mean_coding": cluster_rows["coding_skill"].mean(),
            "mean_aptitude": cluster_rows["aptitude_score"].mean(),
            "mean_projects": cluster_rows["num_projects"].mean(),
            "mean_skills": cluster_rows["skills_count"].mean(),
            "count": len(cluster_rows)
        }
    
    # Rank clusters by a composite score (coding + cgpa*1.2 + aptitude/10)
    def composite_rank(stats):
        return stats["mean_coding"] * 2.0 + stats["mean_cgpa"] * 1.5 + (stats["mean_aptitude"] / 10.0)
    
    sorted_cluster_ids = sorted(cluster_profiles.keys(), key=lambda c: composite_rank(cluster_profiles[c]), reverse=True)
    
    # Assign human-readable pedagogical persona labels
    label_map = {
        sorted_cluster_ids[0]: "Top Performers",
        sorted_cluster_ids[1]: "Average",
        sorted_cluster_ids[2]: "Needs Improvement"
    }
    
    print(" -> Identified Student Personas:")
    for c_id, name in label_map.items():
        stats = cluster_profiles[c_id]
        print(f"    Cluster {c_id} ({name}): CGPA={stats['mean_cgpa']:.2f}, Coding={stats['mean_coding']:.1f}/10, "
              f"Aptitude={stats['mean_aptitude']:.1f}/100, Students={stats['count']}")
        
    # Save artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    joblib.dump(cluster_scaler, os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
    joblib.dump({
        "label_map": label_map,
        "cluster_profiles": cluster_profiles,
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_score": sil_score,
        "features": KMEANS_FEATURES
    }, os.path.join(MODELS_DIR, "kmeans_metadata.pkl"))
    
    metrics = {
        "silhouette_score": sil_score,
        "cluster_profiles": cluster_profiles,
        "label_map": label_map
    }
    return model, metrics


# =============================================================================
# ALGORITHM 5: DECISION TREE CLASSIFIER (Explainability & Feature Importance)
# =============================================================================
def train_decision_tree(X_train_trans, X_test_trans, y_train, y_test, feature_names):
    """
    Decision Tree creates an interpretable hierarchy of if-else threshold rules
    to clearly show students which skills have the highest impact on hiring.
    """
    print("\n-------------------------------------------------------------")
    print(" [5/5] Training Decision Tree (Explainability & Guidance)...")
    print("-------------------------------------------------------------")
    
    # max_depth=4 keeps the tree transparent and easy for students to explain
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train_trans, y_train)
    
    y_pred = model.predict(X_test_trans)
    acc = accuracy_score(y_test, y_pred)
    print(f" -> Decision Tree Accuracy (max_depth=4): {acc*100:.2f}%")
    
    # Calculate feature importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    print(" -> Top 5 Most Influential Placement Factors:")
    for _, row in fi_df.head(5).iterrows():
        print(f"    - {row['feature']}: {row['importance']*100:.1f}%")
        
    # Save model and feature importances
    joblib.dump(model, os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
    joblib.dump(fi_df, os.path.join(MODELS_DIR, "feature_importances.pkl"))
    
    metrics = {
        "accuracy": acc,
        "feature_importances": fi_df
    }
    return model, metrics


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    print("=============================================================")
    print("   CAMPUS PLACEMENT PREDICTOR - MACHINE LEARNING PIPELINE    ")
    print("=============================================================")
    
    # 1. Load Student Dataset
    df = load_data()
    print(f"Loaded dataset: {len(df)} student records.")
    
    # 2. Fit and Save Shared ColumnTransformer
    print("\nFitting preprocessing pipeline (StandardScaler + OneHotEncoder)...")
    preprocessor = create_preprocessor()
    
    # Prepare placement training sets (all students)
    X_train_place, X_test_place, y_train_place, y_test_place = prepare_placement_data(df)
    
    # Fit preprocessor strictly on X_train to prevent data leakage
    X_train_place_trans = preprocessor.fit_transform(X_train_place)
    X_test_place_trans = preprocessor.transform(X_test_place)
    feature_names = get_transformed_feature_names(preprocessor)
    
    # Save fitted preprocessor and feature schema
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    print(f"Preprocessed into {len(feature_names)} numerical & encoded features.")
    
    # Prepare tier & package datasets (placed students only)
    X_train_tier, X_test_tier, y_tier_train, y_tier_test, y_lpa_train, y_lpa_test = prepare_tier_and_package_data(df)
    X_train_tier_trans = preprocessor.transform(X_train_tier)
    X_test_tier_trans = preprocessor.transform(X_test_tier)
    
    # Dictionary to aggregate metrics for the evaluate.py and Streamlit UI
    all_metrics = {}
    
    # Run Algorithm 1: Logistic Regression
    _, lr_metrics = train_logistic_regression(
        X_train_place_trans, X_test_place_trans, y_train_place, y_test_place
    )
    all_metrics["logistic_regression"] = lr_metrics
    
    # Run Algorithm 2: K-Nearest Neighbors
    _, knn_metrics = train_knn(
        X_train_tier_trans, X_test_tier_trans, y_tier_train, y_tier_test, X_train_tier
    )
    all_metrics["knn"] = knn_metrics
    
    # Run Algorithm 3: Linear Regression
    _, linreg_metrics = train_linear_regression(
        X_train_tier_trans, X_test_tier_trans, y_lpa_train, y_lpa_test
    )
    all_metrics["linear_regression"] = linreg_metrics
    
    # Run Algorithm 4: K-Means Clustering
    _, kmeans_metrics = train_kmeans(df)
    all_metrics["kmeans"] = kmeans_metrics
    
    # Run Algorithm 5: Decision Tree
    _, dt_metrics = train_decision_tree(
        X_train_place_trans, X_test_place_trans, y_train_place, y_test_place, feature_names
    )
    all_metrics["decision_tree"] = dt_metrics
    
    # Save all metrics in a unified file
    joblib.dump(all_metrics, os.path.join(MODELS_DIR, "all_training_metrics.pkl"))
    
    print("\n=============================================================")
    print(" ALL 5 MODELS TRAINED, EVALUATED, AND SAVED SUCCESSFULLY!   ")
    print(f" Models directory: {MODELS_DIR}")
    print("=============================================================")

if __name__ == "__main__":
    main()
