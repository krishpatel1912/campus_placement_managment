"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: ml/evaluate.py
Purpose: Generates unified performance metrics comparison tables and exports
         diagnostic visualization plots for all 5 machine learning models.
=============================================================================
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mplcache")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless file saving
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Import preprocessing helpers
from preprocess import (
    load_data, prepare_placement_data, prepare_tier_and_package_data,
    prepare_kmeans_data, KMEANS_FEATURES
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def plot_confusion_matrix(cm, save_path):
    """
    Plots and saves the Confusion Matrix for Logistic Regression.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ["Not Placed", "Placed"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Logistic Regression: Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    
    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, weight="bold")
            
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved Confusion Matrix: {save_path}")

def plot_knn_k_curve(knn_tuning, save_path):
    """
    Plots and saves K vs Cross-Validation Accuracy for K-Nearest Neighbors.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    k_range = knn_tuning["k_range"]
    cv_scores = [s * 100 for s in knn_tuning["cv_scores"]]
    best_k = knn_tuning["best_k"]
    
    ax.plot(k_range, cv_scores, marker="o", color="#2b5c8f", linewidth=2, label="5-Fold CV Accuracy (%)")
    ax.axvline(best_k, color="#e63946", linestyle="--", label=f"Best K = {best_k} ({max(cv_scores):.1f}%)")
    
    ax.set_title("KNN Hyperparameter Tuning: K Value vs Accuracy", fontsize=13, weight="bold")
    ax.set_xlabel("Number of Neighbors (K)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(k_range)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved KNN K-Curve: {save_path}")

def plot_actual_vs_predicted_lpa(model, preprocessor, X_test_tier, y_lpa_test, save_path):
    """
    Plots and saves Actual vs Predicted Salary LPA scatter plot with identity diagonal.
    """
    X_test_trans = preprocessor.transform(X_test_tier)
    y_pred = model.predict(X_test_trans)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_lpa_test, y_pred, alpha=0.55, color="#2a9d8f", edgecolors="k", s=35, label="Student Records")
    
    # 45-degree reference line
    min_val = min(y_lpa_test.min(), y_pred.min())
    max_val = max(y_lpa_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#e76f51", linestyle="--", linewidth=2, label="Ideal Line (y = x)")
    
    ax.set_title("Linear Regression: Actual vs Predicted Package (LPA)", fontsize=13, weight="bold")
    ax.set_xlabel("Actual Package (LPA)", fontsize=11)
    ax.set_ylabel("Predicted Package (LPA)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved Actual vs Predicted LPA Plot: {save_path}")

def plot_kmeans_elbow(kmeans_meta, save_path):
    """
    Plots and saves the K-Means Elbow Curve (Inertia vs Number of Clusters).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    k_range = kmeans_meta["k_range"]
    inertias = kmeans_meta["inertias"]
    
    ax.plot(k_range, inertias, marker="s", color="#8338ec", linewidth=2)
    ax.scatter([3], [inertias[2]], color="#ff006e", s=120, zorder=5, label="Optimal Elbow (K=3)")
    
    ax.set_title("K-Means: Elbow Method for Optimal Clusters (K)", fontsize=13, weight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    ax.set_xticks(k_range)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved K-Means Elbow Curve: {save_path}")

def plot_feature_importance_bar(fi_df, save_path):
    """
    Plots and saves horizontal bar chart of Decision Tree feature importances.
    """
    # Filter only features with non-zero importance
    top_fi = fi_df[fi_df["importance"] > 0.005].sort_values(by="importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top_fi["feature"], top_fi["importance"] * 100, color="#3a86ff", edgecolor="#023e8a")
    
    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                va="center", ha="left", fontsize=10, weight="bold")
        
    ax.set_title("Decision Tree: Feature Importance for Placement", fontsize=13, weight="bold")
    ax.set_xlabel("Importance (%)", fontsize=11)
    ax.set_xlim(0, max(top_fi["importance"] * 100) + 8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved Feature Importance Chart: {save_path}")

def plot_decision_tree_diagram(dt_model, feature_names, save_path):
    """
    Exports clean visual graph of the Decision Tree (depth=3 for crisp rendering).
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3,  # Top 3 levels for clean presentation
        ax=ax
    )
    ax.set_title("Decision Tree Flowchart (Top Decision Rules)", fontsize=16, weight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved Decision Tree Diagram: {save_path}")

def generate_comparison_table():
    """
    Compiles and prints a clean terminal comparison table of all 5 ML algorithms.
    """
    metrics = joblib.load(os.path.join(MODELS_DIR, "all_training_metrics.pkl"))
    knn_meta = joblib.load(os.path.join(MODELS_DIR, "knn_tuning.pkl"))
    
    table_data = [
        {
            "Algorithm": "1. Logistic Regression",
            "Target": "Placement (0 or 1)",
            "Type": "Classification",
            "Key Metric 1": f"Accuracy: {metrics['logistic_regression']['accuracy']*100:.1f}%",
            "Key Metric 2": f"F1-Score: {metrics['logistic_regression']['f1']:.3f}",
            "Purpose in App": "Calculates Placement Probability (%)"
        },
        {
            "Algorithm": "2. K-Nearest Neighbors",
            "Target": "Company Tier (1, 2, 3)",
            "Type": "Classification",
            "Key Metric 1": f"Optimal K: {knn_meta['best_k']}",
            "Key Metric 2": f"Test Acc: {knn_meta['test_acc']*100:.1f}%",
            "Purpose in App": "Predicts Company Tier & 5 Similar Students"
        },
        {
            "Algorithm": "3. Linear Regression",
            "Target": "Package LPA",
            "Type": "Regression",
            "Key Metric 1": f"RMSE: {metrics['linear_regression']['rmse']:.2f} LPA",
            "Key Metric 2": f"R²: {metrics['linear_regression']['r2']:.3f}",
            "Purpose in App": "Predicts Salary Package & ±RMSE Range"
        },
        {
            "Algorithm": "4. K-Means Clustering",
            "Target": "Student Personas (K=3)",
            "Type": "Unsupervised",
            "Key Metric 1": f"Silhouette: {metrics['kmeans']['silhouette_score']:.3f}",
            "Key Metric 2": "Clusters: 3 Personas",
            "Purpose in App": "Peer Benchmarking & Radar Comparison"
        },
        {
            "Algorithm": "5. Decision Tree",
            "Target": "Placement (0 or 1)",
            "Type": "Explainable Tree",
            "Key Metric 1": f"Accuracy: {metrics['decision_tree']['accuracy']*100:.1f}%",
            "Key Metric 2": "Max Depth: 4",
            "Purpose in App": "Feature Importance & Personalized Advice"
        }
    ]
    df_table = pd.DataFrame(table_data)
    
    print("\n" + "=" * 95)
    print("                 CAMPUS PLACEMENT PREDICTOR - MODEL COMPARISON TABLE               ")
    print("=" * 95)
    print(df_table.to_string(index=False))
    print("=" * 95 + "\n")
    
    # Save comparison table for Streamlit app
    joblib.dump(df_table, os.path.join(MODELS_DIR, "model_comparison_table.pkl"))
    return df_table

def main():
    print("=============================================================")
    print("   GENERATING MODEL EVALUATION METRICS & VISUAL CHARTS       ")
    print("=============================================================")
    
    # Load dataset and fitted components
    df = load_data()
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    all_metrics = joblib.load(os.path.join(MODELS_DIR, "all_training_metrics.pkl"))
    knn_tuning = joblib.load(os.path.join(MODELS_DIR, "knn_tuning.pkl"))
    kmeans_meta = joblib.load(os.path.join(MODELS_DIR, "kmeans_metadata.pkl"))
    fi_df = joblib.load(os.path.join(MODELS_DIR, "feature_importances.pkl"))
    dt_model = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
    linreg_model = joblib.load(os.path.join(MODELS_DIR, "linear_reg_model.pkl"))
    
    # 1. Confusion Matrix
    cm = all_metrics["logistic_regression"]["confusion_matrix"]
    plot_confusion_matrix(cm, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    
    # 2. KNN K-Curve
    plot_knn_k_curve(knn_tuning, os.path.join(REPORTS_DIR, "knn_k_accuracy.png"))
    
    # 3. Linear Regression Actual vs Predicted
    _, X_test_tier, _, _, _, y_lpa_test = prepare_tier_and_package_data(df)
    plot_actual_vs_predicted_lpa(linreg_model, preprocessor, X_test_tier, y_lpa_test,
                                os.path.join(REPORTS_DIR, "lpa_actual_vs_predicted.png"))
    
    # 4. K-Means Elbow Curve
    plot_kmeans_elbow(kmeans_meta, os.path.join(REPORTS_DIR, "kmeans_elbow.png"))
    
    # 5. Decision Tree Feature Importance Bar Chart
    plot_feature_importance_bar(fi_df, os.path.join(REPORTS_DIR, "feature_importance.png"))
    
    # 6. Decision Tree Diagram
    plot_decision_tree_diagram(dt_model, feature_names, os.path.join(REPORTS_DIR, "decision_tree.png"))
    
    # 7. Print and Save Comparison Table
    generate_comparison_table()
    
    print("All evaluation plots generated and saved to 'reports/' successfully!\n")

if __name__ == "__main__":
    main()
