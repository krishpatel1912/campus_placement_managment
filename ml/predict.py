"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: ml/predict.py
Purpose: High-level inference engine that runs a student profile through all
         5 machine learning models and produces unified, actionable results.
=============================================================================
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

import joblib
import numpy as np
import pandas as pd

# Load artifacts lazily or cached
_models_cache = {}

def get_models():
    """
    Loads all serialized models, preprocessors, and metadata from disk into memory.
    Caches in memory for rapid repeated inference.
    """
    if "loaded" not in _models_cache:
        _models_cache["preprocessor"] = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        _models_cache["feature_names"] = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        _models_cache["logistic"] = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))
        _models_cache["knn"] = joblib.load(os.path.join(MODELS_DIR, "knn_tier_model.pkl"))
        _models_cache["knn_peers"] = joblib.load(os.path.join(MODELS_DIR, "knn_peer_database.pkl"))
        _models_cache["linreg"] = joblib.load(os.path.join(MODELS_DIR, "linear_reg_model.pkl"))
        _models_cache["linreg_metrics"] = joblib.load(os.path.join(MODELS_DIR, "linear_reg_metrics.pkl"))
        _models_cache["kmeans"] = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
        _models_cache["kmeans_scaler"] = joblib.load(os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
        _models_cache["kmeans_meta"] = joblib.load(os.path.join(MODELS_DIR, "kmeans_metadata.pkl"))
        _models_cache["dt"] = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
        _models_cache["dt_fi"] = joblib.load(os.path.join(MODELS_DIR, "feature_importances.pkl"))
        _models_cache["loaded"] = True
    return _models_cache

def format_student_df(student_dict: dict) -> pd.DataFrame:
    """
    Converts a single student profile dictionary into a standardized DataFrame
    matching the column format expected by the preprocessor.
    """
    expected_cols = [
        "tenth_percent", "twelfth_percent", "cgpa", "backlogs",
        "branch", "experience_type", "num_projects", "num_certifications",
        "coding_skill", "communication_skill", "aptitude_score", "skills_count"
    ]
    data = {col: [student_dict.get(col, 0)] for col in expected_cols}
    return pd.DataFrame(data)

def predict_placement_probability(student_dict: dict) -> dict:
    """
    ALGORITHM 1: Logistic Regression
    Returns the placement chance percentage and binary prediction.
    """
    models = get_models()
    df_student = format_student_df(student_dict)
    X_trans = models["preprocessor"].transform(df_student)
    
    prob = models["logistic"].predict_proba(X_trans)[0, 1] * 100.0
    pred = int(models["logistic"].predict(X_trans)[0])
    
    return {
        "placement_chance_pct": round(float(prob), 1),
        "is_placed": bool(pred == 1)
    }

def predict_tier_and_similar_students(student_dict: dict, n_neighbors: int = 5) -> dict:
    """
    ALGORITHM 2: K-Nearest Neighbors
    Predicts the best fit company tier (1, 2, or 3) and retrieves the
    5 closest historical placed students from the database.
    """
    models = get_models()
    df_student = format_student_df(student_dict)
    X_trans = models["preprocessor"].transform(df_student)
    
    pred_tier_int = int(models["knn"].predict(X_trans)[0])
    
    tier_labels = {
        1: "Tier 3 (Service / Mass Recruiters)",
        2: "Tier 2 (Mid-size IT / Product Consulting)",
        3: "Tier 1 (Product / Top Tech / High Package)"
    }
    
    # Query nearest neighbors
    distances, indices = models["knn"].kneighbors(X_trans, n_neighbors=n_neighbors)
    peer_db = models["knn_peers"]["raw_placed_students"]
    
    similar_students = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < len(peer_db):
            row = peer_db.iloc[idx]
            tier_val = int(row.get("company_tier", pred_tier_int))
            similar_students.append({
                "Rank": f"#{rank}",
                "Branch": str(row.get("branch", "N/A")),
                "CGPA": float(row.get("cgpa", 0.0)),
                "Coding (1-10)": int(row.get("coding_skill", 0)),
                "Aptitude": float(row.get("aptitude_score", 0.0)),
                "Projects": int(row.get("num_projects", 0)),
                "Tier Placed": f"Tier {tier_val}",
                "Package (LPA)": f"{float(row.get('package_lpa', 0.0)):.1f}",
                "Similarity Distance": round(float(dist), 2)
            })
            
    return {
        "tier_code": pred_tier_int,
        "tier_name": tier_labels.get(pred_tier_int, f"Tier {pred_tier_int}"),
        "similar_students": pd.DataFrame(similar_students)
    }

def predict_salary_package(student_dict: dict) -> dict:
    """
    ALGORITHM 3: Linear Regression
    Predicts expected CTC in LPA along with an uncertainty range based on RMSE.
    """
    models = get_models()
    df_student = format_student_df(student_dict)
    X_trans = models["preprocessor"].transform(df_student)
    
    raw_lpa = float(models["linreg"].predict(X_trans)[0])
    rmse = float(models["linreg_metrics"]["rmse"])
    
    # Ensure package is non-negative and bounded realistically
    predicted_lpa = max(3.0, round(raw_lpa, 2))
    min_range = max(3.0, round(predicted_lpa - rmse, 2))
    max_range = round(predicted_lpa + rmse, 2)
    
    return {
        "predicted_lpa": predicted_lpa,
        "rmse": round(rmse, 2),
        "range_min": min_range,
        "range_max": max_range,
        "range_str": f"{min_range:.1f} - {max_range:.1f} LPA"
    }

def predict_cluster_benchmarking(student_dict: dict) -> dict:
    """
    ALGORITHM 4: K-Means Clustering
    Identifies the student's peer cluster persona and calculates radar comparison metrics.
    """
    models = get_models()
    cluster_features = models["kmeans_meta"]["features"]
    
    # Extract the 5 cluster dimensions as a DataFrame with proper column names
    student_df_cluster = pd.DataFrame([{f: student_dict.get(f, 0) for f in cluster_features}])
    
    # Scale with K-Means specific scaler
    scaled_vec = models["kmeans_scaler"].transform(student_df_cluster)
    cluster_id = int(models["kmeans"].predict(scaled_vec)[0])
    
    label_map = models["kmeans_meta"]["label_map"]
    cluster_persona = label_map.get(cluster_id, "Average")
    
    # Retrieve cluster average stats
    profiles = models["kmeans_meta"]["cluster_profiles"]
    cluster_stats = profiles.get(cluster_id, {})
    
    # Prepare comparison data for radar visualization
    dimensions = ["CGPA", "Coding Skill", "Aptitude (/10)", "Projects", "Skills Count"]
    student_radar_vals = [
        round(float(student_dict.get("cgpa", 0)), 1),
        round(float(student_dict.get("coding_skill", 0)), 1),
        round(float(student_dict.get("aptitude_score", 0)) / 10.0, 1),
        round(float(student_dict.get("num_projects", 0)), 1),
        round(float(student_dict.get("skills_count", 0)), 1)
    ]
    cluster_radar_vals = [
        round(float(cluster_stats.get("mean_cgpa", 7.0)), 1),
        round(float(cluster_stats.get("mean_coding", 5.0)), 1),
        round(float(cluster_stats.get("mean_aptitude", 60.0)) / 10.0, 1),
        round(float(cluster_stats.get("mean_projects", 2.0)), 1),
        round(float(cluster_stats.get("mean_skills", 6.0)), 1)
    ]
    
    return {
        "cluster_id": cluster_id,
        "cluster_persona": cluster_persona,
        "radar_dimensions": dimensions,
        "student_values": student_radar_vals,
        "cluster_averages": cluster_radar_vals
    }

def explain_student_placement(student_dict: dict) -> dict:
    """
    ALGORITHM 5: Decision Tree Explainability
    Evaluates individual student metrics against key drivers to generate actionable advice.
    """
    models = get_models()
    fi_df = models["dt_fi"]
    
    # Generate tailored improvement tips based on student's actual values
    tips = []
    
    # 1. Backlogs (Dominant factor)
    backlogs = student_dict.get("backlogs", 0)
    if backlogs > 0:
        tips.append({
            "factor": "Active Backlogs",
            "priority": "HIGH",
            "observation": f"You currently have {backlogs} active backlog(s).",
            "action": "Clear all pending subjects as active backlogs account for ~68% of rejection criteria across campus recruiters."
        })
    else:
        tips.append({
            "factor": "Active Backlogs",
            "priority": "GOOD",
            "observation": "Zero backlogs.",
            "action": "Maintain clean academic standing to stay eligible for Day-1 companies."
        })
        
    # 2. Coding Skill
    coding = student_dict.get("coding_skill", 1)
    if coding < 6:
        tips.append({
            "factor": "Coding & DSA",
            "priority": "HIGH",
            "observation": f"Coding skill self-rating is {coding}/10.",
            "action": "Practice Data Structures & Algorithms on LeetCode/HackerRank (target 100+ easy/medium problems) to raise your rating to 7+."
        })
    elif coding < 8:
        tips.append({
            "factor": "Coding & DSA",
            "priority": "MEDIUM",
            "observation": f"Solid coding foundation ({coding}/10).",
            "action": "Focus on Graph algorithms, Dynamic Programming, and System Design to crack Tier-1 coding rounds."
        })
    else:
        tips.append({
            "factor": "Coding & DSA",
            "priority": "GOOD",
            "observation": f"Excellent coding rating ({coding}/10).",
            "action": "Top-tier strength! Pair this with live system projects."
        })
        
    # 3. CGPA
    cgpa = student_dict.get("cgpa", 6.0)
    if cgpa < 7.0:
        tips.append({
            "factor": "College CGPA",
            "priority": "HIGH",
            "observation": f"Current CGPA is {cgpa:.2f}.",
            "action": "Aim to raise your semester CGPA above 7.0 to unlock 80%+ of campus eligibility shortlists."
        })
    elif cgpa < 8.0:
        tips.append({
            "factor": "College CGPA",
            "priority": "MEDIUM",
            "observation": f"Good CGPA ({cgpa:.2f}).",
            "action": "Maintain above 7.5 to remain competitive for Tier-1 product roles."
        })
        
    # 4. Aptitude Score
    aptitude = student_dict.get("aptitude_score", 50)
    if aptitude < 65:
        tips.append({
            "factor": "Aptitude Round",
            "priority": "MEDIUM",
            "observation": f"Aptitude score is {aptitude:.1f}/100.",
            "action": "Solve daily quantitative, logical reasoning, and verbal mock tests (IndiaBIX / PrepInsta) to cross 75+."
        })
        
    # 5. Projects
    projects = student_dict.get("num_projects", 0)
    if projects < 2:
        tips.append({
            "factor": "Projects & Portfolio",
            "priority": "HIGH",
            "observation": f"Only {projects} completed project(s).",
            "action": "Build at least 2 full-stack or ML end-to-end applications with public GitHub repositories and live deployments."
        })
        
    return {
        "top_features": fi_df.head(5).to_dict(orient="records"),
        "improvement_tips": tips
    }

def predict_all(student_dict: dict) -> dict:
    """
    Unified entry point: Evaluates a student through all 5 algorithms simultaneously.
    """
    placement = predict_placement_probability(student_dict)
    tier_peers = predict_tier_and_similar_students(student_dict)
    salary = predict_salary_package(student_dict)
    cluster = predict_cluster_benchmarking(student_dict)
    explanation = explain_student_placement(student_dict)
    
    return {
        "placement": placement,
        "tier_and_peers": tier_peers,
        "salary": salary,
        "cluster": cluster,
        "explanation": explanation
    }
