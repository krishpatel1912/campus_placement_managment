# Project Code & File Breakdown
## Quick Guide to Explain the Project to Your Teacher / Examiner

Use this sheet to quickly answer: **"Which file contains what code, and what does each file do?"**

---

## 1. Quick 1-Minute Pitch to Mam

> *"Mam, our project is a **Campus Placement Predictor and Career Roadmap System**. It takes a student's academic marks, coding rating, projects, and resume, and runs them through **5 standard machine learning algorithms** using scikit-learn:*
> 1. *Logistic Regression gives their placement chance in percentage.*
> 2. *KNN predicts their company tier and finds 5 similar past seniors.*
> 3. *Linear Regression estimates their salary package (LPA) with an error range.*
> 4. *K-Means groups them into peer clusters (Top Performers, Average, Needs Improvement).*
> 5. *Decision Tree highlights the top factors (like backlogs and coding) and provides improvement tips.*
>
> *We also added a rule-based company eligibility matcher for 25 recruiters and a keyword-based resume parser in Python. The frontend dashboard is built using Streamlit."*

---

## 2. File-by-File Summary (What to Say for Each File)

### Folder: `data/`

#### 1. `data/generate_dataset.py`
- **What code is inside:** A Python script using `numpy` and `pandas` that creates two CSV files: `students.csv` (3,000 synthetic student rows) and `companies.csv` (25 company criteria).
- **What to tell Mam:**
  > *"Mam, since real college placement data has student privacy restrictions, this script generates 3,000 realistic engineering student profiles based on standard bell-curve distributions and realistic placement weighting formulas. It also creates a table of 25 campus recruiters with their cutoffs."*

#### 2. `data/students.csv`
- **What is inside:** The dataset table with columns like `cgpa`, `backlogs`, `branch`, `coding_skill`, `num_projects`, `placed` (0 or 1), `company_tier` (1, 2, 3), and `package_lpa`.
- **What to tell Mam:**
  > *"Mam, this is our main dataset used for model training and testing. Any real college placement spreadsheet with the same column names can replace this file directly."*

#### 3. `data/companies.csv`
- **What is inside:** Table of 25 companies (e.g. Google, Amazon, TCS, Infosys, Cisco) with criteria: minimum CGPA, 10th/12th cutoffs, maximum allowed backlogs, required skills, and package ranges.
- **What to tell Mam:**
  > *"Mam, this is our recruiter database used by our eligibility engine to check which companies a student can sit for."*

---

### Folder: `ml/` (Machine Learning Core)

#### 4. `ml/preprocess.py`
- **What code is inside:** Data preparation pipeline using scikit-learn. Contains `StandardScaler` for numeric values and `OneHotEncoder` for text categories (`branch` and `experience_type`), plus `train_test_split(80/20)`.
- **What to tell Mam:**
  > *"Mam, ML algorithms cannot process text branches like 'CSE' or unscaled numbers where aptitude is 0-100 and CGPA is 5-10. This file normalizes all numbers to have mean 0 and variance 1, one-hot encodes categories, and splits data into training and test sets without data leakage."*

#### 5. `ml/train_models.py`
- **What code is inside:** The training script containing **5 separate functions**, one for each ML algorithm. Saves the trained models into the `models/` folder using `joblib`.
- **What to tell Mam:**
  > *"Mam, this is the core training file. When we run it, it trains all 5 algorithms:*
  > - *Logistic Regression for placement probability.*
  > - *KNN with a loop from K=1 to 20 to find the best K (K=8) for company tier.*
  > - *Linear Regression for salary package in LPA.*
  > - *K-Means (K=3) for peer grouping.*
  > - *Decision Tree (max_depth=4) for explainable feature importance.*
  > *All models and scalers are saved as `.pkl` files."*

#### 6. `ml/predict.py`
- **What code is inside:** Helper functions used by the Streamlit frontend. Takes the student's input dictionary, transforms it using the saved scaler, and feeds it into the trained models to return the predictions.
- **What to tell Mam:**
  > *"Mam, this file connects the frontend to the trained models. It has a `predict_all()` function that takes the user's form inputs and returns the probability, tier, package range, cluster persona, and improvement tips in one call."*

#### 7. `ml/evaluate.py`
- **What code is inside:** Model evaluation script. Calculates metrics (Accuracy, F1, RMSE, R², Silhouette) and generates 6 diagnostic charts into the `reports/` folder (Confusion Matrix, K-curve, LPA scatter plot, Elbow curve, Feature Importance bar chart, Decision Tree diagram).
- **What to tell Mam:**
  > *"Mam, this file evaluates our models on the test set and exports visual plots so we can verify there is no overfitting and inspect performance metrics."*

---

### Folder: `utils/` (Helper Tools)

#### 8. `utils/resume_parser.py`
- **What code is inside:** Resume text extractor using `pdfplumber` (for PDF) and `python-docx` (for Word). Uses regular expressions and keyword lists to detect ~40 technical skills, experience type, and project count.
- **What to tell Mam:**
  > *"Mam, this is a rule-based parser. When a student uploads their resume, it extracts the raw text and scans for programming languages and project keywords, automatically filling the form to save time."*

#### 9. `utils/company_matcher.py`
- **What code is inside:** Deterministic rule engine that compares the student's CGPA, board marks, backlogs, and skills against `companies.csv`.
- **What to tell Mam:**
  > *"Mam, recruitment policies have strict cutoffs. This module checks company eligibility rules and splits companies into Eligible, Almost Eligible (showing what is missing), and Not Eligible. It also uses our KNN tier prediction to highlight Best Fit companies."*

#### 10. `utils/domain_roadmap.py`
- **What code is inside:** Dictionaries containing structured beginner-to-advanced learning roadmaps, required tools, interview questions, and project ideas for 7 tech tracks (Web Dev, Data Science, AI/ML, Cyber Security, Cloud/DevOps, Testing, Core).
- **What to tell Mam:**
  > *"Mam, this file acts as a career guide. If a student chooses AI/ML or Web Dev, it checks their current skills against that domain, shows a readiness progress bar, and gives them a checklist of remaining topics and interview questions."*

---

### Main Files in Root Directory

#### 11. `app.py`
- **What code is inside:** The Streamlit frontend web application. Defines the sidebar navigation, the input form with sliders, the metric cards, interactive Plotly charts (gauge, radar), tabs, and the report download button.
- **What to tell Mam:**
  > *"Mam, this is our user interface built in Streamlit. It calls our backend ML modules (`predict.py`, `company_matcher.py`, `domain_roadmap.py`) and presents the results cleanly across three tabs: Placement Predictor, Model Performance & Metrics, and Dataset Explorer."*

#### 12. `requirements.txt`
- **What is inside:** List of required Python libraries: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `matplotlib`, `plotly`, `joblib`, `pdfplumber`, `python-docx`.
- **What to tell Mam:**
  > *"Mam, this lists all project dependencies so anyone can install them with `pip install -r requirements.txt`."*

#### 13. `README.md`
- **What is inside:** Project documentation, setup commands, and system architecture diagram.

#### 14. `VIVA_NOTES.md`
- **What is inside:** 15 likely viva questions and answers, definitions of evaluation metrics, and project limitations for oral defense.

---

## 3. How Data Flows in the Project (Step-by-Step)

```text
Step 1: data/generate_dataset.py
        └── Creates students.csv & companies.csv

Step 2: ml/preprocess.py
        └── Cleans, scales, and encodes features

Step 3: ml/train_models.py
        └── Trains 5 models & saves them in models/ (.pkl)

Step 4: ml/evaluate.py
        └── Generates charts & saves them in reports/ (.png)

Step 5: app.py (Streamlit UI)
        ├── Reads user input (or utils/resume_parser.py)
        ├── Calls ml/predict.py (runs 5 models)
        ├── Calls utils/company_matcher.py (checks 25 companies)
        ├── Calls utils/domain_roadmap.py (calculates skill gaps)
        └── Displays results and downloads report (.txt)
```

---

## 4. Cheat Sheet: Which Algorithm is Where?

| What Mam Asks | Which File to Point To | What Function / Code to Show |
|---|---|---|
| *"Where is Logistic Regression trained?"* | `ml/train_models.py` | `train_logistic_regression()` (Line 54) |
| *"Where did you tune K for KNN?"* | `ml/train_models.py` | `train_knn()` (Line 95, cross_val_score loop) |
| *"Where is Salary Regression predicted?"* | `ml/train_models.py` | `train_linear_regression()` (Line 150) |
| *"Where is K-Means clustering done?"* | `ml/train_models.py` | `train_kmeans()` (Line 185, K=3 centroids) |
| *"Where are Decision Tree factors calculated?"* | `ml/train_models.py` | `train_decision_tree()` (Line 250, feature_importances_) |
| *"Where is data scaling and one-hot encoding?"* | `ml/preprocess.py` | `create_preprocessor()` (ColumnTransformer) |
| *"Where is the resume parsed?"* | `utils/resume_parser.py` | `parse_resume()` |
| *"Where are company cutoffs checked?"* | `utils/company_matcher.py` | `evaluate_company_eligibility()` |
| *"Where is the web UI defined?"* | `app.py` | Form, Plotly charts, and tabs |
