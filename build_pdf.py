"""
Script to generate a beautifully styled PROJECT_EXPLANATION.pdf using ReportLab.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

pdf_filename = "/Users/krish/Desktop/placement_project/PROJECT_EXPLANATION.pdf"
doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1e3a8a"),
    alignment=0,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=colors.HexColor("#4b5563"),
    spaceAfter=12
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=16,
    textColor=colors.HexColor("#1e3a8a"),
    spaceBefore=10,
    spaceAfter=6
)

h2_style = ParagraphStyle(
    "FileH2",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    textColor=colors.HexColor("#0f172a"),
    spaceBefore=4,
    spaceAfter=2
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#1f2937"),
    spaceAfter=3
)

bold_body = ParagraphStyle(
    "BoldBody",
    parent=body_style,
    fontName="Helvetica-Bold"
)

say_style = ParagraphStyle(
    "WhatToSay",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#1e40af"),
    spaceAfter=6
)

table_header_style = ParagraphStyle(
    "TableHeader",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8.5,
    leading=11,
    textColor=colors.white
)

table_body_style = ParagraphStyle(
    "TableBody",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=10.5,
    textColor=colors.HexColor("#1f2937")
)

table_bold_style = ParagraphStyle(
    "TableBold",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8,
    leading=10.5,
    textColor=colors.HexColor("#1f2937")
)

story = []

# Title & Header
story.append(Paragraph("Campus Placement Predictor & Career Guide", title_style))
story.append(Paragraph("Codebase Structure & Viva Voce Quick Explainer Sheet", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=10))

# 1. Quick 1-Minute Pitch Box
story.append(Paragraph("1. How to Explain the Whole Project in 1 Minute to Mam", h1_style))

pitch_text = (
    "<b>What to say out loud:</b><br/>"
    "<i>\"Mam, our project is a <b>Campus Placement Predictor & Career Roadmap System</b>. It takes a student's "
    "academic marks, coding skill, projects, and resume, and evaluates them using <b>5 standard machine learning algorithms</b> "
    "from scikit-learn:<br/>"
    "1. <b>Logistic Regression</b> calculates placement chance in percentage.<br/>"
    "2. <b>K-Nearest Neighbors (KNN)</b> predicts company tier and retrieves the 5 most similar past students.<br/>"
    "3. <b>Linear Regression</b> estimates the expected salary package (LPA) with an error range.<br/>"
    "4. <b>K-Means Clustering</b> groups students into peer clusters (Top Performers, Average, Needs Improvement).<br/>"
    "5. <b>Decision Tree</b> explains the top hiring factors (active backlogs, coding) and suggests improvements.<br/>"
    "We also built a rule-based company eligibility engine for 25 recruiters and a keyword resume parser. "
    "The interactive dashboard is developed in Streamlit.\"</i>"
)

pitch_table = Table([[Paragraph(pitch_text, body_style)]], colWidths=[540])
pitch_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
]))
story.append(pitch_table)
story.append(Spacer(1, 10))

# 2. File-by-File Breakdown
story.append(Paragraph("2. File-by-File Breakdown (Which file does what?)", h1_style))

files_data = [
    # data/
    ("data/generate_dataset.py", "Data Generation Script",
     "Uses numpy and pandas to generate 3,000 realistic synthetic student profiles and a catalog of 25 recruiters.",
     "\"Mam, because real college records have privacy restrictions, this script creates 3,000 realistic student rows using standard bell curves and weighted placement logic, plus 25 recruiters with cutoff criteria.\""),
    
    ("data/students.csv", "Student Placement Dataset",
     "The main tabular dataset (3,000 rows, 15 columns) containing academic metrics, skills, backlogs, placed status, tier, and LPA.",
     "\"Mam, this is our training and testing dataset. Any real college placement spreadsheet with the same columns can directly replace this file.\""),
     
    ("data/companies.csv", "Recruiter Criteria Table",
     "Contains 25 companies (Google, Amazon, TCS, Infosys, Cisco, etc.) with minimum CGPA, board cutoffs, max backlogs, and required skills.",
     "\"Mam, this is our recruiter database used by our rule-based eligibility engine to check which companies a candidate qualifies for.\""),
     
    # ml/
    ("ml/preprocess.py", "Feature Preprocessing Pipeline",
     "Uses scikit-learn StandardScaler for numeric scaling and OneHotEncoder for branch and experience. Handles stratified 80/20 train/test split.",
     "\"Mam, ML models cannot take raw text like 'CSE' or unscaled values where aptitude is 0-100 and CGPA is 5-10. This file scales numbers to mean 0, encodes categories, and prevents data leakage.\""),

    ("ml/train_models.py", "Core Machine Learning Training File",
     "Trains all 5 algorithms in separate functions and saves models as .pkl files: Logistic Regression, KNN (tuned K=1..20, best K=8), Linear Regression, K-Means (K=3), and Decision Tree (depth=4).",
     "\"Mam, this is the main training file. Running it fits all 5 models, prints accuracy and RMSE to the terminal, and saves the trained models into the models/ folder using joblib.\""),

    ("ml/predict.py", "Frontend Inference Bridge",
     "High-level helper functions. Takes a student's input dictionary, scales it with the saved preprocessor, runs all 5 models, and returns unified results via predict_all().",
     "\"Mam, this file connects our Streamlit UI to the saved models. It takes the form inputs and produces the probability, tier, LPA, cluster persona, and tips in a single call.\""),

    ("ml/evaluate.py", "Model Evaluation & Diagnostic Charts",
     "Evaluates models on the test set and exports 6 diagnostic plots into reports/ (Confusion Matrix, KNN K-curve, LPA scatter, K-Means elbow, Feature Importance, Decision Tree diagram).",
     "\"Mam, this script evaluates model metrics on test data and saves diagnostic graphs into the reports/ folder to verify model behavior and check for overfitting.\""),

    # utils/
    ("utils/resume_parser.py", "Resume Text & Keyword Extractor",
     "Uses pdfplumber (for PDF) and python-docx (for Word) to extract text and regex keyword matching to detect ~40 tech skills, project counts, and experience level.",
     "\"Mam, this is a rule-based parser. When a student uploads their resume, it extracts skills and project mentions to automatically fill the form inputs.\""),

    ("utils/company_matcher.py", "Deterministic Company Matcher",
     "Evaluates student CGPA, marks, backlogs, and skills against companies.csv. Groups companies into Eligible, Almost Eligible (with missing criteria), and Not Eligible.",
     "\"Mam, recruitment policies have strict rules. This file checks eligibility cutoffs and uses our KNN tier prediction to highlight the Best Fit companies.\""),

    ("utils/domain_roadmap.py", "Career Track Roadmaps & Skill Gaps",
     "Roadmap dictionaries for 7 tech domains (Web Dev, Data Science, AI/ML, Cyber Security, Cloud, Testing, Core). Compares student skills against domain skills to show readiness %.",
     "\"Mam, this module acts as a career guide. When a student selects a dream domain like AI/ML, it calculates their readiness percentage and gives a checklist of remaining topics.\""),

    # Root
    ("app.py", "Streamlit Frontend Web Dashboard",
     "Multi-page interactive application featuring candidate input forms, Plotly gauge and radar charts, metric cards, company tabs, and report export.",
     "\"Mam, this is our user interface built in Streamlit. It takes inputs, calls predict_all(), and renders the placement results, company matches, and downloadable text report.\""),

    ("requirements.txt", "Project Dependencies",
     "Specifies required libraries: pandas, numpy, scikit-learn, streamlit, plotly, matplotlib, joblib, pdfplumber, python-docx.",
     "\"Mam, this file lists the exact Python packages required so the project can be run anywhere with pip install -r requirements.txt.\""),

    ("VIVA_NOTES.md", "Examiner Viva Question Bank",
     "Contains 15 high-yield viva questions with concise answers, metric definitions, and project limitations for oral defense.",
     "\"Mam, this is our study guide covering why each algorithm was selected, metric formulas, and typical oral exam questions.\"")
]

for filename, role, what_code, what_to_say in files_data:
    card_content = [
        [Paragraph(f"<b>File:</b> <font color='#1e3a8a'>{filename}</font> &nbsp;|&nbsp; <b>{role}</b>", h2_style)],
        [Paragraph(f"<b>Code inside:</b> {what_code}", body_style)],
        [Paragraph(f"<b>What to tell Mam:</b> {what_to_say}", say_style)]
    ]
    card_table = Table(card_content, colWidths=[540])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 4))

story.append(PageBreak())

# 3. Data Flow Diagram
story.append(Paragraph("3. How Data Flows in the Project (5 Simple Steps)", h1_style))

flow_text = (
    "<b>Step 1: Data Generation</b> (<code>data/generate_dataset.py</code>) "
    "&rarr; Creates <code>students.csv</code> & <code>companies.csv</code>.<br/>"
    "<b>Step 2: Preprocessing</b> (<code>ml/preprocess.py</code>) "
    "&rarr; Normalizes numeric columns with <code>StandardScaler</code> and one-hot encodes branches.<br/>"
    "<b>Step 3: Model Training</b> (<code>ml/train_models.py</code>) "
    "&rarr; Trains all 5 models and saves them as <code>.pkl</code> files in <code>models/</code>.<br/>"
    "<b>Step 4: Evaluation & Reporting</b> (<code>ml/evaluate.py</code>) "
    "&rarr; Evaluates test set and saves 6 diagnostic charts in <code>reports/</code>.<br/>"
    "<b>Step 5: Interactive Web App</b> (<code>app.py</code>) "
    "&rarr; Collects student inputs, calls <code>ml/predict.py</code>, <code>utils/company_matcher.py</code>, "
    "and <code>utils/domain_roadmap.py</code> to display results on screen."
)

flow_table = Table([[Paragraph(flow_text, body_style)]], colWidths=[540])
flow_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
]))
story.append(flow_table)
story.append(Spacer(1, 12))

# 4. Viva Cheat Sheet Table
story.append(Paragraph("4. Quick Viva Cheat Sheet: Where is each algorithm in the code?", h1_style))

cheat_rows = [
    [
        Paragraph("Question Mam Might Ask", table_header_style),
        Paragraph("File Name to Open", table_header_style),
        Paragraph("Function / Key Code to Point To", table_header_style),
        Paragraph("Core Concept", table_header_style)
    ],
    [
        Paragraph("Where is Logistic Regression trained?", table_bold_style),
        Paragraph("ml/train_models.py", table_body_style),
        Paragraph("train_logistic_regression() (Line 54)", table_body_style),
        Paragraph("Sigmoid function squashes score into 0-100% probability via predict_proba()", table_body_style)
    ],
    [
        Paragraph("Where did you tune K for KNN?", table_bold_style),
        Paragraph("ml/train_models.py", table_body_style),
        Paragraph("train_knn() (Line 95, cross_val loop)", table_body_style),
        Paragraph("Looped K from 1 to 20 with 5-fold CV; peak accuracy found at K=8", table_body_style)
    ],
    [
        Paragraph("Where is salary package predicted?", table_bold_style),
        Paragraph("ml/train_models.py", table_body_style),
        Paragraph("train_linear_regression() (Line 150)", table_body_style),
        Paragraph("Ordinary Least Squares regression; salary band is prediction ± RMSE", table_body_style)
    ],
    [
        Paragraph("Where is K-Means clustering done?", table_bold_style),
        Paragraph("ml/train_models.py", table_body_style),
        Paragraph("train_kmeans() (Line 185)", table_body_style),
        Paragraph("Unsupervised K=3 centroids on 5 skill dimensions, labeled by average CGPA/coding", table_body_style)
    ],
    [
        Paragraph("Where are decision rules & feature importance?", table_bold_style),
        Paragraph("ml/train_models.py", table_body_style),
        Paragraph("train_decision_tree() (Line 250)", table_body_style),
        Paragraph("max_depth=4; tree splits show backlogs (68%) and coding (14%) are top drivers", table_body_style)
    ],
    [
        Paragraph("Where is scaling and encoding implemented?", table_bold_style),
        Paragraph("ml/preprocess.py", table_body_style),
        Paragraph("create_preprocessor() (Line 48)", table_body_style),
        Paragraph("StandardScaler for numbers and OneHotEncoder for branch/experience", table_body_style)
    ],
    [
        Paragraph("Where is resume parsing handled?", table_bold_style),
        Paragraph("utils/resume_parser.py", table_body_style),
        Paragraph("parse_resume() (Line 72)", table_body_style),
        Paragraph("Extracts text via pdfplumber and matches against ~40 tech skill keywords", table_body_style)
    ],
    [
        Paragraph("Where are company cutoffs checked?", table_bold_style),
        Paragraph("utils/company_matcher.py", table_body_style),
        Paragraph("evaluate_company_eligibility() (Line 48)", table_body_style),
        Paragraph("Deterministic check of CGPA, 10th/12th, backlogs, and skill overlap %", table_body_style)
    ],
    [
        Paragraph("Where is the user interface defined?", table_bold_style),
        Paragraph("app.py", table_body_style),
        Paragraph("Placement Predictor form & tabs", table_body_style),
        Paragraph("Streamlit forms, sliders, Plotly gauge/radar, and .txt report generator", table_body_style)
    ]
]

cheat_table = Table(cheat_rows, colWidths=[135, 105, 140, 160])
cheat_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
]))
story.append(cheat_table)

doc.build(story)
print(f"PDF successfully generated at: {pdf_filename}")
