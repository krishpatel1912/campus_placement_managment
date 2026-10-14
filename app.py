"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: app.py
Purpose: Main Streamlit Frontend Application
         College Placement Analysis and Career Guidance System
=============================================================================
"""

import os
import sys

# Ensure local directories are in python path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "ml"))
sys.path.insert(0, os.path.join(APP_DIR, "utils"))

# Ensure writable matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.join(APP_DIR, ".mplcache")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Import ML inference and utility modules
from predict import get_models, predict_all
from company_matcher import evaluate_company_eligibility, load_companies
from domain_roadmap import analyze_skill_gaps, DOMAINS_DATA
from resume_parser import parse_resume

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Campus Placement Predictor & Career Guide",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean academic layout
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1.1rem;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        margin-bottom: 0.8rem;
    }
    .card-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .card-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
    }
    .badge-tier1 {
        background-color: #fef08a;
        color: #854d0e;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-tier2 {
        background-color: #bae6fd;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-tier3 {
        background-color: #e2e8f0;
        color: #334155;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_ml_models():
    """
    Loads all trained scikit-learn models and metadata once using Streamlit cache.
    """
    return get_models()

try:
    models = load_all_ml_models()
except Exception as e:
    st.error(f"Failed to load trained models: {e}. Please ensure you ran 'python ml/train_models.py' first.")
    models = None

# Pre-formatted list of skills for the multiselect widget
AVAILABLE_SKILLS = sorted([
    "Python", "Java", "C++", "C", "JavaScript", "TypeScript", "SQL", "HTML", "CSS",
    "React", "Node.js", "Django", "Flask", "Spring Boot", "FastAPI",
    "Pandas", "NumPy", "Scikit-Learn", "Machine Learning", "Deep Learning",
    "Docker", "Kubernetes", "AWS", "Azure", "Git", "Linux",
    "Selenium", "Automation Testing", "Manual Testing", "Ethical Hacking",
    "Cyber Security", "Networking", "Power BI", "Tableau", "Excel",
    "MATLAB", "Embedded Systems", "AutoCAD", "SolidWorks"
])

# -----------------------------------------------------------------------------
# Navigation Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Placement System")
st.sidebar.markdown("**Mini Project | Machine Learning**")

page = st.sidebar.radio(
    "Select View",
    [
        "Placement Predictor",
        "Model Performance & Metrics",
        "Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project Architecture:**\n\n"
    "1. **Logistic Regression:** Placement Chance (%)\n"
    "2. **KNN (K=8):** Company Tier & Peer Matching\n"
    "3. **Linear Regression:** Expected Salary (LPA)\n"
    "4. **K-Means (K=3):** Student Clustering & Radar\n"
    "5. **Decision Tree:** Factor Importance & Advice"
)

st.sidebar.markdown("---")
st.sidebar.caption("Trained on 3,000 student records using scikit-learn.")


# =============================================================================
# PAGE 1: PLACEMENT PREDICTOR (MAIN PAGE)
# =============================================================================
if page == "Placement Predictor":
    st.markdown('<div class="main-header">Campus Placement Prediction & Career Guidance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter academic background, practical experience, and technical skills to evaluate placement readiness.</div>', unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # RESUME UPLOAD SECTION (AUTO-POPULATION)
    # -------------------------------------------------------------------------
    with st.expander("Upload Resume for Form Auto-Fill (PDF / DOCX)", expanded=False):
        col_up1, col_up2 = st.columns([3, 2])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload student resume to automatically extract skills, projects count, and experience:",
                type=["pdf", "docx", "txt"]
            )
        with col_up2:
            st.markdown("""
            **Parser Details:**
            Uses Python text extraction (`pdfplumber` / `python-docx`) and regex keyword
            matching to identify technical skills, project sections, and internship mentions.
            """)
            
        parsed_data = None
        if uploaded_file is not None:
            with st.spinner("Extracting details from resume..."):
                file_bytes = uploaded_file.read()
                parsed_data = parse_resume(file_bytes, uploaded_file.name)
                st.success(parsed_data["summary"])
                st.write("**Identified Skills:**", ", ".join(parsed_data["skills"]) if parsed_data["skills"] else "None detected")

    # -------------------------------------------------------------------------
    # STUDENT PROFILE FORM
    # -------------------------------------------------------------------------
    st.markdown("### Student Profile & Academic Details")
    
    # Prepopulate defaults if resume was parsed
    default_skills = parsed_data["skills"] if (parsed_data and parsed_data["skills"]) else ["Python", "SQL", "Git"]
    valid_defaults = [s for s in default_skills if s in AVAILABLE_SKILLS]
    if not valid_defaults:
        valid_defaults = ["Python", "SQL"]
        
    default_exp = parsed_data["experience_type"] if parsed_data else "Fresher"
    default_proj = parsed_data["num_projects"] if parsed_data else 2
    default_cert = parsed_data["num_certifications"] if parsed_data else 1
    default_coding = parsed_data["estimated_coding_skill"] if parsed_data else 6

    with st.form("placement_form"):
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            student_name = st.text_input("Candidate Name", value="Rahul Sharma")
            branch = st.selectbox("Engineering Branch", ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"], index=0)
        with r1_c2:
            cgpa = st.number_input("Current CGPA (Scale of 10.0)", min_value=5.0, max_value=10.0, value=7.8, step=0.05)
            backlogs = st.number_input("Active Backlogs Count", min_value=0, max_value=5, value=0, step=1)
        with r1_c3:
            tenth_pct = st.number_input("10th Board Percentage (%)", min_value=50.0, max_value=100.0, value=78.5, step=0.5)
            twelfth_pct = st.number_input("12th Board Percentage (%)", min_value=50.0, max_value=100.0, value=75.0, step=0.5)
            
        st.markdown("---")
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1:
            exp_options = ["Fresher", "Internship", "Work Experience"]
            exp_idx = exp_options.index(default_exp) if default_exp in exp_options else 0
            experience_type = st.selectbox("Practical Experience Type", exp_options, index=exp_idx)
            num_projects = st.slider("Academic / Personal Projects", min_value=0, max_value=6, value=default_proj)
        with r2_c2:
            coding_skill = st.slider("Coding & DSA Proficiency (1 to 10)", min_value=1, max_value=10, value=default_coding)
            comm_skill = st.slider("Communication Skill Rating (1 to 10)", min_value=1, max_value=10, value=7)
        with r2_c3:
            num_cert = st.slider("Completed Certifications", min_value=0, max_value=8, value=default_cert)
            aptitude_score = st.slider("Aptitude Assessment Score (0 to 100)", min_value=15, max_value=100, value=70)
            
        st.markdown("---")
        r3_c1, r3_c2 = st.columns([3, 2])
        with r3_c1:
            selected_skills = st.multiselect(
                "Acquired Technical Skills",
                AVAILABLE_SKILLS,
                default=valid_defaults
            )
        with r3_c2:
            target_domain = st.selectbox(
                "Target Career Track",
                list(DOMAINS_DATA.keys()),
                index=2  # Defaults to AI/ML
            )
            
        submit_btn = st.form_submit_button("Run Placement Analysis", use_container_width=True, type="primary")

    # -------------------------------------------------------------------------
    # PREDICTION & RESULTS SECTION
    # -------------------------------------------------------------------------
    if submit_btn:
        if models is None:
            st.error("Trained models not loaded. Please run 'python ml/train_models.py' first.")
        else:
            student_dict = {
                "name": student_name,
                "tenth_percent": float(tenth_pct),
                "twelfth_percent": float(twelfth_pct),
                "cgpa": float(cgpa),
                "backlogs": int(backlogs),
                "branch": branch,
                "experience_type": experience_type,
                "num_projects": int(num_projects),
                "num_certifications": int(num_cert),
                "coding_skill": int(coding_skill),
                "communication_skill": int(comm_skill),
                "aptitude_score": float(aptitude_score),
                "skills_count": len(selected_skills),
                "skills": selected_skills,
                "target_domain": target_domain
            }
            
            with st.spinner("Processing input through trained models..."):
                results = predict_all(student_dict)
                company_eval = evaluate_company_eligibility(student_dict, results["tier_and_peers"]["tier_code"])
                domain_eval = analyze_skill_gaps(target_domain, selected_skills)
                
            st.success("Placement analysis generated successfully. Detailed breakdowns are provided below.")
            
            # Key Metric Highlight Cards
            k_c1, k_c2, k_c3, k_c4 = st.columns(4)
            with k_c1:
                prob = results["placement"]["placement_chance_pct"]
                prob_color = "#15803d" if prob >= 70 else ("#b45309" if prob >= 40 else "#b91c1c")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Placement Probability (Logistic Regression)</div>
                    <div class="card-value" style="color: {prob_color};">{prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with k_c2:
                tier_str = results["tier_and_peers"]["tier_name"].split("(")[0].strip()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Fitted Company Tier (KNN)</div>
                    <div class="card-value">{tier_str}</div>
                </div>
                """, unsafe_allow_html=True)
            with k_c3:
                lpa_range = results["salary"]["range_str"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Expected Package (Linear Regression)</div>
                    <div class="card-value">{lpa_range}</div>
                </div>
                """, unsafe_allow_html=True)
            with k_c4:
                cluster_str = results["cluster"]["cluster_persona"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Peer Group (K-Means Clustering)</div>
                    <div class="card-value">{cluster_str}</div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed Output Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Placement Probability",
                "Tier & Similar Students",
                "Expected Package",
                "Peer Group Analysis",
                "Improvement Suggestions",
                "Company Eligibility",
                "Domain Roadmap"
            ])
            
            # --- TAB 1: PLACEMENT GAUGE ---
            with tab1:
                st.markdown("#### Placement Probability Analysis")
                c1, c2 = st.columns([1, 1])
                with c1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Placement Probability (%)", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#1e3a8a"},
                            'steps': [
                                {'range': [0, 40], 'color': "#fee2e2"},
                                {'range': [40, 70], 'color': "#fef3c7"},
                                {'range': [70, 100], 'color': "#dcfce7"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 3},
                                'thickness': 0.75,
                                'value': prob
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with c2:
                    st.markdown("##### Logistic Regression Assessment")
                    if prob >= 75:
                        st.success(
                            f"**High Placement Probability ({prob:.1f}%)**: Candidate has strong academic standing "
                            f"(CGPA: {cgpa}), coding proficiency ({coding_skill}/10), and zero/low backlogs. "
                            "Favorable for Day-1 on-campus drives."
                        )
                    elif prob >= 45:
                        st.warning(
                            f"**Moderate Placement Probability ({prob:.1f}%)**: Reasonable standing, "
                            "but clearing pending arrears and strengthening coding interview practice "
                            "will significantly increase company shortlist chances."
                        )
                    else:
                        st.error(
                            f"**Below Average ({prob:.1f}%)**: High risk of elimination in screening stages. "
                            f"{'Active backlogs (' + str(backlogs) + ') are the main limiting factor.' if backlogs > 0 else 'Increasing project work and DSA practice is recommended.'}"
                        )
                        
                    st.info(
                        r"**Mathematical Basis:** Features are standardized using StandardScaler and evaluated "
                        r"via the Sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$."
                    )

            # --- TAB 2: TIER & SIMILAR PEERS ---
            with tab2:
                st.markdown("#### Company Tier & Nearest Neighbor Students")
                st.markdown(f"**Predicted Category:** `{results['tier_and_peers']['tier_name']}`")
                st.markdown(
                    "The table below shows the **5 most similar placed students** identified from the training "
                    "dataset based on Euclidean distance across standardized features (K=8):"
                )
                st.dataframe(
                    results["tier_and_peers"]["similar_students"],
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "Note: Demonstrates historical outcomes achieved by past students with comparable academic and technical profiles."
                )

            # --- TAB 3: EXPECTED PACKAGE ---
            with tab3:
                st.markdown("#### Expected Salary Package (LPA)")
                p_c1, p_c2 = st.columns([1, 1])
                with p_c1:
                    st.metric(
                        label="Predicted Base CTC",
                        value=f"{results['salary']['predicted_lpa']} LPA",
                        delta=f"Standard Error: ±{results['salary']['rmse']} LPA"
                    )
                    st.markdown(f"""
                    **Estimated CTC Range:**
                    ### `{results['salary']['range_str']}`
                    """)
                with p_c2:
                    st.markdown("##### Linear Regression Formulation")
                    st.markdown(fr"""
                    - **Predicted Package:** $\hat{{y}} = {results['salary']['predicted_lpa']}$ LPA
                    - **Model Root Mean Squared Error (RMSE):** ${results['salary']['rmse']}$ LPA
                    - **Upper Band:** ${results['salary']['range_max']}$ LPA (Product companies / High performance)
                    - **Lower Band:** ${results['salary']['range_min']}$ LPA (Service recruiters / Base compensation)
                    """)

            # --- TAB 4: PEER RADAR BENCHMARK ---
            with tab4:
                st.markdown("#### Peer Group Clustering (K-Means)")
                st.markdown(f"**Identified Cluster:** `{results['cluster']['cluster_persona']}` (Unsupervised K=3)")
                
                r_c1, r_c2 = st.columns([3, 2])
                with r_c1:
                    dims = results["cluster"]["radar_dimensions"]
                    student_v = results["cluster"]["student_values"]
                    cluster_v = results["cluster"]["cluster_averages"]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=student_v + [student_v[0]],
                        theta=dims + [dims[0]],
                        fill='toself',
                        name='Your Profile',
                        line_color='#2563eb'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=cluster_v + [cluster_v[0]],
                        theta=dims + [dims[0]],
                        fill='toself',
                        name=f"{results['cluster']['cluster_persona']} Mean",
                        line_color='#d97706',
                        opacity=0.6
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 15])),
                        showlegend=True,
                        height=360,
                        margin=dict(l=30, r=30, t=30, b=30)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                with r_c2:
                    st.markdown("##### Dimension Comparisons")
                    for d, sv, cv in zip(dims, student_v, cluster_v):
                        diff = sv - cv
                        status = "Above cluster average" if diff >= 0 else "Below cluster average"
                        st.write(f"- **{d}**: Candidate: `{sv}` | Cluster Avg: `{cv}` ({status})")

            # --- TAB 5: IMPROVEMENT TIPS ---
            with tab5:
                st.markdown("#### Key Factors & Improvement Suggestions")
                st.markdown("Recommendations derived from **Decision Tree Feature Importances and Split Rules**:")
                
                for tip in results["explanation"]["improvement_tips"]:
                    priority_tag = tip["priority"]
                    with st.container():
                        st.markdown(f"##### [{priority_tag} PRIORITY] {tip['factor']}")
                        st.markdown(f"**Status:** {tip['observation']}")
                        st.markdown(f"**Recommendation:** {tip['action']}")
                        st.markdown("---")

            # --- TAB 6: COMPANY ELIGIBILITY ---
            with tab6:
                st.markdown("#### Campus Recruiter Eligibility (25 Recruiter Catalog)")
                
                # Best Fit Highlight
                if company_eval["best_fit"]:
                    st.markdown("##### Recommended Fit Companies (Matching Predicted Tier)")
                    bf_cols = st.columns(min(len(company_eval["best_fit"]), 3))
                    for idx, c in enumerate(company_eval["best_fit"][:3]):
                        with bf_cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left-color: #059669;">
                                <div class="card-title">{c['tier']} | {c['domain']}</div>
                                <div class="card-value" style="font-size: 1.15rem;">{c['company']}</div>
                                <div style="font-size: 0.88rem; margin-top: 4px;">
                                    <b>Package:</b> {c['package_range']}<br>
                                    <b>Skill Match:</b> {c['skill_match_pct']}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                sub_t1, sub_t2, sub_t3 = st.tabs([
                    f"Eligible ({len(company_eval['eligible'])})",
                    f"Almost Eligible ({len(company_eval['almost_eligible'])})",
                    f"Not Eligible ({len(company_eval['not_eligible'])})"
                ])
                
                with sub_t1:
                    if not company_eval["eligible"]:
                        st.info("No companies currently match 100% of eligibility cutoffs. Review 'Almost Eligible' criteria.")
                    else:
                        for c in company_eval["eligible"]:
                            with st.expander(f"{c['company']} ({c['tier']} - {c['domain']}) | CTC: {c['package_range']}"):
                                st.write(f"- **Skill Overlap:** {c['skill_match_pct']}%")
                                st.write(f"- **Matching Skills:** {', '.join(c['matching_skills']) if c['matching_skills'] else 'None'}")
                                st.write(f"- **Cutoffs Passed:** Minimum CGPA: {c['min_cgpa']}, Max Allowed Backlogs: {c['max_backlogs']}")

                with sub_t2:
                    if not company_eval["almost_eligible"]:
                        st.info("None in this category.")
                    else:
                        for c in company_eval["almost_eligible"]:
                            with st.expander(f"{c['company']} ({c['tier']} - {c['domain']}) | CTC: {c['package_range']}"):
                                st.write(f"- **Skill Overlap:** {c['skill_match_pct']}%")
                                st.warning("**Pending Criteria:**\n" + "\n".join([f"- {r}" for r in c.get('missing_criteria', [])]))
                                if c['missing_skills']:
                                    st.write(f"- **Missing Skills:** {', '.join(c['missing_skills'])}")

                with sub_t3:
                    for c in company_eval["not_eligible"]:
                        with st.expander(f"{c['company']} ({c['tier']} - {c['domain']})"):
                            st.error("**Ineligibility Reasons:**\n" + "\n".join([f"- {r}" for r in c.get('reasons', [])]))

            # --- TAB 7: DOMAIN ROADMAP & GAPS ---
            with tab7:
                st.markdown(f"#### Career Track Roadmap: {domain_eval['domain_name']}")
                st.markdown(f"*{domain_eval['description']}*")
                
                st.markdown(f"##### Domain Skill Readiness: **{domain_eval['readiness_pct']}%**")
                st.progress(domain_eval["readiness_pct"] / 100.0)
                
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    st.success(f"**Identified Skills ({len(domain_eval['covered_skills'])}):**\n" +
                               (", ".join(domain_eval['covered_skills']) if domain_eval['covered_skills'] else "None"))
                with d_c2:
                    st.warning(f"**Pending Skills to Learn ({len(domain_eval['missing_skills'])}):**\n" +
                               (", ".join(domain_eval['missing_skills']) if domain_eval['missing_skills'] else "All core skills covered."))
                    
                st.markdown("##### Structured Preparation Topics")
                for step in domain_eval["roadmap_steps"]:
                    st.checkbox(f"**{step['level']}**: {step['topic']}", value=False)
                    
                st.markdown("##### Recommended Projects for Resume")
                for p in domain_eval["suggested_projects"]:
                    st.write(f"- {p}")
                    
                st.markdown("##### Core Interview Topics")
                for q in domain_eval["interview_topics"]:
                    st.write(f"- *\"{q}\"*")

            # -----------------------------------------------------------------
            # DOWNLOADABLE REPORT GENERATOR
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown("#### Download Assessment Summary")
            
            report_text = f"""================================================================================
CAMPUS PLACEMENT ASSESSMENT SUMMARY REPORT
================================================================================
Candidate Name        : {student_name}
Branch                : {branch}
Current CGPA          : {cgpa:.2f}
10th / 12th Marks     : {tenth_pct:.1f}% / {twelfth_pct:.1f}%
Active Backlogs       : {backlogs}
Experience Type       : {experience_type}
Completed Projects    : {num_projects}
Certifications        : {num_cert}
Coding Rating (1-10)  : {coding_skill}/10
Aptitude Score        : {aptitude_score:.1f}/100
Technical Skills      : {', '.join(selected_skills)}
Target Track          : {target_domain}

--------------------------------------------------------------------------------
1. MACHINE LEARNING PREDICTION RESULTS
--------------------------------------------------------------------------------
- Placement Probability (Logistic Regression) : {results['placement']['placement_chance_pct']:.1f}%
- Fitted Company Tier (KNN Classifier)        : {results['tier_and_peers']['tier_name']}
- Expected Package (Linear Regression)        : {results['salary']['range_str']}
- Peer Group Persona (K-Means Clustering)     : {results['cluster']['cluster_persona']}

--------------------------------------------------------------------------------
2. KEY RECOMMENDATIONS (DECISION TREE)
--------------------------------------------------------------------------------
"""
            for tip in results["explanation"]["improvement_tips"]:
                report_text += f"\n- [{tip['priority']}] {tip['factor']}: {tip['action']}"
                
            report_text += f"""

--------------------------------------------------------------------------------
3. ELIGIBLE RECRUITERS ({len(company_eval['eligible'])})
--------------------------------------------------------------------------------
"""
            for c in company_eval["eligible"][:10]:
                report_text += f"- {c['company']} ({c['tier']}, {c['domain']}) | CTC: {c['package_range']} | Skill Overlap: {c['skill_match_pct']}%\n"
                
            report_text += f"""
--------------------------------------------------------------------------------
4. TARGET TRACK ({target_domain}) SKILL READINESS
--------------------------------------------------------------------------------
- Track Readiness     : {domain_eval['readiness_pct']}%
- Acquired Skills     : {', '.join(domain_eval['covered_skills'])}
- Pending Skills      : {', '.join(domain_eval['missing_skills'])}

================================================================================
Generated by Campus Placement Predictor System (Python & scikit-learn)
================================================================================
"""
            st.download_button(
                label="Download Assessment Summary (.txt)",
                data=report_text,
                file_name=f"{student_name.replace(' ', '_')}_Placement_Report.txt",
                mime="text/plain",
                use_container_width=True
            )


# =============================================================================
# PAGE 2: MODEL PERFORMANCE & METRICS
# =============================================================================
elif page == "Model Performance & Metrics":
    st.markdown('<div class="main-header">Model Performance & Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation summary tables and diagnostic plots for all 5 implemented algorithms.</div>', unsafe_allow_html=True)
    
    # Model Comparison Table
    table_path = os.path.join(APP_DIR, "models", "model_comparison_table.pkl")
    if os.path.exists(table_path):
        df_comp = joblib.load(table_path)
        st.markdown("#### Model Performance Summary")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.warning("Comparison table not found. Please run 'python ml/evaluate.py' to generate.")
        
    st.markdown("---")
    st.markdown("#### Diagnostic Visualization Charts")
    
    rep_dir = os.path.join(APP_DIR, "reports")
    
    chart_tabs = st.tabs([
        "Confusion Matrix",
        "KNN K-Tuning",
        "Linear Regression Fit",
        "K-Means Elbow",
        "Feature Importance",
        "Decision Tree Diagram"
    ])
    
    with chart_tabs[0]:
        st.markdown("##### Logistic Regression: Confusion Matrix")
        p = os.path.join(rep_dir, "confusion_matrix.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            st.info("""
            **Observations:**
            - **True Negatives:** Correctly predicted unplaced students.
            - **True Positives:** Correctly predicted placed students.
            - High recall indicates the model reliably identifies candidates who meet placement standards.
            """)
        else:
            st.warning("Chart missing. Run ml/evaluate.py.")
            
    with chart_tabs[1]:
        st.markdown("##### KNN Hyperparameter Tuning: K Value vs Cross-Validation Accuracy")
        p = os.path.join(rep_dir, "knn_k_accuracy.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            st.info("""
            **Observations:**
            - Cross-validation was run across K values from 1 to 20.
            - Lower K values exhibit higher variance, whereas higher K values smooth out boundaries.
            - The optimal accuracy peak was obtained at K = 8.
            """)
        else:
            st.warning("Chart missing. Run ml/evaluate.py.")

    with chart_tabs[2]:
        st.markdown("##### Linear Regression: Actual vs Predicted Package (LPA)")
        p = os.path.join(rep_dir, "lpa_actual_vs_predicted.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            st.info("""
            **Observations:**
            - Points plotted along the dashed line (y = x) represent test set predictions.
            - The model achieves an R-squared of ~0.66, capturing salary trends across tiers.
            """)
        else:
            st.warning("Chart missing. Run ml/evaluate.py.")

    with chart_tabs[3]:
        st.markdown("##### K-Means Clustering: Elbow Curve (Inertia vs K)")
        p = os.path.join(rep_dir, "kmeans_elbow.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            st.info("""
            **Observations:**
            - Evaluates within-cluster sum of squares (Inertia) for K from 1 to 8.
            - The elbow inflection occurs around K = 3, yielding distinct student tiers.
            """)
        else:
            st.warning("Chart missing. Run ml/evaluate.py.")

    with chart_tabs[4]:
        st.markdown("##### Decision Tree: Feature Importance")
        p = os.path.join(rep_dir, "feature_importance.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            st.info("""
            **Observations:**
            - Active backlogs carries the largest weight (~68%), aligning with institutional cutoff rules.
            - Coding proficiency and college CGPA form the next strongest contributors.
            """)
        else:
            st.warning("Chart missing. Run ml/evaluate.py.")

    with chart_tabs[5]:
        st.markdown("##### Decision Tree: Structure (Max Depth = 4)")
        p = os.path.join(rep_dir, "decision_tree.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            st.info("""
            **Observations:**
            - Max depth is restricted to 4 to maintain interpretability.
            - Each node displays the decision threshold, sample count, and Gini impurity.
            """)
        else:
            st.warning("Chart missing. Run ml/evaluate.py.")


# =============================================================================
# PAGE 3: DATASET EXPLORER
# =============================================================================
elif page == "Dataset Explorer":
    st.markdown('<div class="main-header">Placement Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of the 3,000 student training records, feature distributions, and correlation matrix.</div>', unsafe_allow_html=True)
    
    data_path = os.path.join(APP_DIR, "data", "students.csv")
    if os.path.exists(data_path):
        df_students = pd.read_csv(data_path)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{len(df_students):,}")
        m2.metric("Placement Ratio", f"{df_students['placed'].mean()*100:.1f}%")
        m3.metric("Average CGPA", f"{df_students['cgpa'].mean():.2f}")
        m4.metric("Avg Placed CTC", f"{df_students[df_students['placed']==1]['package_lpa'].mean():.2f} LPA")
        
        st.markdown("#### Sample Records")
        st.dataframe(df_students.head(15), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Feature Distributions")
        
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            fig_hist = px.histogram(
                df_students,
                x="cgpa",
                color="placed",
                barmode="overlay",
                title="CGPA Distribution by Placement Outcome",
                color_discrete_map={0: "#dc2626", 1: "#16a34a"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with d_c2:
            fig_coding = px.histogram(
                df_students,
                x="coding_skill",
                color="company_tier",
                title="Coding Rating vs Company Tier (0=Unplaced, 1=T3, 2=T2, 3=T1)",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_coding, use_container_width=True)
            
        st.markdown("#### Correlation Heatmap")
        num_cols = df_students.select_dtypes(include=[np.number]).columns
        corr_matrix = df_students[num_cols].corr().round(2)
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Academic Features and Placement Variables"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("#### Recruiter Eligibility Catalog")
        comp_path = os.path.join(APP_DIR, "data", "companies.csv")
        if os.path.exists(comp_path):
            df_comp_list = pd.read_csv(comp_path)
            st.dataframe(df_comp_list, use_container_width=True)
    else:
        st.error("students.csv not found. Please run data/generate_dataset.py first.")
