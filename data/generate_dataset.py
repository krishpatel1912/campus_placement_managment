"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: data/generate_dataset.py
Purpose: Synthesizes a realistic dataset of 3,000 engineering student profiles
         and a catalog of ~25 campus recruiters with eligibility criteria.
=============================================================================

NOTE FOR STUDENTS & EXAMINERS:
This dataset is synthetically generated using controlled mathematical
distributions and domain logic to mirror genuine Indian campus placement
trends. In a real-world production deployment, this generator script can be
swapped out by simply placing a real college placement CSV with identical
column headers into 'data/students.csv'.
"""

import os
import numpy as np
import pandas as pd

# Set fixed random seed for reproducible results across all runs
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_student_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generates realistic synthetic placement records for engineering students.
    
    Parameters:
        n_samples (int): Number of student records to generate (default: 3000).
        
    Returns:
        pd.DataFrame: Table containing academic metrics, skills, and placement outcomes.
    """
    # -------------------------------------------------------------------------
    # 1. Academic Performance Features
    # -------------------------------------------------------------------------
    # 10th and 12th board percentages generally follow a bell curve centered around 72-76%
    tenth_percent = np.clip(np.random.normal(loc=76.0, scale=10.0, size=n_samples), 50.0, 98.0).round(2)
    twelfth_percent = np.clip(np.random.normal(loc=73.0, scale=10.5, size=n_samples), 50.0, 98.0).round(2)
    
    # College CGPA typically ranges between 5.5 and 9.5
    cgpa = np.clip(np.random.normal(loc=7.4, scale=1.1, size=n_samples), 5.0, 9.8).round(2)
    
    # Active backlogs: mostly 0, few with 1 or 2, rarely 3 to 5
    backlog_probs = [0.65, 0.18, 0.09, 0.05, 0.02, 0.01]
    backlogs = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=backlog_probs)
    
    # Engineering branches with typical proportions
    branches = ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"]
    branch_probs = [0.35, 0.20, 0.18, 0.10, 0.10, 0.07]
    branch = np.random.choice(branches, size=n_samples, p=branch_probs)
    
    # -------------------------------------------------------------------------
    # 2. Experience & Skill Features
    # -------------------------------------------------------------------------
    # Prior practical exposure
    exp_types = ["Fresher", "Internship", "Work Experience"]
    exp_probs = [0.55, 0.38, 0.07]
    experience_type = np.random.choice(exp_types, size=n_samples, p=exp_probs)
    
    # Projects completed during college (0 to 6)
    num_projects = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples, 
                                    p=[0.08, 0.22, 0.32, 0.22, 0.10, 0.04, 0.02])
    
    # Certifications earned (0 to 8)
    num_certifications = np.random.choice(range(9), size=n_samples, 
                                          p=[0.12, 0.25, 0.28, 0.18, 0.09, 0.04, 0.02, 0.01, 0.01])
    
    # Coding skill rating on a scale of 1 to 10 (CSE/IT slightly higher on average)
    coding_base = np.random.normal(loc=5.8, scale=2.1, size=n_samples)
    branch_coding_boost = np.where(np.isin(branch, ["CSE", "IT"]), 1.0, 0.0)
    coding_skill = np.clip(np.round(coding_base + branch_coding_boost), 1, 10).astype(int)
    
    # Soft skills & communication rating on scale of 1 to 10
    communication_skill = np.clip(np.round(np.random.normal(loc=6.4, scale=1.8, size=n_samples)), 1, 10).astype(int)
    
    # Aptitude test score (0 to 100)
    aptitude_score = np.clip(np.random.normal(loc=65.0, scale=16.0, size=n_samples), 15.0, 99.0).round(1)
    
    # Total technical skills count (languages, frameworks, tools)
    skills_count = np.clip(np.round(coding_skill * 1.2 + num_projects * 0.8 + np.random.normal(0, 1.5, n_samples)), 0, 15).astype(int)
    
    # -------------------------------------------------------------------------
    # 3. Ground Truth Placement & Package Logic (Realistic Weighted Formula)
    # -------------------------------------------------------------------------
    # Normalize features to 0.0 - 1.0 for weighted scoring
    norm_cgpa = (cgpa - 5.0) / (9.8 - 5.0)
    norm_coding = (coding_skill - 1.0) / 9.0
    norm_aptitude = aptitude_score / 100.0
    norm_projects = num_projects / 6.0
    norm_comm = (communication_skill - 1.0) / 9.0
    
    exp_boost = np.where(experience_type == "Work Experience", 0.14,
                np.where(experience_type == "Internship", 0.08, 0.0))
    
    backlog_penalty = backlogs * 0.16
    
    # Add small Gaussian noise so models learn statistical patterns, not pure rules
    noise = np.random.normal(0, 0.07, size=n_samples)
    
    # Overall placement index (Formula weights reflect campus hiring priorities)
    placement_score = (
        0.28 * norm_cgpa +
        0.26 * norm_coding +
        0.18 * norm_aptitude +
        0.12 * norm_projects +
        0.08 * norm_comm +
        exp_boost -
        backlog_penalty +
        noise
    )
    
    # Students with placement score >= 0.42 and <= 2 backlogs are typically placed
    placed = ((placement_score >= 0.42) & (backlogs <= 2)).astype(int)
    
    # Occasionally, a high-coding student with 1-2 backlogs gets placed (special cases)
    edge_cases = (coding_skill >= 8) & (aptitude_score >= 70) & (backlogs <= 2)
    placed = np.where(edge_cases, 1, placed)
    
    # -------------------------------------------------------------------------
    # 4. Company Tier Assignment (Only For Placed Students)
    #    Tier 0: Not Placed
    #    Tier 1: Product / FAANG / High-Growth Startups (Highest Package)
    #    Tier 2: Mid-size Tech / Product Consulting
    #    Tier 3: Service / Mass Recruiters
    # -------------------------------------------------------------------------
    company_tier = np.zeros(n_samples, dtype=int)
    package_lpa = np.zeros(n_samples, dtype=float)
    
    for i in range(n_samples):
        if placed[i] == 1:
            # Composite skill index for tier classification
            tier_score = (
                0.35 * norm_coding[i] +
                0.25 * norm_cgpa[i] +
                0.20 * norm_aptitude[i] +
                0.12 * norm_projects[i] +
                0.08 * norm_comm[i] +
                np.random.normal(0, 0.05)
            )
            
            # Tier 1 Requirements: high coding, good CGPA, low backlogs
            if tier_score >= 0.68 and coding_skill[i] >= 7 and backlogs[i] == 0:
                company_tier[i] = 3  # Tier 1
                # Package between 14.0 LPA and 34.0 LPA
                base_pkg = 14.0 + (coding_skill[i] - 7) * 3.2 + (cgpa[i] - 7.5) * 2.5 + num_projects[i] * 1.0
                package_lpa[i] = round(float(np.clip(base_pkg + np.random.normal(0, 1.5), 14.0, 36.0)), 2)
            
            # Tier 2 Requirements: moderate to good coding & aptitude
            elif tier_score >= 0.46 and coding_skill[i] >= 5 and backlogs[i] <= 1:
                company_tier[i] = 2  # Tier 2
                # Package between 7.0 LPA and 13.5 LPA
                base_pkg = 7.0 + (coding_skill[i] - 5) * 1.2 + (aptitude_score[i] - 55) * 0.08
                package_lpa[i] = round(float(np.clip(base_pkg + np.random.normal(0, 0.8), 7.0, 13.5)), 2)
            
            # Tier 3 Requirements: baseline clearance
            else:
                company_tier[i] = 1  # Tier 3
                # Package between 3.3 LPA and 6.5 LPA
                base_pkg = 3.6 + (cgpa[i] - 6.0) * 0.5 + (aptitude_score[i] - 50) * 0.03
                package_lpa[i] = round(float(np.clip(base_pkg + np.random.normal(0, 0.4), 3.2, 6.8)), 2)
        else:
            company_tier[i] = 0
            package_lpa[i] = 0.0
            
    df = pd.DataFrame({
        "tenth_percent": tenth_percent,
        "twelfth_percent": twelfth_percent,
        "cgpa": cgpa,
        "backlogs": backlogs,
        "branch": branch,
        "experience_type": experience_type,
        "num_projects": num_projects,
        "num_certifications": num_certifications,
        "coding_skill": coding_skill,
        "communication_skill": communication_skill,
        "aptitude_score": aptitude_score,
        "skills_count": skills_count,
        "placed": placed,
        "company_tier": company_tier,
        "package_lpa": package_lpa
    })
    
    return df

def generate_companies_dataset() -> pd.DataFrame:
    """
    Generates a table of ~25 realistic campus recruiting companies
    spanning Tier 1, Tier 2, Tier 3 and 7 prominent tech domains.
    """
    companies_data = [
        # --- Tier 1 Companies (Product / High Package) ---
        {
            "company": "Google",
            "domain": "AI/ML",
            "tier": "Tier 1",
            "min_cgpa": 8.0,
            "min_10th": 75.0,
            "min_12th": 75.0,
            "max_backlogs": 0,
            "required_skills": "python|machine learning|c++|data structures|algorithms|deep learning",
            "min_lpa": 24.0,
            "max_lpa": 36.0
        },
        {
            "company": "Microsoft",
            "domain": "Cloud/DevOps",
            "tier": "Tier 1",
            "min_cgpa": 7.8,
            "min_10th": 70.0,
            "min_12th": 70.0,
            "max_backlogs": 0,
            "required_skills": "azure|c++|c#|docker|kubernetes|data structures",
            "min_lpa": 20.0,
            "max_lpa": 32.0
        },
        {
            "company": "Amazon",
            "domain": "Web Dev",
            "tier": "Tier 1",
            "min_cgpa": 7.5,
            "min_10th": 70.0,
            "min_12th": 70.0,
            "max_backlogs": 0,
            "required_skills": "java|aws|react|data structures|distributed systems|system design",
            "min_lpa": 18.0,
            "max_lpa": 28.0
        },
        {
            "company": "Adobe",
            "domain": "Web Dev",
            "tier": "Tier 1",
            "min_cgpa": 7.5,
            "min_10th": 70.0,
            "min_12th": 70.0,
            "max_backlogs": 0,
            "required_skills": "javascript|react|node.js|c++|data structures|algorithms",
            "min_lpa": 16.0,
            "max_lpa": 26.0
        },
        {
            "company": "Flipkart",
            "domain": "Data Science",
            "tier": "Tier 1",
            "min_cgpa": 7.5,
            "min_10th": 65.0,
            "min_12th": 65.0,
            "max_backlogs": 0,
            "required_skills": "python|sql|pandas|machine learning|numpy|statistics",
            "min_lpa": 15.0,
            "max_lpa": 24.0
        },
        {
            "company": "Cisco",
            "domain": "Cyber Security",
            "tier": "Tier 1",
            "min_cgpa": 7.5,
            "min_10th": 70.0,
            "min_12th": 70.0,
            "max_backlogs": 0,
            "required_skills": "networking|python|c++|ethical hacking|linux|cyber security",
            "min_lpa": 14.0,
            "max_lpa": 22.0
        },
        {
            "company": "Qualcomm",
            "domain": "Core Engineering",
            "tier": "Tier 1",
            "min_cgpa": 7.8,
            "min_10th": 70.0,
            "min_12th": 70.0,
            "max_backlogs": 0,
            "required_skills": "c++|c|embedded systems|verilog|linux|microcontrollers",
            "min_lpa": 15.0,
            "max_lpa": 25.0
        },

        # --- Tier 2 Companies (Mid-size Tech / Product Consulting) ---
        {
            "company": "Oracle",
            "domain": "Cloud/DevOps",
            "tier": "Tier 2",
            "min_cgpa": 7.0,
            "min_10th": 65.0,
            "min_12th": 65.0,
            "max_backlogs": 1,
            "required_skills": "java|sql|linux|docker|cloud|database management",
            "min_lpa": 9.0,
            "max_lpa": 13.5
        },
        {
            "company": "SAP Labs",
            "domain": "Web Dev",
            "tier": "Tier 2",
            "min_cgpa": 7.0,
            "min_10th": 65.0,
            "min_12th": 65.0,
            "max_backlogs": 1,
            "required_skills": "javascript|react|sql|java|rest api|git",
            "min_lpa": 8.5,
            "max_lpa": 13.0
        },
        {
            "company": "Zomato",
            "domain": "Web Dev",
            "tier": "Tier 2",
            "min_cgpa": 6.8,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "python|django|react|sql|git|docker",
            "min_lpa": 8.0,
            "max_lpa": 12.5
        },
        {
            "company": "JPMorgan Chase",
            "domain": "Data Science",
            "tier": "Tier 2",
            "min_cgpa": 7.2,
            "min_10th": 65.0,
            "min_12th": 65.0,
            "max_backlogs": 0,
            "required_skills": "python|sql|excel|power bi|pandas|financial modeling",
            "min_lpa": 10.0,
            "max_lpa": 14.0
        },
        {
            "company": "Palo Alto Networks",
            "domain": "Cyber Security",
            "tier": "Tier 2",
            "min_cgpa": 7.0,
            "min_10th": 65.0,
            "min_12th": 65.0,
            "max_backlogs": 0,
            "required_skills": "networking|linux|python|firewalls|cyber security|cryptography",
            "min_lpa": 10.0,
            "max_lpa": 13.5
        },
        {
            "company": "Mu Sigma",
            "domain": "Data Science",
            "tier": "Tier 2",
            "min_cgpa": 6.5,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "python|sql|statistics|machine learning|excel|visualization",
            "min_lpa": 7.5,
            "max_lpa": 10.0
        },
        {
            "company": "ThoughtWorks",
            "domain": "Web Dev",
            "tier": "Tier 2",
            "min_cgpa": 6.5,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "java|javascript|clean code|tdd|react|spring boot",
            "min_lpa": 8.0,
            "max_lpa": 12.0
        },
        {
            "company": "L&T Technology Services",
            "domain": "Core Engineering",
            "tier": "Tier 2",
            "min_cgpa": 6.8,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "autocad|solidworks|c|matlab|embedded systems|mechanical design",
            "min_lpa": 6.5,
            "max_lpa": 9.0
        },
        {
            "company": "BrowserStack",
            "domain": "Testing/QA",
            "tier": "Tier 2",
            "min_cgpa": 6.8,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "selenium|java|python|automation testing|api testing|jenkins",
            "min_lpa": 8.0,
            "max_lpa": 12.0
        },

        # --- Tier 3 Companies (Service / Mass Recruiters) ---
        {
            "company": "Tata Consultancy Services (TCS Ninja)",
            "domain": "Web Dev",
            "tier": "Tier 3",
            "min_cgpa": 6.0,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "c|java|python|sql|html|css",
            "min_lpa": 3.6,
            "max_lpa": 4.5
        },
        {
            "company": "TCS Digital",
            "domain": "AI/ML",
            "tier": "Tier 3",
            "min_cgpa": 6.5,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 0,
            "required_skills": "python|machine learning|sql|cloud|data structures",
            "min_lpa": 7.0,
            "max_lpa": 7.5
        },
        {
            "company": "Infosys",
            "domain": "Web Dev",
            "tier": "Tier 3",
            "min_cgpa": 6.0,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "java|python|sql|javascript|html|css",
            "min_lpa": 3.6,
            "max_lpa": 5.0
        },
        {
            "company": "Wipro",
            "domain": "Testing/QA",
            "tier": "Tier 3",
            "min_cgpa": 6.0,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "c++|java|manual testing|sql|software testing",
            "min_lpa": 3.5,
            "max_lpa": 4.5
        },
        {
            "company": "Cognizant",
            "domain": "Cloud/DevOps",
            "tier": "Tier 3",
            "min_cgpa": 6.0,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "cloud basics|python|linux|networking|sql",
            "min_lpa": 4.0,
            "max_lpa": 5.5
        },
        {
            "company": "Capgemini",
            "domain": "Web Dev",
            "tier": "Tier 3",
            "min_cgpa": 6.0,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "c|c++|java|sql|javascript|web development",
            "min_lpa": 4.0,
            "max_lpa": 5.0
        },
        {
            "company": "Accenture",
            "domain": "Web Dev",
            "tier": "Tier 3",
            "min_cgpa": 6.0,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "java|sql|agile|c++|python|communication",
            "min_lpa": 4.5,
            "max_lpa": 6.5
        },
        {
            "company": "HCL Technologies",
            "domain": "Testing/QA",
            "tier": "Tier 3",
            "min_cgpa": 5.8,
            "min_10th": 55.0,
            "min_12th": 55.0,
            "max_backlogs": 2,
            "required_skills": "manual testing|c|sql|linux|jira",
            "min_lpa": 3.2,
            "max_lpa": 4.2
        },
        {
            "company": "Tech Mahindra",
            "domain": "Cyber Security",
            "tier": "Tier 3",
            "min_cgpa": 6.0,
            "min_10th": 60.0,
            "min_12th": 60.0,
            "max_backlogs": 1,
            "required_skills": "networking|linux|information security|firewalls|python",
            "min_lpa": 3.6,
            "max_lpa": 5.0
        }
    ]
    return pd.DataFrame(companies_data)

if __name__ == "__main__":
    # Ensure target data directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    print("[1/2] Generating synthetic student dataset (3,000 records)...")
    students_df = generate_student_dataset(n_samples=3000)
    students_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "students.csv")
    students_df.to_csv(students_path, index=False)
    print(f" -> Saved: {students_path}")
    print(f" -> Placed count: {students_df['placed'].sum()} / {len(students_df)} ({students_df['placed'].mean()*100:.1f}%)")
    print(f" -> Tier counts:\n{students_df['company_tier'].value_counts().sort_index()}")
    
    print("\n[2/2] Generating company criteria table (~25 recruiters)...")
    companies_df = generate_companies_dataset()
    companies_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companies.csv")
    companies_df.to_csv(companies_path, index=False)
    print(f" -> Saved: {companies_path}")
    print(f" -> Recruiter count: {len(companies_df)} companies across tiers and domains.")
    print("\nData generation completed successfully!")
