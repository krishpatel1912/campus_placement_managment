"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: utils/company_matcher.py
Purpose: Rule-based eligibility validator and KNN tier-guided company matcher.
=============================================================================

NOTE FOR STUDENTS & EXAMINERS:
Company eligibility evaluation is deterministic (rule-based) because corporate
recruitment policies enforce strict minimum cutoffs. However, we fuse this with
our machine learning KNN tier classification to rank and recommend the
"Best Fit Companies" that align with the student's competitive peer level.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPANIES_CSV = os.path.join(BASE_DIR, "data", "companies.csv")

def load_companies() -> pd.DataFrame:
    """
    Loads company criteria catalog from data/companies.csv.
    """
    if not os.path.exists(COMPANIES_CSV):
        raise FileNotFoundError(f"Companies file not found at {COMPANIES_CSV}. Run data/generate_dataset.py first.")
    return pd.read_csv(COMPANIES_CSV)

def calculate_skill_overlap(student_skills: list, company_req_skills_str: str):
    """
    Computes percentage of required company skills satisfied by the student.
    Returns: (match_percentage, matching_skills_list, missing_skills_list)
    """
    if not company_req_skills_str or pd.isna(company_req_skills_str):
        return 100.0, [], []
        
    req_skills = [s.strip().lower() for s in str(company_req_skills_str).split("|") if s.strip()]
    student_skills_clean = [s.strip().lower() for s in student_skills if s and str(s).strip()]
    
    matching = []
    missing = []
    
    for req in req_skills:
        found = False
        for s in student_skills_clean:
            if s in req or req in s:
                found = True
                break
        if found:
            matching.append(req)
        else:
            missing.append(req)
            
    match_pct = (len(matching) / len(req_skills) * 100.0) if req_skills else 100.0
    return round(match_pct, 1), matching, missing

def evaluate_company_eligibility(student_dict: dict, predicted_tier_code: int = None) -> dict:
    """
    Evaluates student credentials against all campus recruiting companies.
    
    Categorizes each company into:
    1. Eligible: Passes all academic cutoffs AND has >= 40% skill match.
    2. Almost Eligible: Narrowly misses cutoff (e.g. CGPA within 0.5, 1 extra backlog, or 25-40% skills).
    3. Not Eligible: Misses hard criteria significantly with specific rejection reasons.
    
    Also identifies "Best Fit Companies" using the predicted KNN tier code.
    """
    df_companies = load_companies()
    
    student_cgpa = float(student_dict.get("cgpa", 0.0))
    student_10th = float(student_dict.get("tenth_percent", 0.0))
    student_12th = float(student_dict.get("twelfth_percent", 0.0))
    student_backlogs = int(student_dict.get("backlogs", 0))
    student_skills = student_dict.get("skills", [])
    
    eligible = []
    almost_eligible = []
    not_eligible = []
    
    for _, comp in df_companies.iterrows():
        min_cgpa = float(comp["min_cgpa"])
        min_10th = float(comp["min_10th"])
        min_12th = float(comp["min_12th"])
        max_backlogs = int(comp["max_backlogs"])
        
        match_pct, matching, missing = calculate_skill_overlap(student_skills, comp["required_skills"])
        
        # Check reasons for failure
        failure_reasons = []
        near_miss_reasons = []
        
        # CGPA check
        if student_cgpa < min_cgpa:
            gap = min_cgpa - student_cgpa
            if gap <= 0.4:
                near_miss_reasons.append(f"CGPA is {student_cgpa:.2f} (Needs {min_cgpa:.1f}, gap: {gap:.2f})")
            else:
                failure_reasons.append(f"CGPA {student_cgpa:.2f} < Minimum {min_cgpa:.1f}")
                
        # 10th percent check
        if student_10th < min_10th:
            gap = min_10th - student_10th
            if gap <= 5.0:
                near_miss_reasons.append(f"10th score is {student_10th:.1f}% (Needs {min_10th:.1f}%)")
            else:
                failure_reasons.append(f"10th score {student_10th:.1f}% < Minimum {min_10th:.1f}%")
                
        # 12th percent check
        if student_12th < min_12th:
            gap = min_12th - student_12th
            if gap <= 5.0:
                near_miss_reasons.append(f"12th score is {student_12th:.1f}% (Needs {min_12th:.1f}%)")
            else:
                failure_reasons.append(f"12th score {student_12th:.1f}% < Minimum {min_12th:.1f}%")
                
        # Backlogs check
        if student_backlogs > max_backlogs:
            if student_backlogs == max_backlogs + 1:
                near_miss_reasons.append(f"Has {student_backlogs} backlog(s) (Maximum allowed is {max_backlogs})")
            else:
                failure_reasons.append(f"Active backlogs ({student_backlogs}) exceed cutoff ({max_backlogs})")
                
        # Skill check
        if match_pct < 40.0:
            if match_pct >= 20.0:
                near_miss_reasons.append(f"Skill match is {match_pct}% (Missing: {', '.join(missing[:3])})")
            else:
                failure_reasons.append(f"Low skill match ({match_pct}%)")
                
        item = {
            "company": comp["company"],
            "domain": comp["domain"],
            "tier": comp["tier"],
            "tier_num": 3 if comp["tier"] == "Tier 1" else (2 if comp["tier"] == "Tier 2" else 1),
            "package_range": f"{comp['min_lpa']:.1f} - {comp['max_lpa']:.1f} LPA",
            "skill_match_pct": match_pct,
            "matching_skills": matching,
            "missing_skills": missing,
            "min_cgpa": min_cgpa,
            "max_backlogs": max_backlogs
        }
        
        # Classification into the 3 buckets
        if len(failure_reasons) == 0 and len(near_miss_reasons) == 0:
            eligible.append(item)
        elif len(failure_reasons) == 0 and len(near_miss_reasons) > 0:
            item["missing_criteria"] = near_miss_reasons
            almost_eligible.append(item)
        else:
            item["reasons"] = failure_reasons + near_miss_reasons
            not_eligible.append(item)
            
    # Sort eligible by tier descending (Tier 1 first) then by skill match descending
    eligible.sort(key=lambda x: (x["tier_num"], x["skill_match_pct"]), reverse=True)
    almost_eligible.sort(key=lambda x: (x["tier_num"], x["skill_match_pct"]), reverse=True)
    not_eligible.sort(key=lambda x: (x["tier_num"], x["skill_match_pct"]), reverse=True)
    
    # Identify "Best Fit Companies" (aligning with student's predicted KNN tier)
    best_fit = []
    if predicted_tier_code is not None:
        target_tier_str = f"Tier {4 - predicted_tier_code}" if predicted_tier_code in [1, 2, 3] else None
        # Note: In our model: 3=Tier1, 2=Tier2, 1=Tier3
        tier_mapping = {3: "Tier 1", 2: "Tier 2", 1: "Tier 3"}
        target_str = tier_mapping.get(predicted_tier_code, "Tier 3")
        
        for c in eligible:
            if c["tier"] == target_str:
                best_fit.append(c)
        # If not enough in eligible, check almost eligible
        if len(best_fit) < 2:
            for c in almost_eligible:
                if c["tier"] == target_str:
                    best_fit.append(c)
                    
    return {
        "eligible": eligible,
        "almost_eligible": almost_eligible,
        "not_eligible": not_eligible,
        "best_fit": best_fit[:5]
    }
