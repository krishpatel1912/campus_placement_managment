"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: utils/resume_parser.py
Purpose: Extracts text from uploaded PDF / DOCX resumes and extracts skills,
         experience level, project counts, and certification tallies.
=============================================================================

NOTE FOR STUDENTS & EXAMINERS:
This module relies on rule-based string search and regular expressions (Regex)
rather than complex Deep Learning Natural Language Processing (NLP).
Rule-based keyword extraction is transparent, fast, deterministic, and easy
for an undergraduate to justify during viva voce.
"""

import re
import os
import io
import pdfplumber

# Try importing docx, handle gracefully if missing
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Comprehensive tech skill keyword list (case-insensitive search)
SKILL_KEYWORDS = [
    # Programming Languages
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "php", "ruby", "go",
    # Web & Frameworks
    "html", "css", "react", "react.js", "angular", "vue", "node.js", "express", 
    "django", "flask", "spring boot", "fastapi", "tailwind", "bootstrap",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis",
    # Data Science & Machine Learning
    "machine learning", "deep learning", "pandas", "numpy", "scikit-learn", 
    "tensorflow", "pytorch", "power bi", "tableau", "excel", "data analysis",
    # Cloud, DevOps & Tools
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "linux", "jira",
    # Testing & Security
    "selenium", "automation testing", "manual testing", "ethical hacking", 
    "cyber security", "networking", "wireshark", "nmap", "cryptography"
]

def extract_text_from_pdf(file_bytes) -> str:
    """
    Extracts concatenated text across all pages of a PDF using pdfplumber.
    """
    text_content = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return ""
    return "\n".join(text_content)

def extract_text_from_docx(file_bytes) -> str:
    """
    Extracts text paragraphs from a Word (.docx) document using python-docx.
    """
    if not DOCX_AVAILABLE:
        return ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Error parsing DOCX: {e}")
        return ""

def parse_resume(file_bytes, filename: str) -> dict:
    """
    Extracts structured credentials from a candidate resume file:
    - Detected technical skills
    - Inferred experience level (Fresher, Internship, or Work Experience)
    - Estimated number of completed projects
    - Estimated number of certifications
    - Heuristic estimate for coding skill rating (1-10)
    
    Parameters:
        file_bytes (bytes): Raw binary file content.
        filename (str): Name of the uploaded file (e.g. 'resume.pdf').
        
    Returns:
        dict: Parsed metrics to prepopulate the Streamlit prediction form.
    """
    text = ""
    lower_fn = filename.lower()
    
    # 1. Extract raw text based on extension
    if lower_fn.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif lower_fn.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    else:
        # Fallback treat as UTF-8 plain text
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
            
    lower_text = text.lower()
    
    # 2. Extract Skills (Keyword Matching)
    found_skills = set()
    for skill in SKILL_KEYWORDS:
        # Use regex word boundaries for short words like "c", "r", "go", "sql"
        if len(skill) <= 3:
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, lower_text):
                found_skills.add(skill.title() if len(skill) > 2 else skill.upper())
        else:
            if skill in lower_text:
                found_skills.add(skill.title() if len(skill) > 3 else skill.upper())
                
    skills_list = sorted(list(found_skills))
    skills_count = len(skills_list)
    
    # 3. Detect Experience Type
    # Look for keywords indicating full-time work vs internships vs fresher
    work_exp_matches = len(re.findall(r"\b(work experience|full[\s-]time|software engineer|developer at|employed)\b", lower_text))
    intern_matches = len(re.findall(r"\b(intern|internship|trainee|summer intern)\b", lower_text))
    
    if work_exp_matches >= 2:
        experience_type = "Work Experience"
    elif intern_matches >= 1:
        experience_type = "Internship"
    else:
        experience_type = "Fresher"
        
    # 4. Estimate Project Count
    # Count occurrences of bullet points or project headers under project sections
    project_matches = re.findall(r"\b(project|projects|developed|built|created)\b", lower_text)
    github_matches = len(re.findall(r"github\.com", lower_text))
    
    # Heuristic estimation: usually students have 1 to 5 projects
    raw_proj_estimate = len(project_matches) // 3 + github_matches
    num_projects = int(min(max(raw_proj_estimate, 1 if "project" in lower_text else 0), 6))
    
    # 5. Estimate Certification Count
    cert_matches = len(re.findall(r"\b(certificate|certification|certified|course|coursera|udemy|nptel|hackerrank|credential)\b", lower_text))
    num_certifications = int(min(max(cert_matches // 2, 0), 8))
    
    # 6. Estimate Coding Skill (1 to 10 scale based on skill depth and project count)
    # Higher count of advanced skills suggests higher rating
    advanced_skills = {"python", "java", "c++", "react", "docker", "machine learning", "kubernetes", "sql"}
    advanced_hits = sum(1 for s in skills_list if s.lower() in advanced_skills)
    estimated_coding = int(min(max(3 + advanced_hits + (num_projects // 2), 1), 10))
    
    return {
        "raw_text_length": len(text),
        "skills": skills_list,
        "skills_count": skills_count,
        "experience_type": experience_type,
        "num_projects": num_projects,
        "num_certifications": num_certifications,
        "estimated_coding_skill": estimated_coding,
        "summary": (
            f"Extracted {skills_count} skills, detected {experience_type} background, "
            f"~{num_projects} projects, and ~{num_certifications} certifications."
        )
    }
