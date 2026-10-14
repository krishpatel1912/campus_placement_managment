"""
=============================================================================
Campus Placement Predictor & Career Guide
Module: utils/domain_roadmap.py
Purpose: Structured learning roadmaps, interview focus areas, and skill-gap
         analysis for 7 major technical placement tracks.
=============================================================================
"""

DOMAINS_DATA = {
    "Web Dev": {
        "title": "Full-Stack Web Development",
        "description": "Building modern responsive web applications, REST APIs, and database-backed platforms.",
        "must_have_skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "SQL", "Git"],
        "tools": ["VS Code", "Postman", "GitHub", "Vercel / Netlify", "Docker"],
        "roadmap_steps": [
            {"level": "1. Foundation (Beginner)", "topic": "HTML5 Semantic Tags & CSS3 Flexbox/Grid"},
            {"level": "1. Foundation (Beginner)", "topic": "Modern JavaScript (ES6+, Promises, Async/Await, DOM manipulation)"},
            {"level": "2. Frontend Engineering (Intermediate)", "topic": "React.js: Hooks, State Management, Component Lifecycle"},
            {"level": "2. Frontend Engineering (Intermediate)", "topic": "Responsive UI frameworks (Tailwind CSS, Bootstrap)"},
            {"level": "3. Backend & APIs (Intermediate)", "topic": "Node.js / Express.js or Python Django / Flask"},
            {"level": "3. Backend & APIs (Intermediate)", "topic": "RESTful API Design & HTTP status codes"},
            {"level": "4. Databases & Storage (Advanced)", "topic": "Relational DBs (PostgreSQL / MySQL) & NoSQL (MongoDB)"},
            {"level": "4. Databases & Storage (Advanced)", "topic": "Authentication (JWT, OAuth) & Security practices"},
            {"level": "5. Deployment & DevOps (Advanced)", "topic": "CI/CD basics, Docker containers & Cloud hosting"}
        ],
        "interview_topics": [
            "Explain Event Loop, Closures, and Prototypal Inheritance in JavaScript.",
            "Difference between Virtual DOM and Real DOM in React.",
            "REST vs GraphQL: Advantages and Trade-offs.",
            "How does HTTPS handshake and SSL encryption work?",
            "Database normalization vs denormalization with examples."
        ],
        "suggested_projects": [
            "Full-Stack E-Commerce Store with payment gateway integration (Stripe/Razorpay)",
            "Collaborative Kanban Board (Trello clone) with WebSockets real-time updates",
            "Developer Portfolio & Blog with Markdown parsing and CMS API",
            "Role-Based Campus Placement Portal with Admin/Student dashboards"
        ]
    },
    "Data Science": {
        "title": "Data Science & Analytics",
        "description": "Extracting actionable insights from data through statistical analysis, SQL queries, and visualization.",
        "must_have_skills": ["Python", "SQL", "Pandas", "NumPy", "Statistics", "Power BI / Tableau", "Excel"],
        "tools": ["Jupyter Notebook", "DBeaver / pgAdmin", "Power BI Desktop", "Git"],
        "roadmap_steps": [
            {"level": "1. Foundation (Beginner)", "topic": "Python Programming (Data structures, Functions, OOPs)"},
            {"level": "1. Foundation (Beginner)", "topic": "Advanced SQL (Aggregations, Window Functions, Complex Joins, CTEs)"},
            {"level": "2. Data Wrangling (Intermediate)", "topic": "Pandas for data cleaning, merging, and reshaping"},
            {"level": "2. Data Wrangling (Intermediate)", "topic": "NumPy for matrix operations and numerical computing"},
            {"level": "3. Mathematics & Stats (Intermediate)", "topic": "Descriptive & Inferential Statistics, Hypothesis Testing (p-value, t-test)"},
            {"level": "3. Mathematics & Stats (Intermediate)", "topic": "Probability Distributions & Bayes Theorem"},
            {"level": "4. Visualization & BI (Advanced)", "topic": "Matplotlib, Seaborn, and Plotly interactive charts"},
            {"level": "4. Visualization & BI (Advanced)", "topic": "Business Intelligence Dashboards in Power BI / Tableau"},
            {"level": "5. Business Analytics (Advanced)", "topic": "A/B Testing, Cohort Analysis, and Churn modeling"}
        ],
        "interview_topics": [
            "Difference between INNER JOIN, LEFT JOIN, and FULL OUTER JOIN.",
            "Explain Central Limit Theorem and why it matters in hypothesis testing.",
            "How do you handle missing values and outliers in a skewed dataset?",
            "What is p-value and Type I vs Type II errors?",
            "Difference between OLTP and OLAP systems."
        ],
        "suggested_projects": [
            "End-to-End Customer Churn Analysis & Interactive Power BI Dashboard",
            "E-Commerce Sales EDA & Cohort Retention Analysis using Pandas",
            "Stock Market Financial Trend Analyzer with Plotly Visuals",
            "Healthcare Patient Readmission Statistical Analysis"
        ]
    },
    "AI/ML": {
        "title": "Artificial Intelligence & Machine Learning",
        "description": "Developing predictive models, supervised/unsupervised learning pipelines, and deploying ML systems.",
        "must_have_skills": ["Python", "Scikit-Learn", "Machine Learning", "Mathematics", "Pandas", "NumPy", "Deep Learning basics"],
        "tools": ["Jupyter Lab", "Google Colab", "Streamlit", "MLflow", "GitHub"],
        "roadmap_steps": [
            {"level": "1. Math Foundations (Beginner)", "topic": "Linear Algebra (Vectors, Matrices, Eigenvalues) & Calculus (Derivatives, Gradients)"},
            {"level": "1. Math Foundations (Beginner)", "topic": "Probability & Statistical Inference"},
            {"level": "2. Supervised Learning (Intermediate)", "topic": "Regression: Linear, Ridge, Lasso, Polynomial"},
            {"level": "2. Supervised Learning (Intermediate)", "topic": "Classification: Logistic Regression, KNN, SVM, Decision Trees, Random Forest, XGBoost"},
            {"level": "3. Unsupervised Learning (Intermediate)", "topic": "Clustering: K-Means, DBSCAN, Hierarchical Clustering"},
            {"level": "3. Unsupervised Learning (Intermediate)", "topic": "Dimensionality Reduction: PCA, t-SNE"},
            {"level": "4. Model Tuning & Evaluation (Advanced)", "topic": "Cross-validation, Hyperparameter tuning (GridSearchCV), ROC-AUC, PR-AUC"},
            {"level": "4. Model Tuning & Evaluation (Advanced)", "topic": "Feature Engineering, Ensembling & Pipelines"},
            {"level": "5. Deep Learning & Deployment (Advanced)", "topic": "Neural Networks (ANN, CNN basics) & Model Deployment via FastAPI/Streamlit"}
        ],
        "interview_topics": [
            "Explain Bias-Variance tradeoff and how to mitigate overfitting.",
            "Mathematical formulation of Gradient Descent and Learning Rate.",
            "Precision vs Recall vs F1-score: when to prioritize which?",
            "How does Random Forest reduce variance compared to a single Decision Tree?",
            "Explain the working of K-Means and the Elbow Method."
        ],
        "suggested_projects": [
            "End-to-End Campus Placement Predictor with Multi-Model Comparison",
            "House Price Valuation Regressor with Feature Importance & FastAPI endpoint",
            "Credit Card Fraud Detection with SMOTE and Imbalanced Learning",
            "Sentiment Analysis Web App using Scikit-Learn and Streamlit"
        ]
    },
    "Cyber Security": {
        "title": "Cyber Security & Information Assurance",
        "description": "Defending digital assets, network vulnerability assessment, cryptography, and penetration testing.",
        "must_have_skills": ["Networking", "Linux", "Ethical Hacking", "Python", "Cyber Security", "Cryptography", "Firewalls"],
        "tools": ["Wireshark", "Nmap", "Kali Linux", "Burp Suite", "Metasploit"],
        "roadmap_steps": [
            {"level": "1. Networking Foundations (Beginner)", "topic": "OSI & TCP/IP Models, Subnetting, DNS, DHCP, ARP"},
            {"level": "1. Networking Foundations (Beginner)", "topic": "Linux Command Line Administration & Bash Scripting"},
            {"level": "2. Security Basics (Intermediate)", "topic": "Symmetric vs Asymmetric Encryption, Hashing (SHA-256), Digital Signatures"},
            {"level": "2. Security Basics (Intermediate)", "topic": "Authentication protocols (Kerberos, LDAP, MFA)"},
            {"level": "3. Network Security (Intermediate)", "topic": "Packet Sniffing with Wireshark, Port Scanning with Nmap, IDS/IPS, Firewalls"},
            {"level": "3. Network Security (Intermediate)", "topic": "Common Network Attacks (MITM, ARP Poisoning, DDoS)"},
            {"level": "4. Web Application Security (Advanced)", "topic": "OWASP Top 10 vulnerabilities (SQLi, XSS, CSRF, SSRF, IDOR)"},
            {"level": "4. Web Application Security (Advanced)", "topic": "Hands-on Burp Suite interception & manual exploitation"},
            {"level": "5. SecOps & Compliance (Advanced)", "topic": "SIEM tools (Splunk basics), Incident Response, ISO 27001 standards"}
        ],
        "interview_topics": [
            "Explain TCP 3-Way Handshake and SYN flood attack.",
            "Difference between Encryption, Encoding, and Hashing.",
            "How do SQL Injection and Cross-Site Scripting (XSS) work and how to prevent them?",
            "What is a Public Key Infrastructure (PKI) and SSL Certificate validation?",
            "Explain CIA Triad (Confidentiality, Integrity, Availability) in system design."
        ],
        "suggested_projects": [
            "Automated Python Port Scanner & Vulnerability Detector using Nmap library",
            "Keylogger and Ransomware Simulation Lab (in isolated VM) with defense script",
            "Secure File Vault with AES-256 encryption and PBKDF2 key derivation",
            "Web App Penetration Testing Report on DVWA (Damn Vulnerable Web App)"
        ]
    },
    "Cloud/DevOps": {
        "title": "Cloud Computing & DevOps",
        "description": "Automating software delivery pipelines, container orchestration, and cloud infrastructure management.",
        "must_have_skills": ["Linux", "AWS / Azure", "Docker", "Git", "Kubernetes", "CI/CD", "Python"],
        "tools": ["Docker Desktop", "Kubernetes (Minikube)", "GitHub Actions", "Terraform", "AWS Console"],
        "roadmap_steps": [
            {"level": "1. System Admin (Beginner)", "topic": "Linux Shell scripting, permissions, process management, SSH"},
            {"level": "1. System Admin (Beginner)", "topic": "Git version control branching, merging, and rebasing"},
            {"level": "2. Cloud Fundamentals (Intermediate)", "topic": "AWS core services: EC2, S3, IAM, VPC, RDS, Lambda"},
            {"level": "2. Cloud Fundamentals (Intermediate)", "topic": "Cloud networking, Security Groups, and Elastic Load Balancing"},
            {"level": "3. Containerization (Intermediate)", "topic": "Docker: Dockerfile creation, multi-stage builds, Docker Compose"},
            {"level": "3. Containerization (Intermediate)", "topic": "Container networking, volumes, and persistent storage"},
            {"level": "4. CI/CD Automation (Advanced)", "topic": "Building automated test/deploy pipelines using GitHub Actions or Jenkins"},
            {"level": "4. CI/CD Automation (Advanced)", "topic": "Infrastructure as Code (IaC) basics with Terraform"},
            {"level": "5. Orchestration (Advanced)", "topic": "Kubernetes: Pods, Deployments, Services, ConfigMaps, Ingress controllers"}
        ],
        "interview_topics": [
            "Difference between Virtual Machines and Docker Containers.",
            "Explain the components of Kubernetes architecture (Control Plane & Worker Nodes).",
            "What is Infrastructure as Code (IaC) and what are its benefits?",
            "How does Blue-Green deployment differ from Canary deployment?",
            "Explain AWS S3 storage classes and IAM role vs user vs policy."
        ],
        "suggested_projects": [
            "Multi-Tier Web App Containerization with Docker Compose and Nginx reverse proxy",
            "Automated CI/CD Pipeline on GitHub Actions deploying to AWS EC2",
            "Kubernetes Cluster Deployment with auto-scaling and health check probes",
            "Terraform Script to provision an AWS VPC with public and private subnets"
        ]
    },
    "Testing/QA": {
        "title": "Software Testing & Quality Assurance",
        "description": "Ensuring software reliability through automated test suites, functional testing, and performance benchmarks.",
        "must_have_skills": ["Selenium", "Java / Python", "Automation Testing", "SQL", "Manual Testing", "API Testing", "Git"],
        "tools": ["Selenium WebDriver", "Postman", "Jira", "JUnit / TestNG / PyTest", "Jenkins"],
        "roadmap_steps": [
            {"level": "1. Testing Fundamentals (Beginner)", "topic": "Software Testing Life Cycle (STLC), Bug Life Cycle, Test Case Design"},
            {"level": "1. Testing Fundamentals (Beginner)", "topic": "Black-box techniques: Boundary Value Analysis, Equivalence Partitioning"},
            {"level": "2. Programming for QA (Intermediate)", "topic": "Core Java or Python OOP concepts, Collections, Exception Handling"},
            {"level": "2. Programming for QA (Intermediate)", "topic": "Database testing with SQL queries for verification"},
            {"level": "3. UI Automation (Intermediate)", "topic": "Selenium WebDriver: Locators (XPath, CSS), Waits (Implicit, Explicit)"},
            {"level": "3. UI Automation (Intermediate)", "topic": "Page Object Model (POM) design pattern"},
            {"level": "4. API Testing (Advanced)", "topic": "REST API testing with Postman and automated RestAssured / Requests library"},
            {"level": "4. API Testing (Advanced)", "topic": "JSON schema validation and status code assertion"},
            {"level": "5. Frameworks & CI/CD (Advanced)", "topic": "TestNG / PyTest reporting, parallel execution, and Jenkins integration"}
        ],
        "interview_topics": [
            "Difference between Severity and Priority of a defect with examples.",
            "Explain Page Object Model (POM) and why it prevents code duplication.",
            "Implicit Wait vs Explicit Wait vs Fluent Wait in Selenium.",
            "Difference between Functional Testing, Regression Testing, and Smoke Testing.",
            "How do you test a REST API with POST request and JSON payload?"
        ],
        "suggested_projects": [
            "Data-Driven Selenium Automation Framework with Page Object Model and TestNG reports",
            "Automated REST API Test Suite for E-Commerce endpoints using Python Requests & PyTest",
            "End-to-End Test Suite for Campus Management System with Jira defect logging",
            "Performance Benchmark Test Suite using Apache JMeter"
        ]
    },
    "Core Engineering": {
        "title": "Core Engineering (ECE / MECH / CIVIL / EEE)",
        "description": "Embedded hardware, IoT firmware, automotive design, structural analysis, and control systems.",
        "must_have_skills": ["C / C++", "MATLAB", "Embedded Systems", "AutoCAD / SolidWorks", "Linux", "Microcontrollers"],
        "tools": ["Arduino IDE", "MATLAB & Simulink", "AutoCAD", "SolidWorks", "Keil uVision"],
        "roadmap_steps": [
            {"level": "1. Core Principles (Beginner)", "topic": "Domain fundamentals (Circuit analysis / Mechanics / Thermodynamics / Structures)"},
            {"level": "1. Core Principles (Beginner)", "topic": "C Programming & bitwise operations for hardware control"},
            {"level": "2. Modeling & CAD (Intermediate)", "topic": "Computer-Aided Design (AutoCAD, SolidWorks, CATIA, or PCB Eagle)"},
            {"level": "2. Modeling & CAD (Intermediate)", "topic": "Mathematical modeling & simulation with MATLAB"},
            {"level": "3. Embedded Systems (Intermediate)", "topic": "Microcontrollers (ARM Cortex, AVR, PIC, ESP32, Arduino)"},
            {"level": "3. Embedded Systems (Intermediate)", "topic": "Communication protocols: I2C, SPI, UART, CAN bus"},
            {"level": "4. Real-Time Systems (Advanced)", "topic": "FreeRTOS basics, interrupt service routines, sensors & actuators interfacing"},
            {"level": "4. Real-Time Systems (Advanced)", "topic": "Signal Processing & Feedback Control Systems (PID tuning)"},
            {"level": "5. Industrial Applications (Advanced)", "topic": "IoT Cloud integration (MQTT), PLC automation, and ISO engineering standards"}
        ],
        "interview_topics": [
            "Explain I2C vs SPI communication protocols: speed, pin count, master-slave.",
            "What is an Interrupt Service Routine (ISR) and rules for writing an ISR in C?",
            "Difference between Von Neumann and Harvard microcontroller architectures.",
            "Explain Stress-Strain curve and Yield Strength with practical significance.",
            "What is Pulse Width Modulation (PWM) and how is it used to control motor speed?"
        ],
        "suggested_projects": [
            "IoT-Enabled Smart Energy Meter with ESP32, MQTT protocol and web dashboard",
            "Autonomous Obstacle Avoiding Robot with Ultrasonic sensors and Arduino PID controller",
            "3D Structural Simulation and Stress Analysis of a Bridge Truss in SolidWorks/ANSYS",
            "Digital Signal Filter Implementation in MATLAB & Simulink"
        ]
    }
}

def analyze_skill_gaps(target_domain: str, student_skills: list) -> dict:
    """
    Compares the student's acquired skills against the target domain requirements.
    Calculates coverage percentage, lists missing must-haves and remaining topics.
    """
    if target_domain not in DOMAINS_DATA:
        target_domain = "Web Dev"
        
    domain_info = DOMAINS_DATA[target_domain]
    must_haves = domain_info["must_have_skills"]
    
    # Normalize student skills for case-insensitive matching
    normalized_student_skills = set(s.strip().lower() for s in student_skills if s)
    
    covered_skills = []
    missing_skills = []
    
    for req in must_haves:
        req_norm = req.lower()
        matched = False
        for s in normalized_student_skills:
            if s in req_norm or req_norm in s:
                matched = True
                break
        if matched:
            covered_skills.append(req)
        else:
            missing_skills.append(req)
            
    total_req = len(must_haves)
    covered_count = len(covered_skills)
    readiness_pct = round((covered_count / total_req) * 100.0, 1) if total_req > 0 else 0.0
    
    return {
        "domain_name": domain_info["title"],
        "description": domain_info["description"],
        "readiness_pct": readiness_pct,
        "covered_skills": covered_skills,
        "missing_skills": missing_skills,
        "roadmap_steps": domain_info["roadmap_steps"],
        "interview_topics": domain_info["interview_topics"],
        "suggested_projects": domain_info["suggested_projects"],
        "tools": domain_info["tools"]
    }
