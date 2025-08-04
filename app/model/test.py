import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
sentence_tokenizer = PunktSentenceTokenizer(punkt_param)

# TF-Hub cache location
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.getcwd(), "tfhub_cache")

# Load trained model and PCA components
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
BASE_DIR = os.path.abspath("C:/Users/email/OneDrive/Documents/eclipse/ResumeScanner/app/model/app/model")

pca_resume = joblib.load(os.path.join(BASE_DIR, "pca_resume.pkl"))
pca_job = joblib.load(os.path.join(BASE_DIR, "pca_job.pkl"))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "test_model.keras"))



# Embed with chunking if over token limit
def embed_text_avg(text, max_tokens=512):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return embed([text]).numpy()[0]

    sentences = sentence_tokenizer.tokenize(text)
    chunks, chunk, token_count = [], [], 0
    for sent in sentences:
        sent_tokens = sent.split()
        token_count += len(sent_tokens)
        chunk.append(sent)
        if token_count >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0
    if chunk:
        chunks.append(" ".join(chunk))

    embeddings = embed(chunks).numpy()
    return np.mean(embeddings, axis=0)

# Match score logic
def get_score(resume_text, job_description):
    resume_embed = embed_text_avg(resume_text)
    job_embed = embed_text_avg(job_description)

    resume_reduced = pca_resume.transform([resume_embed])
    job_reduced = pca_job.transform([job_embed])
    
    cos_sim = cosine_similarity([resume_embed], [job_embed])[0][0]
    input_vector = np.concatenate([resume_reduced[0], job_reduced[0], [cos_sim]]).reshape(1, -1)

    raw_score = model.predict(input_vector)[0][0]
    score_percent = round(np.clip(raw_score, 0, 1) * 100, 2)

    return score_percent

# -----------------------------
# TEST CASE 1: DIA + PDF Resume
# -----------------------------
resume_dia = """Scott Wells
erikacasey@torres.com | 0640963549
http://vasquez-mcmahon.net/ | http://www.blackburn.com/
Skills:
• AWS, Terraform, Docker, CI/CD, Bash, Python
Education:
• BS in Information Systems, Warren, Warren and Hicks, Expected Graduation: Dec 2026
Experience:
• Nichols, Cooke and Valdez | Cloud Infrastructure Intern | June 2025 – Current
    - Worked on backend development using Terraform and AWS.
    - Participated in Agile sprints, wrote unit tests, and used tools like CI/CD.
    - Contributed to team-based feature design and CI/CD pipelines.
Projects:
• Developed a dashboard using Bash and deployed on Heroku.
"""

job_dia = """The Software Engineer will be a member of a small team focusing on developing services and applications in a DevSecOps based environment in support of DIA. Engineering will be performed on JWICS and NSANet systems.
Understanding user requirements and strategizing software solutions
Supporting development of applications and capabilities within Zero Trust Environments.
Updating and refactoring legacy software into web-based applications.
Creating and optimizing in-application SQL and back-end access
Ensuring performance, security, and availability of databases
Profiling server resource usage and optimizing as necessary
Participating in code reviews and contributing to architectural decisions
Requirements:
Top Secret clearance (SCI eligible)
5+ years of software development
Languages/Tools:
JavaScript (ES6+), TypeScript, HTML5, CSS3, React, Angular, Node.js, Java, Python, PHP, C#
Databases: PostgreSQL, Oracle, MS SQL Server, MongoDB
DevOps: Git, CI/CD, Jenkins, Docker, Kubernetes, Terraform, AWS GovCloud, Azure Government
"""

print("DIA JOB + PDF RESUME")
get_score(resume_dia, job_dia)

# ---------------------------------
# TEST CASE 2: High-match example
# ---------------------------------
resume_match = """John Smith

john.smith@email.com | (555) 678-9012 | github.com/johnsmithdev | linkedin.com/in/johnsmithdev

EDUCATION

BS in Computer Science, University of Maryland, May 2025

GPA: 3.7/4.0 | Relevant Courses: Software Engineering, Web Programming, Database Systems

SKILLS

Languages: C#, ASP.NET, JavaScript, SQL  
Frameworks: .NET Core, MVC, Entity Framework, AngularJS  
Tools: Visual Studio, Git, Docker, Microsoft SQL Server

PROJECTS

HealthPortal Web App – Developed a secure patient portal for a healthcare provider  
- Built front-end in AngularJS and back-end in ASP.NET MVC  
- Integrated SQL Server database and implemented stored procedures  
- Created SSRS reports for appointment tracking and billing summaries

E-Commerce Dashboard – Built a .NET Core dashboard for inventory and sales analytics  
- Designed SQL schema and optimized queries  
- Used AJAX for dynamic UI updates  
- Deployed using IIS and Windows Server

EXPERIENCE

Software Developer Intern, MedTech Solutions, Summer 2024  
- Developed .NET APIs to manage patient data  
- Built SSIS packages to automate ETL workflows  
- Participated in Agile sprints and peer code reviews

LEADERSHIP

Vice President, Developer’s Guild – Led peer mentoring and weekly tech talks on .NET technologies


"""

job_match = """Your Tasks
This is a hands-on software developer position responsible for the web applications development and maintenance.

Candidate must be able to coordinate activities with technical leads/team on design, development and testing activities, and must have a good understanding of .NET Technologies and SQL Server.


Your Profile
Ideally three years in a relevant IT environment.
Familiar with Healthcare verticals.
Experience working in an Agile Environment.
Bachelor’s Degree from an accredited college or university in Engineering, Computer Science, Information Systems, or related field or equivalent experience in a software development role.
Must have developed at least three web sites using .NET Technologies and SQL Server.
Minimum 3 Years of hands-on experience working with .NET Technologies (C#, ASP.NET, MVC, WCF, Entity Framework).
Minimum 3 Years of experience working with AJAX, HTML 5, CSS and JavaScript along with frameworks such as JQuery and/or AngularJS.
Minimum 3 Years of experience working with Microsoft SQL Server (SQL queries, stored procedures, triggers, SQL Server Integration Services (SSIS), SQL Reporting Services (SSRS).
Able to update the knowledge in Microsoft technologies.
Exceptional communication and interpersonal skills with meticulous attention to detail.

"""

print("\nHIGH MATCH")
get_score(resume_match, job_match)

# ---------------------------------
# TEST CASE 3: Low-match example
# ---------------------------------
resume_mismatch = """Michelle Delgado
delgado.m@samplemail.com | 917-555-0123
https://delgadodigitalportfolio.com

Experienced Art Director with 8+ years in branding, graphic design, and editorial layout. Proficient in Adobe Creative Suite, especially InDesign and Photoshop. Strong visual storytelling and campaign development.

Education:
• BFA in Graphic Design, Pratt Institute

Experience:
• Lead Designer at UrbanEdge Media
• Art Director at Camden Publications

Skills:
• Typography • Branding • Adobe Suite • UX/UI Mockups • Visual Campaigns
"""

job_mismatch = """Job Title: Research Assistant – Oncology Lab
Assist in cancer research, data analysis, and wet lab sample processing.
Must have lab skills: pipetting, gel electrophoresis, and research documentation.
Enrolled in biology/biomedical program preferred.
Location: Johns Hopkins Medical Campus – 1-year, part-time.
"""

print("\nLOW MATCH")
get_score(resume_mismatch, job_mismatch)
