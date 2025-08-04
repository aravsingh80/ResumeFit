import ollama

MODEL = "qwen2.5:3b-instruct"

def analyze_resume_with_feedback(resume_text, job_text, model_score, model_name: str = MODEL, stream: bool = False):
    prompt = (
        "You are a resume coach. Compare the RESUME to the JOB DESCRIPTION and output ONLY bullet-pointed, actionable suggestions to better match the job.\n"
        "Rules:\n"
        "1) Do NOT copy phrases from the inputs.\n"
        "2) 6–8 bullets, each ≤20 words, imperative voice.\n"
        "3) If a skill/tool is missing, start with 'Add:'. If speculative, start with 'Learn:'.\n"
        "4) Include at least one bullet that quantifies impact.\n"
        f"Model similarity: {model_score}%.\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_text}\n\n"
        "Bullets:"
    )

    if stream:
        parts = []
        for chunk in ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            keep_alive="30m",
            options={"num_predict": 120, "temperature": 0, "repeat_penalty": 1.2}
        ):
            c = chunk.get("message", {}).get("content", "")
            if c:
                print(c, end="", flush=True)
                parts.append(c)
        text = "".join(parts)
    else:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            keep_alive="30m",
            options={"num_predict": 120, "temperature": 0, "repeat_penalty": 1.2}
        )
        text = resp["message"]["content"]

    return {"suggestions": text, "final_score": model_score}

if __name__ == "__main__":
    sample_resume = """
    Emily Zhang
    emily.zhang@email.com | (555) 987-6543 | linkedin.com/in/emilyzhang

    Skills:
    • Python, Java, SQL, HTML/CSS, Flask, Git, Tableau
    • Object-Oriented Programming, Data Analysis, Agile, Unit Testing

    Experience:
    Software Developer Intern | GreenTech Analytics | June 2024 – August 2024
    • Built a web dashboard in Flask to visualize energy usage data for B2B clients
    • Cleaned and transformed large CSV datasets with Python and SQL for monthly reports
    • Participated in sprint planning, daily standups, and retrospectives using Agile methods

    Teaching Assistant | CS Department, University of California | Jan 2024 – May 2024
    • Led weekly lab sessions on Java and data structures
    • Held office hours and graded projects for 120+ students

    Projects:
    • Travel Planner App – Designed a full-stack app using Flask, PostgreSQL, and Bootstrap
    • COVID-19 Data Dashboard – Used Tableau to create live dashboards for California county-level case tracking

    Education:
    BS in Computer Science | University of California, Davis | Expected Graduation: June 2025
    GPA: 3.7/4.0 | Relevant Coursework: Databases, Operating Systems, Software Engineering
    """

    sample_job = """
    Position: Full-Stack Software Engineering Intern – Summer 2025

    Role Overview:
    Our internship program is designed for passionate software engineering students eager to work on production-level systems. Interns will contribute to new feature development, API design, and integration with AWS services.

    Responsibilities:
    • Build and deploy scalable web applications using React, Node.js, and REST APIs
    • Work with AWS (Lambda, S3, DynamoDB) to integrate backend functionality
    • Collaborate with product, design, and QA to deliver secure and performant features
    • Participate in code reviews and CI/CD automation

    Minimum Requirements:
    • Strong foundation in JavaScript, Python, or Java
    • Experience with web frameworks (e.g., Express, Flask, or Django)
    • Understanding of version control systems and APIs
    • Team player with excellent communication skills

    Preferred Qualifications:
    • Exposure to AWS services
    • Experience with React or similar frontend frameworks
    • Familiarity with Docker and cloud deployment
    """

    score = 64.8  # Simulated score from your trained model
    result = analyze_resume_with_feedback(sample_resume, sample_job, score)

    print("Suggestions:\n", result["suggestions"])
    print("Final Score:", result["final_score"])
