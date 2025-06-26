from app.utils.feedback_generator import analyze_resume_with_feedback

resume = "Experienced Python developer with background in machine learning and backend systems."
job = "Looking for a backend developer skilled in Python, APIs, and distributed systems."

result = analyze_resume_with_feedback(resume, job, model_score=0.6)
print("RESULT:", result)