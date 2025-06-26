from flask import Flask, render_template, request, redirect, url_for, session
import os
from app.utils.extract_resume import extract_text
from app.utils.job_web_scraper import scrape_job
from app.model.model import compute_score
from app.utils.feedback_generator import analyze_resume_with_feedback

app = Flask(__name__)
app.secret_key = "key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("resume")
        job_url = request.form.get("job_url", "").strip()
        job_text_input = request.form.get("job_text", "").strip()

        resume_text = ""
        job_text = ""

        if file and file.filename.endswith(".pdf"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            resume_text = extract_text(filepath)

        if job_url:
            job_text = scrape_job(job_url)
        if not job_text and job_text_input:
            job_text = job_text_input

        if resume_text and job_text:
            score = compute_score(resume_text, job_text)
            result_obj = analyze_resume_with_feedback(resume_text, job_text, model_score=score)

            # Ensure all values are JSON serializable
            clean_result = {
                "model_score": float(result_obj["model_score"]),
                "cosine_similarity": float(result_obj["cosine_similarity"]),
                "final_score": float(result_obj["final_score"]),
                "suggestions": result_obj["suggestions"]
            }

            session["result"] = clean_result
        else:
            session["result"] = {
                "final_score": None,
                "error": "Missing resume or job description (or website extraction failed)."
            }
        return redirect(url_for("result"))

    return render_template("index.html")

@app.route("/result")
def result():
    result = session.get("result", {})
    return render_template("result.html", result=result)
