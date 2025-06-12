from flask import Flask, render_template, request, redirect, url_for, session
import os
from app.utils.extract_resume import extract_text
from app.utils.job_web_scraper import scrape_job

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

@app.route("/", methods = ["GET", "POST"])
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
            session["result"] = job_text #Just a placeholder until we get the model to work
        else:
            session["result"] = "Error: Either missing resume/job description or the job description website can not be extracted and the description needs to be extracted manually."
        return redirect(url_for("result"))
    
@app.route("/result")
def result():
    result = session.get("result", "No result to show")
    return render_template("result.html", result = result)