from flask import Flask, request, jsonify
from flask_cors import CORS
from app.utils.extract_resume import extract_text
from app.utils.job_web_scraper import scrape_job
from app.model.test import get_score
from app.utils.feedback_generator import analyze_resume_with_feedback
import os
import tempfile

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Missing resume file or job description"}), 400

    resume_file = request.files['resume']
    job_description = request.form['job_description']

    if resume_file.filename == '':
        return jsonify({"error": "Empty file uploaded"}), 400

    # Save resume temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        resume_path = temp_file.name
        resume_file.save(resume_path)

    try:
        resume_text = extract_text(resume_path)
        curr_score = get_score(resume_text, job_description)
        print(f"Score: {curr_score}")

        result_obj = analyze_resume_with_feedback(resume_text, job_description, model_score=curr_score)
        score = float(result_obj["final_score"])
        feedback = result_obj["suggestions"]
        print(f"Feedback: {feedback}")

        return jsonify({
            "job_description": job_description[:200] + "...",  # short preview
            "score": score,
            "feedback": feedback
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": "Failed to analyze resume"}), 500
    finally:
        if os.path.exists(resume_path):
            os.remove(resume_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
