# from app.routes import app  # moved after env var is set

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from app.utils.extract_resume import extract_text
from app.utils.job_web_scraper import scrape_job
from app.model.test import get_score
from app.utils.feedback_generator import analyze_resume_with_feedback

app = Flask(__name__)
CORS(app)

STATIC_RESUME = "uploads/AravSinghResume - Final.pdf"  # repo root

PLACEHOLDER_JOB_DESC = """
Who We Are:
TransPerfect was founded with a mission to help the world's businesses navigate the global marketplace. Today, we have grown to be an industry leader organization by helping clients globalize their business no matter what service they might need. TransPerfect provides a full array of language and business support services, including translation, multicultural marketing, website globalization, legal support, and any kind of technology solution.

What You Will Be Doing:
TransPerfect is an established company with a start-up culture seeking creative entrepreneurial people like you to join our team. We’re seeking a Software Engineer Intern to join our global team to be responsible for ensuring the integrity, reliability and maintenance of our internally developed software products from conception to release. If you’re ready to join a growing company and make an immediate impact, we want to hear from you! 

Multi service-oriented architecture is our strategy. You will be working on one or more of our REST API services written in DotNet. These communicate with each other using REST API endpoints and in some cases RabbitMQ messages. All our services and web applications are sharing common services such as oauth authentication and file storage.

A typical client or internal facing project will use one or two services created specifically for it as well as one or two services shared among our general infrastructure.

Attention to detail and quality overall is very important in all software we maintain. A typical day on the job would be attending the project standup and collaborating with different parts of our cross functional project team. This includes design, QA, and product management.

Responsibilities:
Perform responsive software development and integration, within an agile development process
Support software life-cycle, to include design, developing and testing under
Work closely with users to provide 1 – 1 personalized software support as needed
Design, build, and maintain efficient, reusable, and reliable code/databases
Ensure the best possible performance, quality, and responsiveness of applications
Identify bottlenecks and bugs, and devise solutions to mitigate and address these issues
Help maintain code quality, organization, and automation

Who We Are Looking For:
Your experience includes:
Bachelor's degree in Computer Science, or related field
Ability to turn designs into backend features
Team coordination skills
Excellent communication
Knowledge of Git
Knowledge of REST API best practices
Database knowledge (document and/or relational)
Troubleshooting expertise and attention to detail
Ability to learn and understand code
Knowledge of the C# programming language
Zero - two years of experience in software development and testing
Willingness to adapt and evolve to technology changes
Ability to work independently and within a team

Desired Skills:
SQL
Docker
API Debugging
"""

def run_model(resume_bytes: bytes, job_desc: str):
    resume_text = extract_text(STATIC_RESUME)
    curr_score = get_score(resume_text, PLACEHOLDER_JOB_DESC)
    print(curr_score)
    result_obj = analyze_resume_with_feedback(resume_text, PLACEHOLDER_JOB_DESC, model_score=curr_score)
    score = float(result_obj["final_score"])
    feedback = result_obj["suggestions"]
    print(feedback)
    return score, feedback


@app.route("/analyze", methods=["GET"])
def analyze():
    with open(STATIC_RESUME, "rb") as f:
        resume_bytes = f.read()

    score, feedback = run_model(resume_bytes, PLACEHOLDER_JOB_DESC)

    return jsonify({
        "job_description": "TransPerfect Software Engineer Intern",
        "score": score,
        "feedback": feedback
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)