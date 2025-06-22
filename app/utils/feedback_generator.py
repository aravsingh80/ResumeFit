from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import re

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
gen_pipeline = pipeline("text-generation", model="distilgpt2")

def split_chunks(text):
    return [seg.strip() for seg in re.split(r'[.;:\n]', text) if len(seg.strip()) > 15]

def generate_feedback(resume_text, job_description, sim_threshold=0.65, max_segs=3):
    resume_vec = embed_model.encode([resume_text])[0]
    segments = split_chunks(job_description)

    feedbacks = []
    for segment in segments:
        seg_vec = embed_model.encode([segment])[0]
        sim = cosine_similarity([resume_vec], [seg_vec])[0][0]

        if sim < sim_threshold:
            prompt = f"The candidate resume does not mention \"{segment}\". Suggest how they can improve:\n"
            gen = gen_pipeline(prompt, max_new_tokens=40, do_sample=True, temperature=0.8)[0]['generated_text']
            feedbacks.append(gen.strip())
        if len(feedbacks) >= max_segs:
            break

    return feedbacks

def analyze_resume_with_feedback(resume_text, job_description, model_score, cosine_weight=0.1):
    resume_vec = embed_model.encode([resume_text])
    job_vec = embed_model.encode([job_description])
    cosine_sim = cosine_similarity(resume_vec, job_vec)[0][0]
    final_score = (1 - cosine_weight) * model_score + cosine_weight * cosine_sim

    suggestions = []
    if final_score < 0.7:
        suggestions = generate_feedback(resume_text, job_description)

    return {
        "model_score": round(model_score, 3),
        "cosine_similarity": round(cosine_sim, 3),
        "final_score": round(final_score, 3),
        "suggestions": suggestions
    }