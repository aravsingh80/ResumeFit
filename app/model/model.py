import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def compute_score(resume_text, job_text):
    embeddings = embed([resume_text, job_text])
    vec1, vec2 = embeddings[0], embeddings[1]

    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return round(float(cosine_sim) * 100, 2)