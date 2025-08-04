import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TFHUB_CACHE_DIR"] = os.path.abspath("./tfhub_cache")

from keras.saving import legacy_load_model
model = legacy_load_model(os.path.join(BASE_DIR, "test_model.keras"))
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def embed_model(data_values):
    return embed([data_values])[0].numpy()

def compute_score(resume_text, job_text):
    resume_vectors = embed_model(resume_text)
    job_vectors = embed_model(job_text)
    cosine_sim = cosine_similarity([resume_vectors], [job_vectors])[0][0] * 1.0  # Match training scale
    input_vec = np.concatenate([resume_vectors, job_vectors, [cosine_sim]]).reshape(1, -1)
    raw_score = model.predict(input_vec)[0][0]
    return round(float(np.clip(raw_score, 0, 1)) * 100, 2)  # Return percent
