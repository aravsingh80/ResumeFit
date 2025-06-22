import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
model = tf.keras.models.load_model("app/model/new_model.h5")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed_model(data_values):
    return embed([data_values])[0].numpy()
def compute_score(resume_text, job_text):
    resume_vectors = embed_model(resume_text)
    job_vectors = embed_model(job_text)
    weight_value = 0.1 #can be changed
    cosine_sims = cosine_similarity([resume_vectors], [job_vectors])[0][0] * weight_value
    input_vec = np.concatenate([resume_vectors, job_vectors, [cosine_sims]]).reshape(1, -1)
    score = model.predict(input_vec)[0][0]
    return int(score * 100)
#     embeddings = embed([resume_text, job_text])
#     vec1, vec2 = embeddings[0], embeddings[1]

#     cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#     return round(float(cosine_sim) * 100, 2)