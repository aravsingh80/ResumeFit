import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

import os
os.environ['TFHUB_CACHE_DIR'] = './tfhub_cache'

# Load dataset
with open("regression_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed_long_list(text_list, max_tokens=512):
    embedded_texts = []
    for text in text_list:
        token_count = len(text.split())
        if token_count <= max_tokens:
            embedding = embed([text]).numpy()[0]
        else:
            sentences = sent_tokenize(text)
            chunks, chunk = [], []
            current_tokens = 0
            for sentence in sentences:
                current_tokens += len(sentence.split())
                chunk.append(sentence)
                if current_tokens >= max_tokens:
                    chunks.append(" ".join(chunk))
                    chunk, current_tokens = [], 0
            if chunk:
                chunks.append(" ".join(chunk))
            embeddings = embed(chunks).numpy()
            embedding = np.mean(embeddings, axis=0)
        embedded_texts.append(embedding)
    return np.array(embedded_texts)


resumes = [d["resume_text"] for d in data]
jobs = [d["job_description"] for d in data]
labels = [float(d["label"]) for d in data]  # <-- Use float values

resume_vectors = embed_long_list(resumes)
job_vectors = embed_long_list(jobs)
pca_resume = PCA(n_components=64)
pca_job = PCA(n_components=64)
resume_vectors_reduced = pca_resume.fit_transform(resume_vectors)
job_vectors_reduced = pca_job.fit_transform(job_vectors)

cosine_sims = [cosine_similarity([resume_vectors[i]], [job_vectors[i]])[0][0] for i in range(len(resumes))]
cosine_sims = np.array(cosine_sims).reshape(-1, 1)


X = np.concatenate([resume_vectors_reduced, job_vectors_reduced, cosine_sims], axis=1)
y = np.array(labels).astype(np.float32)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Outputs value in [0, 1]
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=7e-6),
    loss="mean_squared_error",  # <-- Regression loss
    metrics=["mae"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# Evaluation metrics
y_pred = model.predict(X_val).flatten()
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"Mean Absolute Error: {mae:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")

model.save("app/model/test_model.keras")

import joblib
joblib.dump(pca_resume, "app/model/pca_resume.pkl")
joblib.dump(pca_job, "app/model/pca_job.pkl")
