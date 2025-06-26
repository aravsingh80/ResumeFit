import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.callbacks import EarlyStopping
#import pickle as pkl
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

with open("new_training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed_model(data_values):
    return embed(data_values).numpy()
resumes = [d["resume_text"] for  d in data]
jobs = [d["job_description"] for  d in data]
labels = [d["label"] for  d in data]
resume_vectors = embed_model(resumes)
job_vectors = embed_model(jobs)
cosine_sims = [cosine_similarity([resume_vectors[x]], [job_vectors[x]])[0][0] for x in range(len(resume_vectors))]
weight_value = 0.1 #Can be changed
cosine_sims = np.array(cosine_sims).reshape(-1, 1) * weight_value
x = np.concatenate([resume_vectors, job_vectors, cosine_sims], axis = 1)
y = np.array(labels).astype(np.float32)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(x.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8, callbacks=[early_stop])

y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype("int32")

report_text = []
report_text.append("Accuracy: {:.2f}".format(accuracy_score(y_val, y_pred)))
report_text.append("\n Classification Report:\n")
report_text.append(classification_report(y_val, y_pred, digits=2, zero_division=0))
report_text.append("\n Confusion Matrix:\n")
report_text.append(str(confusion_matrix(y_val, y_pred)))

with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_text))

model.save("app/model/new_model.keras")
# with open("app/model/temp_data.pkl", "wb") as f:
#     pkl.dump({
#         "resume_vectors": resume_vectors,
#         "job_vectors": job_vectors,
#         "cosine_sims": cosine_sims,
#         "x": x,
#         "y": y
#     }, f)
