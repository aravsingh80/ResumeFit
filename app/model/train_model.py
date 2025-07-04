import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.callbacks import EarlyStopping

#import pickle as pkl
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
#from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization

with open("most_recent_dataset.json", "r", encoding="utf-8") as f:
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
weight_value = 0.0
cosine_sims = np.array(cosine_sims).reshape(-1, 1) * weight_value
x = np.concatenate([resume_vectors, job_vectors, cosine_sims], axis = 1)
y = np.array(labels).astype(np.float32)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)

# from sklearn.utils import class_weight
# class_weights_array = class_weight.compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weight_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation="relu", input_shape=(x.shape[1],)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(32, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(x.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

# model = tf.keras.Sequential([
#     Dense(256),
#     BatchNormalization(),
#     LeakyReLU(alpha=0.01),
#     Dropout(0.3),

#     Dense(128),
#     BatchNormalization(),
#     LeakyReLU(alpha=0.01),
#     Dropout(0.2),

#     Dense(32),
#     LeakyReLU(alpha=0.01),

#     Dense(1, activation="sigmoid")
# ])
#7.943282347242815e-05
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=7.07945784384138e-06),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    #metrics=["accuracy"]
)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
    #class_weight=class_weight_dict
)

# Predict probabilities
y_pred_probs = model.predict(X_val)

# --- STEP 1: Optimize threshold using F1 score ---
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_val, (y_pred_probs > t).astype("int32")) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold based on F1 score: {best_threshold:.2f}")

# Use best threshold
y_pred = (y_pred_probs > best_threshold).astype("int32")

# Report
report_text = []
report_text.append("Best Threshold: {:.2f}".format(best_threshold))
report_text.append("Accuracy: {:.2f}".format(accuracy_score(y_val, y_pred)))
report_text.append("\n Classification Report:\n")
report_text.append(classification_report(y_val, y_pred, digits=2, zero_division=0))
report_text.append("\n Confusion Matrix:\n")
report_text.append(str(confusion_matrix(y_val, y_pred)))

with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_text))

model.save("test_model.keras")
# with open("app/model/temp_data.pkl", "wb") as f:
#     pkl.dump({
#         "resume_vectors": resume_vectors,
#         "job_vectors": job_vectors,
#         "cosine_sims": cosine_sims,
#         "x": x,
#         "y": y
#     }, f)
