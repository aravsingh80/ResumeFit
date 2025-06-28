import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.callbacks import EarlyStopping
#import pickle as pkl
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization

class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-6, max_lr=1e-1, steps=100):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.lrs = []
        self.losses = []

    def on_train_batch_end(self, batch, logs=None):
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (len(self.lrs) / self.steps)
        self.model.optimizer.learning_rate.assign(lr)
        self.lrs.append(lr)
        self.losses.append(logs["loss"])
        if len(self.lrs) >= self.steps:
            self.model.stop_training = True

def run_lr_finder(model, X_train, y_train, min_lr=1e-6, max_lr=1e-1, steps=100, batch_size=32):
    lr_finder = LRFinder(min_lr=min_lr, max_lr=max_lr, steps=steps)
    print(type(min_lr), "hello")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=min_lr), loss="binary_crossentropy")
    print("hi")
    model.fit(X_train, y_train, batch_size=batch_size, callbacks=[lr_finder], verbose=0)
    print("hi2")
    plt.plot(lr_finder.lrs, lr_finder.losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.show()
    losses = np.array(lr_finder.losses)
    lrs = np.array(lr_finder.lrs)

    # Find index of minimum loss and print learning rate just before that point
    min_loss_idx = losses.argmin()
    best_lr = lrs[min_loss_idx - 1] if min_loss_idx > 0 else lrs[0]
    print(best_lr)

with open("test_training_data.json", "r", encoding="utf-8") as f:
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
weight_value = 0.0 #Can be changed
cosine_sims = np.array(cosine_sims).reshape(-1, 1) * weight_value
x = np.concatenate([resume_vectors, job_vectors, cosine_sims], axis = 1)
y = np.array(labels).astype(np.float32)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation="relu", input_shape=(x.shape[1],)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(32, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64),
    LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation="sigmoid")
])
run_lr_finder(model, X_train, y_train)