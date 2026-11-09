"""Modul inti proyek: pipeline tf.data, arsitektur MLP/CNN, pelatihan, dan evaluasi."""
import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

DATA = os.environ.get("DATA_DIR", "data")
RES = os.environ.get("RES_DIR", "results")
os.makedirs(RES, exist_ok=True)
IMG, NC, BATCH, SEED = 64, 10, 64, 42
BN_MOMENTUM = 0.9   # bawaan Keras 0.99; diturunkan ke 0.9 karena val_acc sangat berosilasi pada percobaan awal


def load_data():
    d = np.load(f"{DATA}/tomato64.npz")
    meta = json.load(open(f"{DATA}/meta.json"))
    X, y = d["X"], d["y"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te]), d["X_ood"], meta["class_names"]


def make_ds(X, y, training, batch=BATCH, seed=SEED):
    """Input pipeline (materi P1-P2): from_tensor_slices -> shuffle -> map(scale) -> batch -> prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def class_weights(y):
    cnt = np.bincount(y, minlength=NC)
    return {i: float(len(y) / (NC * cnt[i])) for i in range(NC)}


def augmentation():
    """Augmentasi citra (materi P3); aktif hanya saat training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
    ], name="augmentasi")


# ------------------------------------------------------------------ MLP
def build_mlp_baseline():
    return keras.Sequential([
        layers.Input(shape=(IMG, IMG, 3)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(NC, activation="softmax"),
    ], name="MLP_baseline")


def build_mlp_reg(l2=1e-4, drop=0.3, aug=True):
    reg = regularizers.l2(l2)
    m = [layers.Input(shape=(IMG, IMG, 3))]
    if aug:
        m.append(augmentation())
    m.append(layers.Flatten())
    for u in (256, 128):                       # Dense -> BN -> ReLU -> Dropout (materi P4)
        m += [layers.Dense(u, use_bias=False, kernel_regularizer=reg),
              layers.BatchNormalization(momentum=BN_MOMENTUM), layers.Activation("relu"), layers.Dropout(drop)]
    m.append(layers.Dense(NC, activation="softmax"))
    return keras.Sequential(m, name="MLP_reg")


# ------------------------------------------------------------------ CNN
def build_cnn_baseline():
    """CNN dasar sesuai materi P6: Conv-Pool x3 -> Flatten -> Dense."""
    return keras.Sequential([
        layers.Input(shape=(IMG, IMG, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NC, activation="softmax"),
    ], name="CNN_baseline")


def build_cnn_reg(drop=0.3, l2=1e-4, aug=True, filters=(16, 32, 64, 128)):
    """CNN teregulasi (model akhir). Functional API agar Grad-CAM dapat mengakses feature map terakhir."""
    reg = regularizers.l2(l2)
    inp = keras.Input(shape=(IMG, IMG, 3))
    x = augmentation()(inp) if aug else inp
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=reg,
                          name=f"conv{i+1}")(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"bn{i+1}")(x)
        x = layers.Activation("relu", name=f"relu{i+1}")(x)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D(name=f"pool{i+1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(drop, name="dropout")(x)
    out = layers.Dense(NC, activation="softmax", kernel_regularizer=reg, name="logits")(x)
    return keras.Model(inp, out, name="CNN_reg")


# ------------------------------------------------------------------ training
class HistLogger(keras.callbacks.Callback):
    """Mencatat riwayat per epoch ke JSON (idempotent terhadap epoch) agar riwayat utuh walau pelatihan dilanjutkan dari backup."""
    def __init__(self, path):
        super().__init__(); self.path = path
        self.h = json.load(open(path)) if os.path.exists(path) else {}
    def on_epoch_begin(self, epoch, logs=None): self._t = time.time()
    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {}); logs["epoch_time"] = time.time() - self._t
        logs["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        for k, v in logs.items():
            self.h[k] = self.h.get(k, [])[:epoch] + [float(v)]
        json.dump(self.h, open(self.path, "w"))


def cosine_lr(lr0, total, lr_min=1e-5):
    """Cosine annealing per epoch (materi P5: learning rate scheduling)."""
    return lambda e, lr=None: float(lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * e / total)))


def train(model, name, data, optimizer, epochs=30, use_cw=True, sched=None, es_patience=6, schedule_note=""):
    """sched: None (lr konstan) | "plateau" (ReduceLROnPlateau) | ("cosine", lr0) (cosine annealing selama `epochs`).
    Pelatihan dapat dilanjutkan otomatis dari epoch terakhir (BackupAndRestore) jika proses terputus."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data[:3]
    keras.utils.set_random_seed(SEED)
    train_ds, val_ds, test_ds = make_ds(Xtr, ytr, True), make_ds(Xva, yva, False), make_ds(Xte, yte, False)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hl = HistLogger(f"{RES}/{name}_history_partial.json")
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True), hl,
           keras.callbacks.BackupAndRestore(f"{RES}/backup_{name}")]
    if sched == "plateau":
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5))
    elif isinstance(sched, tuple) and sched[0] == "cosine":
        cbs.append(keras.callbacks.LearningRateScheduler(cosine_lr(sched[1], epochs)))
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     class_weight=class_weights(ytr) if use_cw else None, callbacks=cbs, verbose=2)
    h = hl.h
    res = evaluate(model, test_ds, yte)
    res_val = evaluate(model, val_ds, yva)
    res.update(name=name, params=int(model.count_params()), epochs_run=len(h["loss"]),
               best_epoch=int(np.argmin(h["val_loss"]) + 1), train_time_s=round(float(sum(h["epoch_time"])), 1),
               sec_per_epoch=round(float(np.mean(h["epoch_time"])), 1), val_acc=res_val["acc"],
               val_macro_f1=res_val["macro_f1"], schedule_note=schedule_note, history=h)
    proba = res.pop("_proba")
    json.dump(res, open(f"{RES}/{name}.json", "w"))
    np.save(f"{RES}/{name}_test_proba.npy", proba)
    return model, res


def evaluate(model, ds, y_true):
    proba = model.predict(ds, verbose=0)
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, pred, average="weighted", zero_division=0)
    return dict(acc=float(accuracy_score(y_true, pred)), macro_precision=float(p), macro_recall=float(r),
                macro_f1=float(f), weighted_f1=float(fw), cm=confusion_matrix(y_true, pred, labels=range(NC)).tolist(),
                _proba=proba)
