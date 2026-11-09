"""Inferensi + dua fitur unik:
  (1) Grad-CAM          : penjelasan visual area daun yang mendasari prediksi.
  (2) Confidence threshold : model menolak (abstain) prediksi dengan keyakinan < tau,
                             sekaligus berfungsi sebagai deteksi input di luar domain (bukan daun tomat).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

IMG = 64
CLASS_NAMES = ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
               "Spider_mites", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "healthy"]
LABEL_ID = {
    "Bacterial_spot": "Bercak Bakteri (Bacterial Spot)",
    "Early_blight": "Hawar Awal (Early Blight)",
    "Late_blight": "Hawar Akhir (Late Blight)",
    "Leaf_Mold": "Jamur Daun (Leaf Mold)",
    "Septoria_leaf_spot": "Bercak Daun Septoria",
    "Spider_mites": "Tungau Laba-laba (Spider Mites)",
    "Target_Spot": "Bercak Target (Target Spot)",
    "Tomato_Yellow_Leaf_Curl_Virus": "Virus Keriting Kuning (YLCV)",
    "Tomato_mosaic_virus": "Virus Mosaik Tomat (ToMV)",
    "healthy": "Daun Sehat",
}
LAST_CONV_ACT = "relu4"


def preprocess(img):
    """PIL.Image / ndarray -> float32 (1, 64, 64, 3) pada rentang [0, 1]. Sama persis dengan pipeline training."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    img = img.convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None]


class Predictor:
    def __init__(self, model_path, tau=0.90):
        self.model = keras.models.load_model(model_path, compile=False)
        self.tau = tau
        self._cam_model = keras.Model(self.model.inputs, [self.model.get_layer(LAST_CONV_ACT).output])
        self._W, self._b = self.model.get_layer("logits").get_weights()

    def proba(self, x):
        return self.model.predict(x, verbose=0)

    def gradcam(self, x, class_idx):
        """Grad-CAM (Selvaraju dkk., 2017) terhadap logit kelas `class_idx` pada feature map konvolusi terakhir."""
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            fmap = self._cam_model(x, training=False)
            tape.watch(fmap)
            gap = tf.reduce_mean(fmap, axis=(1, 2))
            logits = tf.matmul(gap, self._W) + self._b
            score = logits[:, class_idx]
        grads = tape.gradient(score, fmap)
        alpha = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(alpha * fmap, axis=-1))[0]
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[..., None], (IMG, IMG), method="bilinear")[..., 0]
        return cam.numpy()

    def predict(self, img, with_cam=True):
        x = preprocess(img)
        p = self.proba(x)[0]
        k = int(p.argmax())
        out = dict(proba=p, index=k, label=CLASS_NAMES[k], confidence=float(p[k]),
                   accepted=bool(p[k] >= self.tau), x=x[0])
        if with_cam:
            out["cam"] = self.gradcam(x, k)
        return out


def overlay(x, cam, alpha=0.45):
    """Tumpuk heatmap Grad-CAM (colormap 'jet') pada citra x (HxWx3, [0,1])."""
    import matplotlib
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * x + alpha * heat, 0, 1)
