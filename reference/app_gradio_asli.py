"""Aplikasi web klasifikasi penyakit daun tomat (Gradio).

Model : CNN + regularisasi (BN, Dropout, L2, augmentasi) - 64x64 RGB, 10 kelas PlantVillage-Tomato.
Fitur unik:
  1) Grad-CAM            -> menampilkan area daun yang mendasari prediksi.
  2) Confidence threshold -> menolak prediksi berkeyakinan rendah / gambar yang bukan daun tomat.
"""
import json, os
import numpy as np
import gradio as gr
from PIL import Image
from inference import Predictor, overlay, CLASS_NAMES, LABEL_ID

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "model", "config.json")))
TAU = float(CFG["tau"])
predictor = Predictor(os.path.join(HERE, "model", "CNN_reg_final.keras"), tau=TAU)


def run(img, tau):
    if img is None:
        return {}, "Silakan unggah foto satu daun tomat.", None
    predictor.tau = float(tau)
    out = predictor.predict(img, with_cam=True)
    p = out["proba"]
    top = np.argsort(-p)[:3]
    label_dict = {LABEL_ID[CLASS_NAMES[i]]: float(p[i]) for i in top}
    best = LABEL_ID[out["label"]]
    if out["accepted"]:
        verdict = (f"### ✅ {best}\nKeyakinan **{out['confidence']*100:.1f}%** (≥ ambang {tau:.2f}). "
                   f"Area merah pada peta Grad-CAM menunjukkan bagian daun yang paling memengaruhi keputusan model.")
    else:
        verdict = (f"### ⚠️ Prediksi ditolak\nKeyakinan tertinggi hanya **{out['confidence']*100:.1f}%** (< ambang {tau:.2f}) "
                   f"untuk *{best}*. Gambar kemungkinan **bukan daun tomat**, buram, atau berada di luar kondisi data latih. "
                   f"Coba foto ulang satu daun dengan latar polos dan pencahayaan merata.")
    cam_img = overlay(out["x"], out["cam"])
    cam_img = np.asarray(Image.fromarray((cam_img * 255).astype(np.uint8)).resize((320, 320), Image.BICUBIC))
    return label_dict, verdict, cam_img


examples_dir = os.path.join(HERE, "examples")
examples = [[os.path.join(examples_dir, f), TAU] for f in sorted(os.listdir(examples_dir)) if f.lower().endswith((".jpg", ".png"))] if os.path.isdir(examples_dir) else []

with gr.Blocks(title="Klasifikasi Penyakit Daun Tomat") as demo:
    gr.Markdown("# 🍅 Klasifikasi Penyakit Daun Tomat\n"
                "CNN yang dilatih pada PlantVillage (10 kelas). Dilengkapi **Grad-CAM** (penjelasan visual) dan "
                "**confidence threshold** (menolak gambar yang tidak yakin / bukan daun tomat).")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Foto daun tomat")
            tau = gr.Slider(0.30, 0.99, value=TAU, step=0.01, label="Ambang keyakinan (tau)")
            btn = gr.Button("Klasifikasikan", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            lbl = gr.Label(num_top_classes=3, label="3 prediksi teratas")
            cam = gr.Image(label="Grad-CAM (area penentu prediksi)")
    btn.click(run, [inp, tau], [lbl, verdict, cam])
    if examples:
        gr.Examples(examples=examples, inputs=[inp, tau], outputs=[lbl, verdict, cam], fn=run, cache_examples=False)
    gr.Markdown("**Batasan:** model dilatih pada foto laboratorium berlatar polos (PlantVillage) berukuran 64x64; "
                "performa pada foto lapangan dengan latar kompleks dapat menurun. Hasil ini bukan pengganti diagnosis ahli.")

if __name__ == "__main__":
    demo.launch()
