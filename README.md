> **Tampilan minimal (revisi UI).** Tema putih/hitam mengikuti sistem dan dapat diubah manual. Aksen hijau hanya untuk status, tombol utama, dan hasil. Perubahan ini tidak memodifikasi `app.js`, `model/`, `reference/`, ataupun metodologi penelitian.

# TomatoLeaf Lab — Versi Browser (GitHub + Vercel)

**Tujuan:** Demonstrasi web atas eksperimen yang sudah ada di `Tomato_MLP_vs_CNN.ipynb` dan `Laporan_UTS_Pembelajaran_Mendalam_Daun_Tomat`. Ini bukan model baru, bukan training baru, bukan penggantian metode, dan bukan evaluasi akurasi baru.

## 1. Berkas yang dipertahankan dan sumbernya

- `reference/notebook/` dan `reference/laporan/`: salinan notebook dan laporan penelitian asli; tidak diubah.
- `reference/CNN_reg_final.keras`: salinan asli model final **CNN + regularisasi**; tidak diubah.
- `reference/config.json`: konfigurasi asli (tau = **0.945**, gambar **64×64**).
- `reference/inference.py`: implementasi inferensi Python asli sebagai acuan.
- `examples/`: 13 gambar dari `app/examples/` proyek semula, termasuk uji non-tomat dan contoh gagal.
- `tools/export_weights.py`: mengekspor bobot dari arsip `.keras` menjadi `model/weights.bin` tanpa melatih ulang.
- `reference/demo_examples.json`: catatan prediksi dataset contoh asli untuk uji konsistensi.
- `model/manifest.json`: metadata urutan kelas, parameter, dimensi, hash SHA256 model asli dan bobot hasil ekspor.

## 2. Kesetaraan dengan notebook / laporan

| Aspek | Python asli | Web ini |
|---|---|---|
| Model | `CNN_reg_final.keras` | Tensor Float32 hasil ekspor model yang **sama** |
| Blok inferensi | Conv2D → BN → ReLU → Pool untuk blok 1–3, lalu Conv2D → BN → ReLU blok 4 | Sama; `tf.conv2d`, `tf.batchNorm`, `tf.relu`, `tf.maxPool` |
| Head | GlobalAveragePooling2D → Dense(10,softmax) | Mean spasial → Dense berbobot sama → softmax |
| Augmentasi, dropout | Otomatis nonaktif saat inferensi | Tidak dieksekusi (sesuai inferensi) |
| Urutan label | 10 kelas tetap | Identik dari `inference.py` |
| Normalisasi | RGB float32 / 255 | RGB float32 / 255 |
| Resize | PIL bilinear 64×64 | Canvas bilinear 64×64 (implementasi berbeda; **potensi selisih piksel** pada batas tertentu) |
| Threshold | 0.945 | 0.945 default, slider hanya uji interaktif |
| Grad-CAM | Feature map `relu4`, gradien **logit** | Formula gradien analitik layer GAP + Dense yang sama; peta 8×8 di-upsample ke 64×64 |

**Catatan penting:** TF.js dan TensorFlow Python dapat menghasilkan perbedaan pembulatan numerik kecil, khususnya WebGL. Browser Canvas dan PIL BILINEAR juga dapat menghasilkan nilai resize yang sedikit berbeda. Untuk membuktikan ekuivalensi secara ketat, jalankan uji perbandingan output di bawah. Jangan mengklaim kesetaraan numerik sempurna sebelum menjalankannya.

**Angka laporan tidak boleh diubah:** CNN + regularisasi: **99.450 parameter, akurasi uji 92,55%**. CNN baseline yang lebih tinggi akurasinya tidak digunakan sebagai model deployment karena laporan memilih CNN + regularisasi. Nilai ini berasal dari dataset uji PlantVillage, **bukan** hasil pengujian kamera/HP.

## 3. Jalankan lokal

Folder ini sudah website statis, tidak butuh Node.js, Python backend, atau npm install:

```bash
cd TomatoLeaf-Vercel-Original
python -m http.server 8000
```

Buka `http://localhost:8000`. Membuka `index.html` langsung lewat `file://` **tidak bisa**, karena browser memblokir pembacaan model lokal melalui `fetch`.

**Internet diperlukan pertama kali** untuk mengunduh TensorFlow.js melalui CDN. Saat deploy ke Vercel, website juga membutuhkan akses pengguna ke CDN tersebut.

## 4. Deploy GitHub → Vercel

1. Buat repository GitHub dan unggah **isi folder ini** (bukan ZIP, dan jangan meng-upload hanya `index.html`).
2. Masuk ke https://vercel.com/new, pilih **Import Git Repository**, pilih repo.
3. Framework Preset: **Other**. Root Directory: `./` bila file `index.html` ada di root repo.
4. Build Command: **kosongkan**. Output Directory: **kosongkan** (atau `.` bila diminta). Install Command: **kosongkan**.
5. Deploy dan buka URL `*.vercel.app` (HTTPS otomatis).
6. Tunggu indikator **CNN siap digunakan** lalu coba `Contoh data → 09_healthy.jpg` atau upload gambar.
7. Untuk kamera, beri izin saat browser meminta akses. Gunakan kamera belakang HP jika tersedia.

Jika file ini berada di **subfolder** repository, set Root Directory ke subfolder tersebut.

## 5. Uji menggunakan layar HP

1. Buka foto daun tomat di HP lain. Perbesar agar daun terlihat jelas, gunakan brightness sedang-tinggi.
2. Arahkan webcam laptop ke layar HP; pastikan keseluruhan daun terlihat jelas dan tidak terlalu banyak pantulan.
3. Di website pilih **Kamera langsung → Aktifkan kamera**.
4. Tunggu prediksi atau tekan **Ambil 1 frame**.
5. **Ini uji demonstrasi input eksternal (domain shift)**. Kualitas prediksi tidak dijamin dan hasil tidak boleh dipakai sebagai perhitungan ulang akurasi laporan.

## 6. Cara memverifikasi dengan program Python asli

1. Instal TensorFlow di lingkungan Python terpisah sesuai versi notebook.
2. Jalankan skrip `tools/python_reference_predictions.py` (menggunakan `reference/inference.py`) untuk menghasilkan output prediksi contoh asli.
3. Jalankan versi web pada contoh yang sama. Cocokkan kelas dan nilai probabilitas (toleransi wajar karena resize/float32/browser backend).
4. Untuk perbandingan paling ketat, berikan tensor RGB 64×64 yang sama **sebelum inferensi** pada kedua implementasi, misalnya menggunakan data Float32 yang diekspor dari PIL. Lalu bandingkan selisih probabilitas maksimum.

Rincian penggunaan lebih lanjut tersedia di `PANDUAN_DEMO_DOSEN.md`.

## 7. Batasan ilmiah dan teknis

- Website adalah **klasifikasi gambar penuh**, bukan pendeteksian lokasi/bounding box daun. Tidak bisa menjanjikan deteksi andal untuk gambar dari layar HP.
- Input di luar distribusi PlantVillage dapat memperoleh confidence tinggi tetapi salah. Threshold MSP bukan deteksi OOD sempurna.
- Kamera **tidak mengirim** gambar ke server; seluruh pemrosesan di browser.
- Performa webcam bergantung perangkat dan ketersediaan backend TensorFlow.js (WebGL/CPU).
- Browser modern HTTPS diperlukan untuk kamera dari domain publik.
- Web Grad-CAM adalah translasi matematis metode asli, tetapi warna colormap browser disusun untuk tampilan dan dapat berbeda sedikit dari Matplotlib `jet`.
