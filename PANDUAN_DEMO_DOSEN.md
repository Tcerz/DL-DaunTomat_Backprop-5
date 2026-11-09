# Panduan demonstrasi untuk dosen

## Pernyataan awal saat demo

"Website ini merupakan media deployment untuk model CNN + regularisasi hasil pelatihan dari notebook kami. Tidak ada retraining, arsitektur baru, atau pemilihan model baru. Bobotnya diekspor dari `CNN_reg_final.keras` asli ke Float32 agar bisa dijalankan di browser. Hasil akurasi 92,55% dalam laporan tetap merujuk pada dataset uji, bukan uji webcam."

## Urutan demonstrasi 3–5 menit

1. Buka website Vercel dan periksa indikator `CNN siap digunakan`.
2. Tunjukkan angka `99.450 parameter`, input `64×64`, dan threshold `0,945`.
3. Buka **Contoh data** dan pilih satu citra sehat atau berpenyakit. Perlihatkan prediksi, confidence, 10 probabilitas, dan Grad-CAM.
4. Pilih contoh `ood_Pepper` / `ood_Potato` untuk menjelaskan keterbatasan threshold OOD; **jangan menjanjikan seluruh gambar non-tomat pasti ditolak**.
5. Buka **Upload gambar** dan uji citra eksternal.
6. Buka **Kamera langsung**, berikan izin kamera, lalu tampilkan foto daun dari layar HP lain. Kamera mengambil frame setiap ~1,2 detik bila fitur otomatis aktif.
7. Jelaskan bahwa performa pada input kamera/lapangan belum diukur di laporan; ini hanya uji kualitatif.

## Jika dosen bertanya

**Mengapa web tanpa FastAPI?** Model dan semua tensor sudah berada di browser. Vercel hanya mengirim file statis; TensorFlow.js menjalankan perhitungan pada perangkat penguji. Ini perubahan cara eksekusi, bukan perubahan model.

**Apakah benar-benar model yang sama?** Ya, `reference/CNN_reg_final.keras` ikut disertakan. `tools/export_weights.py` mengekstrak persis tensor tersimpan tanpa training, dan `model/manifest.json` menyimpan hash serta nama layer. Perbandingan nilai probabilitas dengan Python disarankan untuk verifikasi lanjut.

**Apakah kamera meningkatkan akurasi?** Tidak ada klaim itu. Kamera adalah input baru di luar pengujian asli. Nilai akurasi harus diukur pada dataset berlabel tersendiri jika ingin diklaim.

**Apa risiko foto daun melalui layar HP?** Moiré, pantulan, tingkat kecerahan, blur, dan background berbeda dari distribusi training. Model hanya melakukan klasifikasi seluruh frame, tidak mendeteksi bounding box daun.

**Mengapa ada pilihan threshold?** Agar bisa menunjukkan cara kerja abstention. Nilai awal selalu 0,945 sesuai laporan. Menurunkan threshold tidak membuat model lebih akurat.
