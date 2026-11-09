"""Bandingkan ekspor terhadap keluaran referensi 13 gambar dari laporan proyek.
Requires: python -m pip install torch pillow numpy
"""
import subprocess,sys,json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
expect=json.loads((p/'reference/demo_examples.json').read_text())['rows']
lines=subprocess.check_output([sys.executable,str(p/'tools/verify_export_with_torch.py')],text=True).splitlines()
results={}
for line in lines:
 name,cl,conf=line.rsplit(' ',2);results[name]=(cl,float(conf))
errors=[]
for row in expect:
 pred,confidence=results[row['file']]
 if pred!=row['pred'] or abs(confidence-row['conf'])>0.0001:
  errors.append((row['file'],row['pred'],pred,row['conf'],confidence))
if errors:
 print('FAILED',errors);sys.exit(1)
print('PASS: 13/13 prediksi dan confidence cocok dengan hasil arsip asli (toleransi 0.0001).')
