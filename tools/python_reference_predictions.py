"""Opsional, jalankan di Python dengan tensorflow dan Pillow: prediksi referensi dari kode ASLI."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from inference import Predictor
from PIL import Image
base=Path(__file__).resolve().parents[1]
model=Predictor(str(base/'reference/CNN_reg_final.keras'),tau=0.945)
results=[]
for p in sorted((base/'examples').glob('*.jpg')):
 r=model.predict(Image.open(p),with_cam=False)
 results.append({'file':p.name,'class':r['label'],'accepted':r['accepted'],'confidence':r['confidence'],'proba':[float(x) for x in r['proba']]})
print(json.dumps(results,ensure_ascii=False,indent=2))
