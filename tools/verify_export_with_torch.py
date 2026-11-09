"""Sanity check bobot export menggunakan PyTorch CPU, bukan hasil evaluasi ilmiah.
Requires: torch numpy pillow. Dapat dibandingkan dengan results/demo_examples.json dari ZIP asli.
"""
import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
p=Path(__file__).resolve().parents[1]
m=json.loads((p/'model/manifest.json').read_text()); blob=(p/'model/weights.bin').read_bytes()
w={name:torch.from_numpy(np.frombuffer(blob,dtype='<f4',count=v['length'],offset=v['offset']*4).copy()).reshape(v['shape']) for name,v in m['weights'].items()}
torch.set_num_threads(1)
for f in sorted((p/'examples').glob('*.jpg')):
 img=Image.open(f).convert('RGB').resize((64,64),Image.Resampling.BILINEAR)
 x=torch.from_numpy(np.asarray(img,dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
 with torch.no_grad():
  for i in range(1,5):
   k=w[f'conv{i}.kernel'].permute(3,2,0,1)
   x=F.conv2d(x,k,padding=1)
   x=(x-w[f'bn{i}.mean'][None,:,None,None])/torch.sqrt(w[f'bn{i}.variance'][None,:,None,None]+.001)
   x=x*w[f'bn{i}.gamma'][None,:,None,None]+w[f'bn{i}.beta'][None,:,None,None]
   x=F.relu(x)
   if i < 4: x=F.max_pool2d(x,2)
  logits=x.mean((-2,-1))@w['logits.kernel']+w['logits.bias']
  prob=logits.softmax(-1)[0]
  i=prob.argmax().item()
 print(f.name, m['classNames'][i],round(prob[i].item(),7))
