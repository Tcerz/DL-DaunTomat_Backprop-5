"""Ekspor langsung bobot (bukan train ulang) dari model Keras asli.
Output binary Float32 little endian + manifest shape untuk TensorFlow.js ops.
Requires: pip install h5py numpy
"""
from pathlib import Path
import json,zipfile,io,hashlib
import numpy as np
import h5py
BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'reference'/'CNN_reg_final.keras'
OUT=BASE/'model'
WEIGHTS={}
with zipfile.ZipFile(SOURCE) as z:
    cfg=json.loads(z.read('config.json'))
    layernames=[(l['class_name'],l['config'].get('name')) for l in cfg['config']['layers']]
    expected=[('Conv2D',f'conv{i}') for i in range(1,5)]+ [('BatchNormalization',f'bn{i}') for i in range(1,5)]+ [('Dense','logits')]
    assert all(x in layernames for x in expected), 'Arsitektur tidak sesuai model yang terdokumentasi'
    with h5py.File(io.BytesIO(z.read('model.weights.h5')),'r') as h:
        for i in range(4):
            p=f'layers/conv2d{"_"+str(i) if i else ""}/vars/0'
            WEIGHTS[f'conv{i+1}.kernel']=h[p][()].astype('<f4')
            p=f'layers/batch_normalization{"_"+str(i) if i else ""}/vars'
            for j,n in enumerate(['gamma','beta','mean','variance']):
                WEIGHTS[f'bn{i+1}.{n}']=h[f'{p}/{j}'][()].astype('<f4')
        WEIGHTS['logits.kernel']=h['layers/dense/vars/0'][()].astype('<f4')
        WEIGHTS['logits.bias']=h['layers/dense/vars/1'][()].astype('<f4')
shapes={}; pos=0; bufs=[]
for name, arr in WEIGHTS.items():
    arr=np.ascontiguousarray(arr)
    bufs.append(arr.tobytes())
    shapes[name]={'shape':list(arr.shape),'offset':pos,'length':int(arr.size)}
    pos+=int(arr.size)
binary=b''.join(bufs)
assert pos == 99450, f'parameter model berubah: {pos}'
(OUT/'weights.bin').write_bytes(binary)
manifest={'format':'keras-cnn-f32-v1','name':'CNN_reg_final.keras','parameters':pos,
 'inputSize':64,'normalization':'RGB / 255.0','classNames': ['Bacterial_spot','Early_blight','Late_blight','Leaf_Mold','Septoria_leaf_spot','Spider_mites','Target_Spot','Tomato_Yellow_Leaf_Curl_Virus','Tomato_mosaic_virus','healthy'],
 'tau':json.loads((BASE/'reference'/'config.json').read_text())['tau'],
 'gradcamLayer':'relu4','weights':shapes,'sha256':hashlib.sha256(binary).hexdigest(),
 'sourceSha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'notes':'Exact Keras tensors exported to IEEE Float32. Browser uses tf.conv2d + inference BN + relu + maxPool + GAP + dense softmax; augmentation/dropout inactive.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Exported',pos,'parameters',len(binary),'bytes')
