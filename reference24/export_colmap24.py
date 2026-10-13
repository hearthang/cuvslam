"""Export calibrated stereo keyframes for an independent COLMAP pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import sys
import json
import pickle
import hashlib
import argparse
from pathlib import Path
sys.path.append(str(Path.cwd()))
import numpy as np
import cv2
import backend as B

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('tag');ap.add_argument('--output-root',type=Path,default=Path('results/reference_24h_20261010/colmap'))
ap.add_argument('--max-keyframes',type=int,default=120)
ap.add_argument('--calib',default='/mnt/f/data_set/slam/calibration_front_small.json')
ap.add_argument('--cam1-fscale',type=float,default=-0.00038)
cli=ap.parse_args();tag=cli.tag
if cli.max_keyframes<10:raise ValueError('Need at least ten keyframes')
root=Path.cwd();out=cli.output_root.resolve()/tag
out.mkdir(parents=True,exist_ok=True)
if (out/'input.json').exists():
    raise FileExistsError('Export exists; inspect before replacing')
fe=root/'results/fe5';z=np.load(fe/f'{tag}_fe.npz')
imgs=pickle.load(open(fe/f'{tag}_kf.pkl','rb'))
B.configure(cli.calib,cli.cam1_fscale);cv2.setNumThreads(1)
ids=np.unique(np.linspace(0,len(imgs)-1,min(cli.max_keyframes,len(imgs)),dtype=int))
held=B.handling_mask(z['imu'],z['t'])[z['kf_idx']]
ids=ids[~held[ids]]
valid=[];tiny=[];shape=None
for i in ids:
    decoded=[cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_GRAYSCALE) for b in imgs[i]]
    if any(a is None for a in decoded):
        continue
    valid.append(int(i));tiny.append(B.tiny_descriptor(imgs[i][0]))
    for camera,a in zip(['left','right'],decoded):
        shape=a.shape
        folder=out/'images'/camera;folder.mkdir(parents=True,exist_ok=True)
        maskfolder=out/'masks'/camera;maskfolder.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(folder/f'{i:06d}.png'),a)
        mask=np.full(a.shape,255,np.uint8);mask[-B.MASK_BOTTOM:]=0
        cv2.imwrite(str(maskfolder/f'{i:06d}.png.png'),mask)
ids=np.array(valid);tiny=np.array(tiny);sim=tiny@tiny.T
pairs=set()
def add(a,b):
    if a!=b:pairs.add(tuple(sorted([a,b])))
for n,i in enumerate(ids):
    add(f'left/{i:06d}.png',f'right/{i:06d}.png')
    nearby=list(range(max(0,n-3),n))
    far=np.flatnonzero(np.abs(ids- i)>len(imgs)*.05)
    ranked=far[np.argsort(-sim[n,far])[:6]]
    for m in np.unique(np.r_[nearby,ranked]).astype(int):
        j=ids[m]
        for c0,c1 in [('left','left'),('right','right'),('left','right')]:
            add(f'{c0}/{i:06d}.png',f'{c1}/{j:06d}.png')
(out/'pairs.txt').write_text(''.join(f'{a} {b}\n' for a,b in sorted(pairs)))
def camera(K,D):
    return dict(model='OPENCV_FISHEYE',width=shape[1],height=shape[0],
                params=[float(K[0,0]),float(K[1,1]),float(K[0,2])+.5,float(K[1,2])+.5,*D.tolist()])
metadata=dict(tag=tag,selected_kf=ids.tolist(),timestamps_s=(z['t'][z['kf_idx'][ids]]*1e-9).tolist(),
              camera_left=camera(B.K0,B.D0),camera_right=camera(B.K1,B.D1),
              right_from_left=B.T_C1C0[:3].tolist(),imu_from_left=B.T_IC0.tolist(),
              pairs=len(pairs),pixel_convention='COLMAP half-pixel centers; OpenCV/Kalibr cx and cy shifted by +0.5',
              calibration_sha256=hashlib.sha256(Path(B.args.calib).read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              input_sha256={name:hashlib.sha256((fe/name).read_bytes()).hexdigest() for name in [f'{tag}_fe.npz',f'{tag}_kf.pkl']},
              note='No trajectory or Basalt pose used. Fixed fisheye calibration and stereo extrinsics. Image candidates selected by time and thumbnails.')
(out/'input.json').write_text(json.dumps(metadata,indent=2))
print(out,'images',len(ids)*2,'pairs',len(pairs))
