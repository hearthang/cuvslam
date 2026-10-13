"""Export raw observations from fixed-pose COLMAP triangulation, with held tracks."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pycolmap as p
root=Path(sys.argv[1]).resolve();meta=json.load(open(root/'input.json'))
source=sys.argv[2] if len(sys.argv)>2 else 'seeded'
r=p.Reconstruction(root/source/'initial')
held=json.load(open(root/source/'heldout_tracks.json'))
reserved={t['point_id'] for t in held}
frame_map={int(k):i for i,k in enumerate(meta['selected_kf'])}
def obs(image_id,xy):
    im=r.images[image_id];k=int(Path(im.name).stem)
    return frame_map[k],0 if im.name.startswith('left/') else 1,xy
train=[]
for pid,point in r.points3D.items():
    if pid in reserved or len(point.track.elements)<4:continue
    observations=[];valid=True
    for e in point.track.elements:
        im=r.images[e.image_id];Y=im.cam_from_world()*point.xyz
        if Y[2]<.3:valid=False;break
        xy=im.points2D[e.point2D_idx].xy
        if np.linalg.norm(r.cameras[im.camera_id].img_from_cam(Y)-xy)>8:valid=False;break
        observations.append(obs(e.image_id,xy.tolist()))
    if valid and len({a[0] for a in observations})>=3:
        train.append((len(observations),int(pid),point.xyz,observations))
train=sorted(train,key=lambda row:(-row[0],row[1]))[:6000]
points=[];pose_indices=[];camera_indices=[];point_indices=[];pixels=[]
for n,(_,pid,xyz,oo) in enumerate(train):
    points.append(xyz)
    for i,c,xy in oo:
        pose_indices.append(i);camera_indices.append(c);point_indices.append(n);pixels.append(xy)
validation=[dict(point_id=t['point_id'],train=[obs(i,xy) for i,xy in t['train']],
                 test=[obs(i,xy) for i,xy in t['test']]) for t in held]
dest=root/'vi_bundle_input.npz'
if dest.exists():raise FileExistsError(dest)
np.savez(dest,points=np.array(points),pose_idx=np.array(pose_indices,int),camera_idx=np.array(camera_indices,int),
         point_idx=np.array(point_indices,int),pixels=np.array(pixels),timestamps=np.array(meta['timestamps_s']),
         metadata=json.dumps(meta),validation=json.dumps(validation),
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print('BA points',len(points),'observations',len(pixels),'withheld',len(validation),flush=True)
