"""Construct a stereo rig model from the database and fixed v1 trajectory."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pycolmap as p
root=Path(sys.argv[1]).resolve();meta=json.load(open(root/'input.json'));out=root/'geometry';out.mkdir(exist_ok=True)
if (out/'heldout_tracks.json').exists():raise FileExistsError(out)
r=p.Reconstruction()
with p.Database.open(root/'database.db') as db:
    for camera in db.read_all_cameras():r.add_camera(camera)
    for rig in db.read_all_rigs():r.add_rig(rig)
    for frame in db.read_all_frames():r.add_frame(frame)
    for image in db.read_all_images():
        image.points2D=[p.Point2D(xy) for xy in db.read_keypoints(image.image_id)[:,:2]]
        r.add_image(image)
base=np.loadtxt(Path('results/reference_v1/trajectories')/f"{meta['tag']}.txt")
before={}
for im in r.images.values():
    if not im.name.startswith('left/'):continue
    k=int(Path(im.name).stem);ts=meta['timestamps_s'][meta['selected_kf'].index(k)]
    n=int(np.argmin(abs(base[:,0]-ts)))
    if abs(base[n,0]-ts)>1e-5:raise ValueError('Missing v1 frame timestamp')
    T=np.eye(4);T[:3,:3]=p.Rotation3d(base[n,4:8]).matrix();T[:3,3]=base[n,1:4]
    T=T@np.array(meta['imu_from_left'])
    r.frames[im.frame_id].rig_from_world=p.Rigid3d(np.linalg.inv(T)[:3]);r.register_frame(im.frame_id)
    before[im.frame_id]=r.frames[im.frame_id].rig_from_world.matrix().copy()
options=p.IncrementalPipelineOptions();options.num_threads=2;options.random_seed=19
options.fix_existing_frames=True;options.mapper.fix_existing_frames=True
options.ba_refine_focal_length=False;options.ba_refine_principal_point=False
options.ba_refine_extra_params=False;options.ba_refine_sensor_from_rig=False
r=p.triangulate_points(r,root/'database.db',root/'images',out/'initial',clear_points=True,options=options,refine_intrinsics=False)
for fid,T in before.items():np.testing.assert_allclose(r.frames[fid].rig_from_world.matrix(),T,atol=1e-6,rtol=0)
held=[]
for pid,point in r.points3D.items():
    elements=list(point.track.elements);frames={r.images[e.image_id].frame_id for e in elements}
    if len(elements)<5 or len(frames)<3:continue
    signature=';'.join(f'{e.image_id}:{e.point2D_idx}' for e in sorted(elements,key=lambda e:(e.image_id,e.point2D_idx)))
    if int(hashlib.sha256(signature.encode()).hexdigest()[:8],16)%5:continue
    last=max(frames,key=lambda f:int(Path(next(r.images[e.image_id].name for e in elements if r.images[e.image_id].frame_id==f)).stem))
    train=[];test=[]
    for e in elements:
        im=r.images[e.image_id];entry=(e.image_id,im.points2D[e.point2D_idx].xy.tolist())
        (test if im.frame_id==last else train).append(entry)
    held.append(dict(point_id=int(pid),train=train,test=test))
(out/'heldout_tracks.json').write_text(json.dumps(held))
(out/'summary.json').write_text(json.dumps(dict(tag=meta['tag'],frames=r.num_reg_frames(),images=r.num_reg_images(),points=r.num_points3D(),
    held_tracks=len(held),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    initial_trajectory_sha256=hashlib.sha256((Path('results/reference_v1/trajectories')/f"{meta['tag']}.txt").read_bytes()).hexdigest(),
    note='Fixed v1 poses and calibrated rig; no free SfM mapper needed; reserved tracks excluded from subsequent BA.'),indent=2))
print('fixed-pose map',meta['tag'],r.num_reg_frames(),r.num_points3D(),'reserved',len(held),flush=True)
