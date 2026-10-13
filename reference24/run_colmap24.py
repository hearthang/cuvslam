"""Bounded COLMAP CPU rig pilot in an isolated environment; poses are not truth."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import sys
import json
import time
import datetime
from pathlib import Path
import numpy as np
import pycolmap as p

root=Path(sys.argv[1]).resolve();meta=json.load(open(root/'input.json'))
state=dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),version=p.__version__,cuda=p.has_cuda)
def stage(name):
    state.update(stage=name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root/'status.json').write_text(json.dumps(state,indent=2));print(name,flush=True)
database=root/'database.db';images=root/'images'
if database.exists():
    raise FileExistsError('Pilot database exists; inspect stages before resuming')
stage('extracting')
ex=p.FeatureExtractionOptions();ex.num_threads=2;ex.sift.max_num_features=2500
for name in ['left','right']:
    c=meta[f'camera_{name}'];reader=p.ImageReaderOptions()
    reader.camera_model=c['model'];reader.camera_params=','.join(map(str,c['params']))
    reader.mask_path=str(root/'masks')
    p.extract_features(database,images,image_names=[f'{name}/{i:06d}.png' for i in meta['selected_kf']],
                       camera_mode=p.CameraMode.SINGLE,reader_options=reader,extraction_options=ex,device=p.Device.cpu)
stage('applying_fixed_rig')
left=p.RigConfigCamera();left.ref_sensor=True;left.image_prefix='left/'
right=p.RigConfigCamera();right.image_prefix='right/'
right.cam_from_rig=p.Rigid3d(np.array(meta['right_from_left']))
config=p.RigConfig();config.cameras=[left,right]
with p.Database.open(database) as db:
    p.apply_rig_config([config],db)
stage('matching')
matching=p.FeatureMatchingOptions();matching.num_threads=2;matching.guided_matching=True
pairing=p.ImportedPairingOptions();pairing.match_list_path=str(root/'pairs.txt')
verification=p.TwoViewGeometryOptions();verification.ransac.random_seed=19
p.match_image_pairs(database,matching_options=matching,pairing_options=pairing,verification_options=verification,device=p.Device.cpu)
if '--match-only' in sys.argv:
    stage('matched')
    sys.exit(0)
stage('mapping')
options=p.IncrementalPipelineOptions();options.num_threads=2;options.random_seed=19
options.max_num_models=3;options.max_runtime_seconds=3600
options.ba_refine_focal_length=False;options.ba_refine_principal_point=False
options.ba_refine_extra_params=False;options.ba_refine_sensor_from_rig=False
options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
options.ba_global_max_num_iterations=60;options.ba_local_max_num_iterations=25
models=p.incremental_mapping(database,images,root/'sparse',options=options)
summary=[]
for key,r in models.items():
    output=[]
    for image in r.images.values():
        if not image.name.startswith('left/') or not image.has_pose:
            continue
        k=int(Path(image.name).stem);idx=meta['selected_kf'].index(k)
        Twc=image.cam_from_world().inverse().matrix()
        M=np.eye(4);M[:3]=Twc
        Twi=M@np.linalg.inv(np.array(meta['imu_from_left']))
        quat=p.Rotation3d(Twi[:3,:3]).quat
        output.append([meta['timestamps_s'][idx],*Twi[:3,3],*quat])
    output.sort(key=lambda v:v[0]);np.savetxt(root/f'model_{key}_imu_tum.txt',output,fmt='%.9f')
    summary.append(dict(model=key,images=r.num_reg_images(),points=r.num_points3D(),
                        reprojection_error_px=r.compute_mean_reprojection_error(),left_poses=len(output)))
state['models']=summary;state['note']='Independent visual geometric cross-check; no IMU optimization; sparse keyframes, not final trajectory.'
stage('finished')
print(json.dumps(summary),flush=True)
