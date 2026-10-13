"""Replace training stereo observations with subpixel forward/backward KLT.

All temporal left observations and reserved validation tracks stay unchanged.
Only retain landmarks with at least three left frames and two valid stereo pairs.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import cv2

def stereo(left,right,xy_colmap,K0,D0,K1,D1,T):
    p0=np.asarray(xy_colmap,np.float32).reshape(-1,1,2)-.5
    lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    p1,s1,_=cv2.calcOpticalFlowPyrLK(left,right,p0,None,**lk)
    p2,s2,_=cv2.calcOpticalFlowPyrLK(right,left,p1,None,**lk)
    xy=p1.reshape(-1,2)+.5
    valid=(s1.ravel()==1)&(s2.ravel()==1)&(np.linalg.norm((p2-p0).reshape(-1,2),axis=1)<.3)
    valid&=(xy[:,0]>=10)&(xy[:,0]<right.shape[1]-10)&(xy[:,1]>=10)&(xy[:,1]<right.shape[0]-90)
    a=cv2.fisheye.undistortPoints(np.asarray(xy_colmap,float).reshape(-1,1,2),K0,D0).reshape(-1,2)
    b=cv2.fisheye.undistortPoints(xy.astype(float).reshape(-1,1,2),K1,D1).reshape(-1,2)
    Xh=cv2.triangulatePoints(np.eye(3,4),T[:3],a.T,b.T)
    with np.errstate(divide='ignore',invalid='ignore'):
        X=(Xh[:3]/Xh[3]).T;Y=X@T[:3,:3].T+T[:3,3]
        valid&=np.isfinite(X).all(1)&(X[:,2]>.3)&(X[:,2]<60)&(Y[:,2]>.3)
        valid&=(np.linalg.norm(X[:,:2]/X[:,2:]-a,axis=1)<.5/K0[0,0])
        valid&=(np.linalg.norm(Y[:,:2]/Y[:,2:]-b,axis=1)<.5/K1[0,0])
    return xy,valid

def main():
    cv2.setNumThreads(1);root=Path(sys.argv[1]);source=root/'vi_bundle_input.npz';dest=root/'vi_bundle_klt.npz'
    if dest.exists():raise FileExistsError(dest)
    # NpzFile indexes decompress a fresh array each time. Cache once: storing
    # row views of repeated loads otherwise retains a full array for every row.
    with np.load(source) as archive:z={name:archive[name] for name in archive.files}
    meta=json.loads(str(z['metadata']));validation=json.loads(str(z['validation']))
    if 'half-pixel' not in meta.get('pixel_convention',''):raise ValueError('Requires corrected pixel convention')
    kd=[]
    for name in ('left','right'):
        fx,fy,cx,cy,*d=meta['camera_'+name]['params'];kd.append((np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),np.array(d)))
    T=np.eye(4);T[:3]=meta['right_from_left'];by_point={};attempts=0;accepted=0;duplicate_heldout=0
    held_right={}
    for track in validation:
        for i,c,xy in track['train']+track['test']:
            if c==1:held_right.setdefault(i,[]).append(xy)
    for i,k in enumerate(meta['selected_kf']):
        sel=np.flatnonzero((z['pose_idx']==i)&(z['camera_idx']==0))
        if not len(sel):continue
        left=cv2.imread(str(root/'images/left'/f'{k:06d}.png'),0);right=cv2.imread(str(root/'images/right'/f'{k:06d}.png'),0)
        if left is None or right is None:raise ValueError('Missing stereo frame')
        uv,valid=stereo(left,right,z['pixels'][sel],*kd[0],*kd[1],T);attempts+=len(sel)
        if i in held_right:
            held=np.array(held_right[i]);near=np.min(np.linalg.norm(uv[:,None,:]-held[None,:,:],axis=2),axis=1)<2
            duplicate_heldout+=int((valid&near).sum());valid&=~near
        for n,idx in enumerate(sel):
            j=int(z['point_idx'][idx]);rows=by_point.setdefault(j,[])
            rows.append((i,0,z['pixels'][idx]))
            if valid[n]:rows.append((i,1,uv[n]));accepted+=1
    points=[];pose_idx=[];camera_idx=[];point_idx=[];pixels=[]
    for old,rows in sorted(by_point.items()):
        if sum(c==1 for _,c,_ in rows)<2 or len({i for i,c,_ in rows if c==0})<3:continue
        new=len(points);points.append(z['points'][old])
        for i,c,xy in rows:pose_idx.append(i);camera_idx.append(c);point_idx.append(new);pixels.append(xy)
    stats=dict(attempted_stereo=attempts,accepted_stereo=accepted,heldout_proximity_rejections=duplicate_heldout,
        points_before=len(z['points']),points_after=len(points),observations=len(pixels),
        source_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Only training right observations replaced/added; reserved tracks unchanged; no pose or endpoint used to accept KLT matches.')
    np.savez(dest,points=np.array(points),pose_idx=np.array(pose_idx,int),camera_idx=np.array(camera_idx,int),point_idx=np.array(point_idx,int),
        pixels=np.array(pixels),timestamps=z['timestamps'],metadata=z['metadata'],validation=z['validation'],stereo_refinement=json.dumps(stats))
    (root/'klt_refinement.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
