"""Landmarks behind any observing calibrated camera cannot support pose BA."""
import numpy as np

def invalid_landmarks(poses,points,pose_idx,camera_idx,point_idx,extr,active,min_depth=.3):
    bad=np.zeros(len(points),bool)
    for i in np.unique(pose_idx):
        for c in (0,1):
            take=(pose_idx==i)&(camera_idx==c)&active[point_idx]
            if not take.any():continue
            ids=point_idx[take];T=np.linalg.inv(poses[i]@extr[c])
            xyz=points[ids]@T[:3,:3].T+T[:3,3]
            bad[ids[~np.isfinite(xyz).all(1)|(xyz[:,2]<=min_depth)]]=True
    return np.flatnonzero(bad)
