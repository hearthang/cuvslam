"""Optional approximate physical return region; permits nonzero endpoint offset."""
import numpy as np
import gtsam

def return_region_factor(first,last,radius_m=.2,sigma_m=.1):
    if radius_m<0 or sigma_m<=0:raise ValueError('Invalid radius/sigma')
    def error(this,values,H):
        a=values.atPose3(this.keys()[0]);b=values.atPose3(this.keys()[1])
        d=b.translation()-a.translation();distance=np.linalg.norm(d)
        outside=distance>radius_m
        if H is not None:
            direction=(d/distance).reshape(1,3) if outside else np.zeros((1,3))
            H[0]=np.hstack([np.zeros((1,3)),-direction@a.rotation().matrix()])
            H[1]=np.hstack([np.zeros((1,3)),direction@b.rotation().matrix()])
        return np.array([max(0.,distance-radius_m)])
    return gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(1,sigma_m),[first,last],error)
