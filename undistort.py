import numpy as np 
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from calib_lines import calib_camera_vanilla, calib_camera_ransac
from experiment import json2params, project_n_lines, create_synthetic_data
import cv2 

def dist(x,y):
    L2 = euclidean(x,y)
    return L2 

def get_reprojection_error(a:np.ndarray,
                       A:np.ndarray,
                       b:np.ndarray,
                       B:np.ndarray,
                       params):
    """
    Obj:

        This method target for optimization of calibration results with estimated camera parameters as initial guesses.
    
    params: 
        a is observed foot points in image plane   
        b is observed head points in image plane
        A is 3D points of foot
        B is 3D points of head 
        params included K(focal length),theta,phi,heigth
        
    
    return:
        f,R,t
    
    """
    f, R, T = params 
    
    K = np.array([f,0,0],
                 [0,f,0],
                 [0,0,1])
    
    A_reprojected = K @ (R @ A + T)
    B_reprojected = K @ (R @ B + T)

    errors = [dist(a_i,A_i)^2 + dist(b_i,B_i)^2 for a_i, A_i, b_i, B_i in zip(a, A_reprojected, b, B_reprojected)]
    return errors

def get_leastsq(params,func):
    # params: f,R,T 
    # func: will be 

    # get result of least square, res will be f, R, t 
    initial_guess = params
    result = least_squares(func, initial_guess)
    res = result.x

    # get RMS 
    residuals_squared = result.fun**2
    mean_squared_residuals = np.mean(residuals_squared)
    rms = np.sqrt(mean_squared_residuals)

    return res, rms 


def create_3D_data(n, l):
    Xhs, Xfs = [], []
    for _ in range(n):
        x = np.random.uniform(-2.5,2.5)
        y = np.random.uniform(2,5) #Important to calculate Working distance correctly  
        Xfs.append(np.array([x,y,0]))
        Xhs.append(np.array([x,y,l]))
    Xfs, Xhs = np.array(Xfs), np.array(Xhs)
    return Xfs, Xhs


def project_n_lines(Xfs, Xhs, **config):
    xfs, xhs = [], [] 

    # Camera paramters 
    Rx = np.array([[1., 0, 0], 
                   [0, np.cos(config["theta"]), -np.sin(config["theta"])], 
                   [0, np.sin(config["theta"]), np.cos(config["theta"])]])
    Rz = np.array([[np.cos(config["phi"]), -np.sin(config["phi"]), 0], 
                   [np.sin(config["phi"]), np.cos(config["phi"]), 0], 
                   [0, 0, 1.]])
    R_gt =  Rz @ Rx 
    r_gt = Rotation.from_matrix(R_gt)
    tvec_gt = -R_gt @ config["cam_pos"]
    K =np.array([[config["f"], 0., config["cam_w"]/2], 
                 [0., config["f"], config["cam_h"]/2], 
                 [0., 0., 1.]]) 
    
    # Project 3D points 
    for Xh, Xf in zip(Xhs, Xfs): 
        xh, _ = cv2.projectPoints(Xh, r_gt.as_rotvec(), tvec_gt, K, np.zeros(4))
        xhs.append(xh.squeeze(1))

        xf, _ = cv2.projectPoints(Xf, r_gt.as_rotvec(), tvec_gt, K, np.zeros(4))
        xfs.append(xf.squeeze(1))
    
    xfs, xhs = np.array(xfs), np.array(xhs)
    return xfs, xhs 


if __name__ == "__main__":

    METADATA = "metadata/"
    CONFIG = METADATA + "calib_synthetic.json"
    LINESEGDATA = METADATA+"line_seg_panoptic.json"

    # create data 
    config = json2params(CONFIG,"syn")
    Xfs, Xhs = create_3D_data(n = 10, l = config["l"])
    xfs, xhs = project_n_lines(Xfs, Xhs,
                            theta = config["theta"],
                            phi = config["phi"],
                            cam_pos = config["cam_pos"],
                            cam_w = config["cam_w"],
                            cam_h = config["cam_h"],
                            f = config["f"])
    breakpoint()
    # 

