import numpy as np 
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from experiment import json2params, project_n_lines, calibrate
import cv2 
from pprint import pprint

def gaussian_noise(x, mu, std):
    x_noisy = []
    for i in range(len(x)):
        noise = mu + std*np.random.randn(len(x[0])) 
        x_n = x[i] + noise
        x_noisy.append(x_n)
    x_noisy = [x[i] + np.random.normal(mu,std,size=x[0].shape) for i in range(len(x))]
    return x_noisy


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
        params included K(focal length),R,T 
        
    
    return:
        f,R,t
    
    """
    K, R, T = params 
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


def project_n_lines(Xfs, Xhs, noise, **config):
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
        xhs.append(xh.squeeze())

        xf, _ = cv2.projectPoints(Xf, r_gt.as_rotvec(), tvec_gt, K, np.zeros(4))
        xfs.append(xf.squeeze())

    # Put the noises in image plane
    xfs = gaussian_noise(xfs, 0, noise)
    xhs = gaussian_noise(xhs, 0, noise)

    xfs, xhs = np.array(xfs), np.array(xhs)
    return xfs, xhs 

def params2KRT(f,h,theta, phi):

    K =np.array([[f, 0., 0.], 
                 [0.,f,  0.], 
                 [0., 0., 1.]]) 
    Rx = np.array([[1., 0, 0], 
                   [0, np.cos(theta), -np.sin(theta)], 
                   [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], 
                   [np.sin(phi), np.cos(phi), 0], 
                   [0, 0, 1.]])
    
    R =  Rz @ Rx 
    T = -h * R[:,2]
    return K, R, T
    

if __name__ == "__main__":

    METADATA = "metadata/"
    CONFIG = METADATA + "calib_synthetic.json"
    LINESEGDATA = METADATA+"line_seg_panoptic.json"

    # Create data 
    config = json2params(CONFIG,"syn")
    Xfs, Xhs = create_3D_data(n = 10, l = config["l"])
    xfs, xhs = project_n_lines(Xfs, Xhs, noise=2.0,
                            theta = config["theta"],
                            phi = config["phi"],
                            cam_pos = config["cam_pos"],
                            cam_w = config["cam_w"],
                            cam_h = config["cam_h"],
                            f = config["f"])

    # Implement calibration 
    calib_result = calibrate(xfs, xhs, config)

    # choose the results 
    calibration_method = "IQR"
    f = calib_result[calibration_method]['f']
    theta = calib_result[calibration_method]['theta']
    phi = calib_result[calibration_method]['phi']
    h = calib_result[calibration_method]['height']

    # convert params to K,R,T 
    K,R,T = params2KRT(f, h, theta, phi)

