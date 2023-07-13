import numpy as np 
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import cv2 
from pprint import pprint
import warnings
from utils import json2params, calibrate, gaussian_noise

class ReprojectionError:
    def __init__(self, a_s:np.ndarray,
                       b_s:np.ndarray,
                       n: int):
        """
        Initialize the ReprojectionError object with the given parameters.

        Args:
            a_s: observed foot points in image plane   
            As: 3D points of foot
            b_s: observed head points in image plane
            Bs: 3D points of head
        """
        self.a_s = a_s
        #self.As = As
        self.b_s = b_s
        #self.Bs = Bs
        self.n = n 

    def __call__(self, params):
        """
        Call this object as a function to compute the reprojection error.

        Args:
            params: Camera parameters, np.array([f,theta,phi,h])

        Returns:
            errors of each reprojection of line segments
        """
        f = params[0]
        theta = params[1]
        phi = params[2]
        h = params[3]
        xs = params[4 : 4 + self.n]
        ys = params[4 + self.n : ]

        
        # Convert params to K R T 
        K =np.array([[f, 0., config["cam_w"]/2], 
                    [0., f, config["cam_h"]/2], 
                    [0., 0., 1.]]) 
        
        Rx = np.array([[1., 0, 0], 
                       [0, np.cos(theta), -np.sin(theta)], 
                       [0, np.sin(theta), np.cos(theta)]])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], 
                       [np.sin(phi), np.cos(phi), 0], 
                       [0, 0, 1.]])

        R =  Rz @ Rx 
        T = -h * R[:,2]

        # remake Xfs, Xhs 
        Xfs = np.array([[x, y, 0] for x, y in zip(xs,ys)])
        Xhs = np.array([[x, y, h] for x, y in zip(xs,ys)])
    
        A_ps, B_ps = [], [] 
        for A, B in zip(Xfs, Xhs):
            A_p = K @ (R @ A + T)
            B_p = K @ (R @ B + T)
            
            A_p /= A_p[2]
            A_p = A_p[:2]
            B_p /= B_p[2]
            B_p = B_p[:2]
            A_ps.append(A_p)
            B_ps.append(B_p)
         
        return np.array([(A_ps-self.a_s),(B_ps-self.b_s)]).flatten()

def get_leastsq(params, func):
    # params: f, theta, phi, height

    initial_guess = params.flatten()

    result = least_squares(func, initial_guess)
    res = result.x
    print(f'Optimization Result:{result.success}')
    print(" ")

    # get RMS 
    residuals_squared = result.fun**2
    mean_squared_residuals = np.mean(residuals_squared)
    rms = np.sqrt(mean_squared_residuals)

    return res, rms 

def create_3D_data(n, l):
    Xhs, Xfs = [], []
    for _ in range(n):
        x = np.random.uniform(-2.5,2.5)
        y = np.random.uniform(1,5) #Important to calculate Working distance correctly  
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

    

if __name__ == "__main__":
    # Ignore warings in numpy 
    warnings.filterwarnings('ignore', category = np.VisibleDeprecationWarning)
    warnings.filterwarnings('ignore', category = RuntimeWarning)
    
    # Metadata
    METADATA = "metadata/"
    CONFIG = METADATA + "calib_synthetic.json"
    LINESEGDATA = METADATA + "line_seg_panoptic.json"

    # Create data 
    config = json2params(CONFIG, "syn")
    pprint(f"1. Groundtruth\n")
    print("\n")
    pprint(f"focal length: {config['f']}, theta:{np.rad2deg(config['theta'])}, phi:{np.rad2deg(config['phi'])}, h: {config['height']}")
    pprint("======================")

    n = 40
    Xfs, Xhs = create_3D_data(n = n, l=config["l"])
    xfs, xhs = project_n_lines(Xfs, Xhs, noise = 4,
                            theta = config["theta"],
                            phi = config["phi"],
                            cam_pos = config["cam_pos"],
                            cam_w = config["cam_w"],
                            cam_h = config["cam_h"],
                            f = config["f"])

    # Implement calibration 
    calib_result = calibrate(xfs, xhs, config)

    # choose the results 
    calibration_method = "Vanilla model"
    f = calib_result[calibration_method]['f']
    theta = calib_result[calibration_method]['theta']
    phi = calib_result[calibration_method]['phi']
    h = calib_result[calibration_method]['height']
    xy = calib_result[calibration_method]['xy']
    x = xy.T[0]
    y = xy.T[1]
    pprint(f"2. initial guess")
    print("\n")
    pprint(f"focal length: {f}, theta:{np.rad2deg(theta)}, phi:{np.rad2deg(phi)}, h: {h}")
    pprint("======================")
    
    # Get estimated paramters from calibration as a optimization initial guess.
    params = np.array([f, theta, phi, h])
    params = np.hstack([params, x.flatten()])
    params = np.hstack([params, y.flatten()])


    # Reprojection error 
    func = ReprojectionError(xfs, xhs, n) 

    # Least-square 
    params_opt, RMS = get_leastsq(params, func)

    pprint(f"3. Optimization result")
    print("\n")
    pprint(f"focal length: {params_opt[0]}, theta:{np.rad2deg(params_opt[1])}, phi:{np.rad2deg(params_opt[2])}, h: {params_opt[3]}")
    pprint(f"RMS error: {RMS}")

