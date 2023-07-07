import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json 
import matplotlib.pyplot as plt 
from calib_lines import calib_camera_nlines_ransac, calib_camera_stat, calib_camera_vanilla, calib_camera_ransac
from pprint import pprint
import multiprocessing
from multiprocessing import cpu_count
import argparse
import warnings
import cv2 
import os 
import statistics

# Put noises in data 
def gaussian_noise(x, mu, std):
    x_noisy = []
    for i in range(len(x)):
        noise = mu + std*np.random.randn(len(x[0])) 
        x_n = x[i] + noise
        x_noisy.append(x_n)
    x_noisy = [x[i] + np.random.normal(mu,std,size=x[0].shape) for i in range(len(x))]
    return x_noisy

# Convert Json configurations to parameter 
def json2params(filepath, 
              mode :str):
    
    with open(filepath,"r") as f: 
        config = json.load(f)
        config = config[mode]

        # deg2rad 
        if "theta" in config.keys():
            config["theta"] = np.deg2rad(config["theta"])
        
        if "phi" in config.keys():
            config["phi"] = np.deg2rad(config["phi"])

    return config

# Create synthetic line segmenet data 
def create_synthetic_data(n = 50, l = 1, noise = 2):

    # Draw lines  randomly
    As, Bs = [], []
    for _ in range(n):
        x = np.random.uniform(-2.5,2.5)
        y = np.random.uniform(1,5) 
        As.append(np.array([x,y,0]))
        Bs.append(np.array([x,y,l]))
    As, Bs = np.array(As), np.array(Bs)

    # Projecst line segments
    a_s = project_n_lines(As, 
                           theta = config["theta"],
                           phi = config["phi"],
                           cam_pos = config["cam_pos"],
                           cam_w = config["cam_w"],
                           cam_h = config["cam_h"],
                           f = config["f"])
    b_s = project_n_lines(Bs, 
                        theta = config["theta"],
                        phi = config["phi"],
                        cam_pos = config["cam_pos"],
                        cam_w = config["cam_w"],
                        cam_h = config["cam_h"],
                        f = config["f"])

    # Put noises in it 
    a_s = gaussian_noise(a_s, 0, noise)
    b_s = gaussian_noise(b_s, 0, noise)

    return As, Bs, a_s, b_s

# Project line segments to image plane 
def project_n_lines(Xs, **config):
    xs= [] 

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
    for X in Xs:
        x,_= cv2.projectPoints(X, r_gt.as_rotvec(), tvec_gt, K, np.zeros(4))
        xs.append(x.squeeze(1))
    
    return xs

# Get median value of exp results         
def get_median(f_list):
    median = statistics.median(f_list)
    return median

# Bind all calibration algorithms  
def calibrate(a,b,config):
    result = dict()
    # iqr
    iqr = calib_camera_stat(a, b, iqr = True, 
                            line_height = config["l"], 
                            cam_w = config["cam_w"], 
                            cam_h = config["cam_h"])
    
    # iqr + RANSAC
    ran = calib_camera_nlines_ransac(a, b, 
                                    line_height= config["l"],
                                    r_iter = 50,
                                    trsh= 0.0099,
                                    cam_w = config["cam_w"], 
                                    cam_h = config["cam_h"])
    
    # iqr + RANSAC_twolines 
    ran2 = calib_camera_ransac(a, b, 
                                line_height= config["l"],
                                r_iter = 50,
                                trsh= 0.0099,
                                cam_w = config["cam_w"], 
                                cam_h = config["cam_h"])
    
    van = calib_camera_vanilla(a,b,
                                line_heigth=config["l"],
                                cam_w = config["cam_w"],
                                cam_h = config["cam_h"])
    
    zsc = calib_camera_stat(a, b, 
                            iqr = False, 
                            line_height = config["l"], 
                            cam_w = config["cam_w"], 
                            cam_h = config["cam_h"])
    
    # zscor+ RANSAC
    ran_z = calib_camera_nlines_ransac(a, b, 
                                        iqr = False,
                                        line_height= config["l"],
                                        r_iter = 50,
                                        cam_w = config["cam_w"], 
                                        cam_h = config["cam_h"])
    
    # iqr + RANSAC_twolines 
    ran2_z= calib_camera_ransac(a, b, 
                                iqr = False,
                                line_height= config["l"],
                                r_iter = 50,
                                cam_w = config["cam_w"], 
                                cam_h = config["cam_h"])
    
    
    result["IQR"] = iqr
    result["RANSAC_IQR"] = ran
    result["RANSAC_IQR_2"] = ran2
    result["Vanilla model"] = van 
    result["ZSCORE"] = zsc
    result["RANSAC_ZSCORE"] = ran_z
    result["RANSAC_ZSCORE_2"] = ran2_z
    

    return result
