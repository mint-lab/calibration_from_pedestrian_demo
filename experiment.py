import cv2 
import statistics
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import pandas as pd 
import json 
from calib_lines import calib_camera_nlines_ransac, calib_camera_stat
from pprint import pprint
config ={
        "cam_f" : 1000, 
        "l" : 1.,  
        "h" : 3,   #[m]
        "cam_pos" : [0, 0, 3],
        "theta_gt" : np.deg2rad(40+90), 
        "phi_gt" : np.deg2rad(10),  
        "cam_w" : 1920, 
        "cam_h" : 1080
}



def load_json2exp(filepath = 'line_segment.json'):
    with open(filepath,'r') as f: 
        from_file = json.load(f)
        a = from_file['a']
        b = from_file['b']
        W = from_file['cam_w']
        H = from_file['cam_h']





def create_synthetic_data(n = 50, l = 1):

    # Draw lines  randomly
    lines = []
    for i in range(n):
        x = np.random.uniform(-2.5,2.5)
        y = np.random.uniform(2,15) #Important to calculate Working distance correctly  
        lines.append(np.array([[x,y,0],
                               [x,y,l]]))
        
    # Projecst line segments
    data = project_n_lines(lines, 
                           theta_gt = config["theta_gt"],
                           phi_gt = config["phi_gt"],
                           cam_pos = config["cam_pos"],
                           cam_w = config["cam_w"],
                           cam_h = config["cam_h"],
                           cam_f = config["cam_f"])

    # Put noises in it 
    data = gaussian_noise(data,0,2)

    a = [aa[0] for aa in data]
    b = [bb[1] for bb in data]

    return a , b 

def project_n_lines(lines:list, **config):
    
    p_lines= [] 

    Rx = np.array([[1., 0, 0], [0, np.cos(config["theta_gt"]), -np.sin(config["theta_gt"])], [0, np.sin(config["theta_gt"]), np.cos(config["theta_gt"])]])
    Rz = np.array([[np.cos(config["phi_gt"]), -np.sin(config["phi_gt"]), 0], [np.sin(config["phi_gt"]), np.cos(config["phi_gt"]), 0], [0, 0, 1.]])
    R_gt =  Rz @ Rx 
    r_gt = Rotation.from_matrix(R_gt)

    tvec_gt = -R_gt @ config["cam_pos"]

    K =np.array([[config["cam_f"], 0., config["cam_w"]/2], [0., config["cam_f"], config["cam_h"]/2], [0., 0., 1.]]) 

    for line in lines: #cv.projectpoints
        x,_= cv2.projectPoints(line, r_gt.as_rotvec(), tvec_gt, K, np.zeros(4))
        p_lines.append(x.squeeze(1))
    
    return p_lines
          
def gaussian_noise(x, mu, std):

    x_noisy = []
    for i in range(len(x)):
        
        noise = mu + std*np.random.randn(len(x[0])) 
        x_n = x[i]+noise
        x_noisy.append(x_n)
    x_noisy = [x[i] + np.random.normal(mu,std,size=x[0].shape) for i in range(len(x))]
    return x_noisy

def get_median(f_list:list,
               theta_gt_list:list,
               phi_gt_list:list,
               height_list:list):

    #Get median value of 1000 trials in n lines 
    pack = (statistics.median(f_list),
             statistics.median(theta_gt_list),
             statistics.median(phi_gt_list),
             statistics.median(height_list))
    
    return pack 


if __name__ =="__main__" : 

    # get Dataset (1. Synthetic | 2. recorded video by my own | 3. Public datasets)

    # Synthetic
    a, b = create_synthetic_data(n = 240)

    # iqr
    ret_list = [] 
    ret = calib_camera_stat(a, b, iqr =True, 
                            line_height = config["l"], 
                            cam_w = config["cam_w"], 
                            cam_h = config["cam_h"])
    


