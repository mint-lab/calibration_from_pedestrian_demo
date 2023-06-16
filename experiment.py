import cv2 
from collections import defaultdict
import statistics
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
import os 
METADATA = "metadata/"
CONFIG = METADATA + "calib_synthetic.json"
LINESEGDATA = METADATA+"line_seg_panoptic.json"


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


def create_synthetic_data(n = 50, l = 1, noise = 2):

    # Draw lines  randomly
    lines = []
    for i in range(n):
        x = np.random.randint(-2.5,2.5)
        y = np.random.randint(2,5) #Important to calculate Working distance correctly  
        lines.append(np.array([[x,y,0],
                               [x,y,l]]))

    # Projecst line segments
    data = project_n_lines(lines, 
                           theta = config["theta"],
                           phi = config["phi"],
                           cam_pos = config["cam_pos"],
                           cam_w = config["cam_w"],
                           cam_h = config["cam_h"],
                           f = config["f"])

    # Put noises in it 
    data = gaussian_noise(data, 0, noise)

    a = [aa[0] for aa in data]
    b = [bb[1] for bb in data]

    return a , b 

def project_n_lines(lines:list, **config):
    p_lines= [] 

    Rx = np.array([[1., 0, 0], [0, np.cos(config["theta"]), -np.sin(config["theta"])], [0, np.sin(config["theta"]), np.cos(config["theta"])]])
    Rz = np.array([[np.cos(config["phi"]), -np.sin(config["phi"]), 0], [np.sin(config["phi"]), np.cos(config["phi"]), 0], [0, 0, 1.]])
    R_gt =  Rz @ Rx 
    r_gt = Rotation.from_matrix(R_gt)

    tvec_gt = -R_gt @ config["cam_pos"]

    K =np.array([[config["f"], 0., config["cam_w"]/2], [0., config["f"], config["cam_h"]/2], [0., 0., 1.]]) 
    for line in lines: #cv.projectpoints
        x,_= cv2.projectPoints(line, r_gt.as_rotvec(), tvec_gt, K, np.zeros(4))
        p_lines.append(x.squeeze(1))
    
    return p_lines
          
def gaussian_noise(x, mu, std):
    x_noisy = []
    for i in range(len(x)):
        noise = mu + std*np.random.randn(len(x[0])) 
        x_n = x[i] + noise
        x_noisy.append(x_n)
    x_noisy = [x[i] + np.random.normal(mu,std,size=x[0].shape) for i in range(len(x))]
    return x_noisy

def get_median(f_list):
    median = statistics.median(f_list)
    return median

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
                                    r_iter = 200,
                                    cam_w = config["cam_w"], 
                                    cam_h = config["cam_h"])
    
    # iqr + RANSAC_twolines 
    ran2 = calib_camera_ransac(a, b, 
                                line_height= config["l"],
                                r_iter = 200,
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
                                        r_iter = 200,
                                        cam_w = config["cam_w"], 
                                        cam_h = config["cam_h"])
    
    # iqr + RANSAC_twolines 
    ran2_z= calib_camera_ransac(a, b, 
                                iqr = False,
                                line_height= config["l"],
                                r_iter = 200,
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


if __name__ =="__main__" : 

    # Ignore warings in numpy 
    warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    # Get Dataset (1. Synthetic | 2. recorded video by my own | 3. Public datasets)
    
    # Experiment options 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d',type=str, default='syn', help="What type of dataset when do experiment")
    parser.add_argument('--iv', '-i',type=str,default ="num", help = "Indepenedent Variable: One is number of lines and the other is noise levl")
    parser.add_argument("--file", '-f', type = str,default=" ")
    args = parser.parse_args()
    dataset = args.dataset
    LINESEGDATA = METADATA + args.file
    
    # Synthetic
    n = 10
    iters = 10
    noise_limit = 10 

    errors = defaultdict(lambda: defaultdict(list))
    med = defaultdict(lambda: defaultdict(list))
    params = ["f",
              "theta",
              "phi",
              "height"]
 

    save = dict()

    # Implement Experiments : Focal length by the number of line segment 
    if dataset == "syn":
        config = json2params(CONFIG, dataset)
        if args.iv == "num":
            for i in tqdm(range(2,n)):
                # To evaluate Properly, Randomness derived from the algorithms need be eliminated => Evaluate by median value of a lot of trials
                for iter in tqdm(range(iters)): 
                    pool = multiprocessing.Pool(processes = int(2 * cpu_count()))
                    a, b = create_synthetic_data(n = i ,l = config["l"])
                    args = (a, b, config)
                    result = pool.apply_async(calibrate, args).get()
                    methods = result.keys()
                    
                    for method in methods:
                        if result[method] !="nan":
                            for param in params:
                                error = np.abs(100*(result[method][param]-config[param])/config[param])
                                errors[method][param].append(error)
                    
                    pool.close()    
                    pool.join()


                # After 100 trials, Get median values 
                for m in methods:
                    for p in params:
                        med[m][p].append(float(get_median(errors[m][p])))
            
            save["median"] = med 
            with open("metadata/exp_result_syn.json","w") as f:
                json.dump(save,f)
                print(f"Synthetic data 1: independent variable is number of line segemnts"
                      f"....Experiment completed !")
        
        elif args.iv == "noise":
            for i in tqdm(range(noise_limit)):
                # To evaluate Properly, Randomness derived from the algorithms need be eliminated => Evaluate by median value of a lot of trials
                for iter in tqdm(range(iters)): 
                    # Multiprocessing code to boost the speed of iteration
                    pool = multiprocessing.Pool(processes = 16)

                    # Create Data along the noise level 
                    a, b = create_synthetic_data(n = 100, 
                                                 noise = i, 
                                                 l = config["l"])
                    args = (a, b, config)
                    result = pool.apply_async(calibrate, args).get()
                    methods = result.keys()
                    
                    for method in methods:
                        if result[method] !="nan":
                            for param in params:
                                error = np.abs(100*(result[method][param]-config[param])/config[param])
                                errors[method][param].append(error)
                    
                    pool.close()    
                    pool.join()


                # After 1000 trials, Get median values 
                for m in methods:
                    for p in params:
                        med[m][p].append(float(get_median(errors[m][p])))
            
            save["median"] = med 
            with open("metadata/exp_result_syn_noise.json","w") as f:
                json.dump(save,f)

    elif dataset == "vid":
        iter = 100
        f_list = defaultdict(list)
        config = dict()             
        with open(LINESEGDATA ,'r') as f: 
            from_file = json.load(f)
            a = from_file['a'][:400]
            b = from_file['b'][:400]
            pprint(len(a))
            config = from_file   
            # temporary
            config["l"] = 0.5         
            print(f'{config["cam_w"]} X {config["cam_h"]}')

        results = []

        for i in tqdm(range(iter)):
            args = (a, b, config)
            pool = multiprocessing.Pool(processes=cpu_count()-1)
            result = pool.apply_async(calibrate, args).get()
        
            for m in result.keys():
                try:
                    f_list[m].append(result[m]["f"])
                except: None

        pool.close()
        pool.join()
        
        #tqdm library can effects the speed of iteration code
        for m in f_list.keys():
            print(f'{m}: {get_median(f_list[m])}')
            
        with open("metadata/video_result.json","w") as f:
                json.dump(f_list,f)

    elif dataset == "public":
        import pandas as pd 
        import os 

        # Load Panoptic line segments
        CONFIG_LINES = [METADATA + "line_seg_panoptic_" + str(i) +".json"
                        for i in range(5)]
       
        # Experiment setting
        iter = 1000
        f_list = defaultdict(list)
        config = dict()  
        df = pd.DataFrame(columns = CONFIG_LINES)
   
        
        for i, CONFIG_LINE in enumerate(CONFIG_LINES):
            with open(CONFIG_LINE ,'r') as f: 
                from_file = json.load(f)
                a = from_file['a']
                b = from_file['b']
                a_np = np.asarray(a)
                b_np = np.asarray(b)
                
                # Choose 100 data Randomly without replacement
                np.random.seed(717)
                random_idx = np.random.choice(500, 100, replace = False)
            
                # Create filter mask
                filter_size = a_np[...,0]
                filter_mask = np.zeros_like(filter_size, dtype = bool)
                filter_mask[random_idx] = True
                
                # Filter a and b 
                a_np = a_np[filter_mask]
                b_np = b_np[filter_mask]
                
                print(f"data proceeding at...{CONFIG_LINE}")
                print(f'num of a:{len(a_np)}')
                print(f'num of b:{len(b_np)}')
                
                config = from_file
                print(f'{config["cam_w"]} X {config["cam_h"]}')
            
            pool = multiprocessing.Pool(processes = int(1.5 * cpu_count()))
            results = []

            for i in tqdm(range(iter)): 
                args = (a_np, b_np, config)
                result = pool.apply_async(calibrate, args).get()
            
                for m in result.keys():
                    try:
                        f_list[m].append(result[m]["f"])
                    except: None
          
            pool.close()
            pool.join()
            
            
            if i == 0:
                df.index = f_list.keys()
                # Allocate using row and col
            for method in f_list.keys():
                df.loc[method, CONFIG_LINE] = get_median(f_list[method])
            df.to_csv("result/panoptic.csv")        
        



