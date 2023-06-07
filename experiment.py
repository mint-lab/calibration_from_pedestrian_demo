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
METADATA = "metadata/"
CONFIG = METADATA + "config.json"
LINESEGDATA = METADATA+"line_seg_panoptic.json"


def json2params(filepath, 
              mode:str):
    with open(filepath,"r") as f: 
        config = json.load(f)
        config = config[mode]

        # deg2rad 
        if "theta_gt" in config.keys():
            config["theta_gt"] = np.deg2rad(config["theta_gt"])
        
        if "phi_gt" in config.keys():
            config["phi_gt"] = np.deg2rad(config["phi_gt"])

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
                           theta_gt = np.deg2rad(config["theta_gt"]),
                           phi_gt = np.deg2rad(config["phi_gt"]),
                           cam_pos = config["cam_pos"],
                           cam_w = config["cam_w"],
                           cam_h = config["cam_h"],
                           cam_f = config["cam_f"])

    # Put noises in it 
    data = gaussian_noise(data, 0, noise)

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

def get_median(f_list):

    #Get median value of 1000 trials in n lines 
    median = statistics.median(f_list)

    return median

def do (a,b,config):
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

count = 0 
def visualize_result(x_value:np.array,
                    y_value:dict,
                    title:str):
   
    plt.title(title+" of models")
    plt.figure()
    for method in methods:
        plt.plot(x_value, y_value[method], label = method)
        plt.legend()
    
    plt.savefig(f"result/{title}.png")



if __name__ =="__main__" : 

    # get Dataset (1. Synthetic | 2. recorded video by my own | 3. Public datasets)
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset','-d',type=str, default='syn', help="What type of dataset when do experiment")
    parser.add_argument('--iv', '-i',type=str,default ="num", help = "Indepenedent Variable: One is number of lines and the other is noise levl")
    parser.add_argument("--file", '-f', type = str)
    args = parser.parse_args()
    dataset = args.dataset
    LINESEGDATA = METADATA + args.file
    # Synthetic
    n = 10
    trials = 10
    noise_limit = 10 
    thousands = defaultdict(list)
    med = defaultdict(list)
    
    success_rate = defaultdict(list)
    save = dict()

    # Implement Experiments : Focal length by the number of line segment 
    if dataset == "syn":
        config = json2params(CONFIG, dataset)
        if args.iv == "num":
            for i in tqdm(range(2,n)):
                # To evaluate Properly, Randomness derived from the algorithms need be eliminated => Evaluate by median value of a lot of trials
                success = defaultdict(int)
                for trial in range(trials): 
                    
                    a, b = create_synthetic_data(n = i ,l = config["l"])
                    result = do(a,b)
                    methods = result.keys()
                    
                    for m in methods:
                        if result[m] !="nan":
                            relative_err = 100*(result[m]['f']-config["cam_f"])/config["cam_f"]
                            relative_err = abs(relative_err)
                            thousands[m].append(relative_err)

                            if abs(relative_err) < 10: 
                                success[m]+=1

                    # Store success rate 
                for m in methods:
                    success_rate[m].append(success[m]/trials) 


                # After 100 trials, Get median values 
                for m in methods:
                    med[m].append(float(get_median(thousands[m])))
            
            save["median"] = med 
            save["success_rate"] = success_rate
            with open("metadata/exp_result_tmp.json","w") as f:
                json.dump(save,f)
                

            # Show Result
            visualize_result(np.arange(2,n), med, "Accuracy")

      
    # Implement Experiments :  Focal length by Noise level(std of normal distribution)                
        elif args.iv == "noise":
            methods=["IQR",
                    "RANSAC_IQR",
                    "RANSAC_IQR_2",
                    "Vanilla model",
                    "ZSCORE",
                    "RANSAC_ZSCORE",
                    "RANSAC_ZSCORE_2"]
            for i in tqdm(range(1, noise_limit)):
                success = defaultdict(int)
                pool = multiprocessing.Pool(processes=cpu_count()-1) # 병렬 처리할 프로세스 수를 지정하세요

                results = []
                for trial in range(trials):
                    a, b = create_synthetic_data(n=50, l=config["l"])
                    results.append(pool.apply_async(do, (a, b)))

                for result in results:
                    r = result.get()
                    for m in methods:
                        if r[m] != "nan":
                            relative_err = 100 * (r[m]['f'] - config["cam_f"]) / config["cam_f"]
                            relative_err = abs(relative_err)
                            thousands[m].append(relative_err)

                            if relative_err < 10:
                                success[m] += 1

                pool.close()
                pool.join()
                # After 100 trials, Get median values 
            for m in methods:
                med[m].append(float(get_median(thousands[m])))
            
            save["median"] = med 
            with open("metadata/exp_noise_result_tmp.json","w") as f:
                json.dump(save,f)
                

            # Show Result
            visualize_result(np.arange(2,noise_limit), med, "Noise_Accuracy")

           


                    
                

    elif dataset == "vid":
        iter = 1000   
        f_list = defaultdict(list)
        config = dict()             
        with open(LINESEGDATA ,'r') as f: 
            from_file = json.load(f)
            a = from_file['a'][:400]
            b = from_file['b'][:400]
            
            config = from_file
            print(f'{config["cam_w"]} X {config["cam_h"]}')
        for i in tqdm(range(iter)):
            
            result = do(a,b,config)
            for m in result.keys():
                try:
                    f_list[m].append(result[m]["f"])
                except: breakpoint()
            
        with open("metadata/video_result.json","w") as f:
                json.dump(f_list,f)
        

    elif dataset == "public":
        def process_data(x):
            a, b, config = x 
            return do(a, b, config)
            
        iter = 100  
        f_list = defaultdict(list)
        config = dict()             
        with open(LINESEGDATA ,'r') as f: 
            from_file = json.load(f)
            a = from_file['a'][:400]
            b = from_file['b'][:400]
          
            print(f'num of a:{len(a)}')
            print(f'num of b:{len(b)}')
            
            config = from_file
            print(f'{config["cam_w"]} X {config["cam_h"]}')
    
        pool = multiprocessing.Pool(processes=cpu_count()-1)
        results = []

        for i in tqdm(range(iter)):
            args = (a, b, config)
            result = pool.apply_async(do, args).get()
        
            for m in result.keys():
                try:
                    f_list[m].append(result[m]["f"])
                except: None

        pool.close()
        pool.join()
        
        #tqdm library can effects the speed of iteration code
        for m in f_list.keys():
            print(f'{m}: {get_median(f_list[m])}')

        with open("metadata/public_result.json","w") as f:
                json.dump(f_list,f)
        
        
    



