from pprint import pprint
import argparse
import json 
import numpy as np 
import numpy as np 
import matplotlib.pyplot as plt 


def visualize_result(x_value:np.array,
                    y_value:dict,
                    title:str):
    methods=["IQR",
    "RANSAC_IQR",
    "RANSAC_IQR_2",
    "Vanilla model",
    "ZSCORE",
    "RANSAC_ZSCORE",
    "RANSAC_ZSCORE_2"]

    params = ["f",
              "theta",
              "phi",
              "height"]
   
    plt.title(title+" of models")
    plt.figure()

    for i, param in enumerate(params):
        plt.subplot(int(len(params)/2)+1, int(len(params)/2),i+1)
        for _ , method in enumerate(methods):
            plt.plot(x_value, y_value[method][param], label = method)
            if i == 0:
                plt.legend(fontsize = 7, bbox_to_anchor = (1,-1.5))
            else: pass 
            plt.ylim([0,100])
            plt.ylabel(f"accuarcy of {param}")
    plt.savefig(f"result/{title}1.png")


if __name__ == "__main__":
    with open("metadata/exp_result_tmp.json",'r') as f:
        x = json.load(f)
        x = x["median"]
    
    visualize_result(np.arange(2,400),x,"Synthetic data experiment")
    