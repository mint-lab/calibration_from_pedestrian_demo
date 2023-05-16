from pprint import pprint
import argparse
import json 
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
   
    plt.title(title+" of models")
    plt.figure()
    for method in methods:
        if method == "ZSCORE":
            plt.plot(x_value, y_value[method],linewidth=4,label = method )
        else:
            plt.plot(x_value, y_value[method], label = method)
    
        plt.legend()
        plt.ylim([0,100])
    pprint(y_value["ZSCORE"])
    plt.savefig(f"result/{title}1.png")


if __name__ == "__main__":
    with open("exp_noise_result.json",'r') as f:
        x = json.load(f)
        ret = x["median"]
       
    visualize_result(np.arange(2,100),ret, "Accuracy")
    