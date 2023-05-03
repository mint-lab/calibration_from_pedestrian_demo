import numpy as np
import cv2 
import matplotlib.pyplot as plt 
import json

from pathlib import Path 
from calibration.calib_lines import calib_camera_nlines_ransac, calib_camera_ransac
from calibration.calib_lines_stat import calib_camera_stat


if __name__ =="__main__" : 

    # get Dataset from Json
    
    # Camera 1 : 
    # Public Data Set 
