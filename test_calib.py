import numpy as np 
from calib_lines import calib_camera_ransac, calib_camera_nlines_ransac
import json 
import cv2 
from calib_lines_stat import calib_camera_iqr

with open('line_segment.json','r') as f: 
    from_file = json.load(f)
    a = from_file['a']
    b = from_file['b']
    W = from_file['cam_w']
    H = from_file['cam_h']



focal_length = 558.2434389349814
# Calibration results

results = calib_camera_ransac(a,
                              b,
                              line_height=1,
                              r_iter = 200,
                              trsh =4.5,
                              cam_w = W,
                              cam_h = H
                              )



# results = calib_camera_iqr(a,b,line_height=1,
#                            iqr = False,
#                            cam_w = W,
#                            cam_h = H)

#results = calib_camera_nlines_ransac(a, b, 1, cam_w = W, cam_h = H)

f = results['f']
print(f"grountruth: {focal_length}")
print("---------------------------")
print(f'calibration result: {f}')
print(f'{abs(f-focal_length)*100/focal_length} %')



