import numpy as np
import cv2
from mediapipe.python import solutions
import json
import os 
import argparse
import matplotlib.pyplot as plt 
mp_pose = solutions.pose
mp_holistic =solutions.holistic
mp_drawing_styles = solutions.drawing_styles
pose = mp_pose.Pose()

#Read Videoo 
path ='/home/chahnoseo/video_for_exp'
filename = "data_miraehall_center.mp4"
filePath = os.path.join(path, filename)
cap = cv2.VideoCapture(filePath)

if os.path.isfile(filePath):
    cap = cv2.VideoCapture(filePath)
else:
    print("The video file doesn't exists")

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	

# List to store head (u,v)
head_list = []
# List to store bottom (u,v)
bottom_list = []
t = 0

while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        final = image
        break
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False    

    results = pose.process(image)
    
    if not results.pose_landmarks:
        continue

    # Get line segment 
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    landmarks = results.pose_landmarks.landmark

    shoulder_x = int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * cam_w / 2)
    shoulder_y = int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)  * cam_h / 2)
    
    head = np.array((shoulder_x,shoulder_y))

    hip_x = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)  * cam_w / 2 )
    hip_y = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) * cam_h / 2 )
    
    bottom = np.array((hip_x,hip_y))
    
    cv2.line(image, head ,bottom , color=(0,0,255))

    head_list.append(head.tolist())
    bottom_list.append(bottom.tolist())
    # 보기 편하게 이미지를 좌우 반전합니다.
    cv2.imshow('mediapipe results',image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    

# Output 
# 1. cam_w, cam_h
# 2. pixel point of head and bottom point of line segment 

if "center" in filename:
    json_name = "line_segment_center.json"
elif "leftside" in filename:
    json_name = "line_segment_leftside.json"
    
with open(json_name,'w') as f:
    result = dict()
    result['cam_w'] = cam_w
    result['cam_h'] = cam_h
    result['a'] = bottom_list
    result['b'] = head_list
    
    json.dump(result,f,indent=4)



