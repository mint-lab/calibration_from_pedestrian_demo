import numpy as np
import cv2
from mediapipe.python import solutions
import json
import os 
import argparse
import matplotlib.pyplot as plt 
import warnings
mp_pose = solutions.pose
mp_holistic =solutions.holistic
mp_drawing_styles = solutions.drawing_styles
pose = mp_pose.Pose()

# Ignore Warnings
warnings.filterwarnings("ignore")

# Read Videoo 
parser = argparse.ArgumentParser()
parser.add_argument('--index','-i',type=str, default='syn', help="index of video")
FILENUMBER = parser.parse_args().index
PATH = "/home/chahnoseo/panoptic-toolbox/scripts/160401_ian2/hdVideos"
videos = os.listdir(PATH)
videos = [os.path.join(PATH, video) for video in videos]
for i, video in enumerate(videos):
    # Video
    cap = cv2.VideoCapture(video)
    if os.path.isfile(video):
        cap = cv2.VideoCapture(video)
    else:
        print("The video file doesn't exists")


    #Read resolution of video 
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
        if len(head_list) > 500:
            break



        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('mediapipe results',image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Release current VideoCapture instance
    cap.release()
    # Close window of current video 
    cv2.destroyAllWindows()

    if "center" in video:
        json_name = "metadaline_segment_center.json"
    elif "leftside" in video:
        json_name = "metadata/line_segment_leftside.json"
    elif "panoptic" in video:
        json_name = "metadata/line_seg_panoptic"+"_"+str(i)+".json"
        
    with open(json_name,'w') as f:
        result = dict()
        result['cam_w'] = cam_w
        result['cam_h'] = cam_h
        result['a'] = bottom_list
        result['b'] = head_list
        result['l'] = 0.5
        
        json.dump(result,f,indent=4)



