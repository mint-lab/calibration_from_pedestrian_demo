import cv2
import numpy as np
import copy
import yaml

web_cam = 2 
board_pattern = (8,6)
select_images = True

capture = cv2.VideoCapture(web_cam)
capture.set(cv2.CAP_PROP_AUTOFOCUS,0)

images = []

while capture.isOpened():
    ret1, image = capture.read()

    if select_images:
        cv2.imshow("3DV Tutorial: Camera Calibration", image)
        key = cv2.waitKey(1)
        if key == 27: break # 'ESC' key: Exit
        elif key == 32:     # 'Space' key: Pause
            ret2, pts = cv2.findChessboardCorners(image, board_pattern, None) # No flags
            # criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            # pts = cv2.cornerSubPix(image, pts, (5,5), (-1,-1), criteria=criteria)
    
            # display = image.clone()
            display = copy.deepcopy(image)
            display = cv2.drawChessboardCorners(display, board_pattern, pts, ret2)
            cv2.imshow("3DV Tutorial: Camera Calibration", display)
            key = cv2.waitKey() # ESC 
            if key == 27: break
            elif key == 13: images.append(image) # 'Enter' key: Save
    else: 
        images.append(image)


capture.release()

if(len(images)) == 0:
    print("no images")
    raise Exception("There is no captured images!")

# Find 2D corner points from given images
img_points = []
h, w = 0,0
for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ret3, corners = cv2.findChessboardCorners(gray, board_pattern) # No flags
    criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria=criteria)
    
    if ret3 == True:
        img_points.append(corners)

if len(img_points) == 0:
    raise Exception("No 2d Corner pts")

# Prepare 3D points of the chess board
objp = np.zeros((8*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
obj_points = []
for _ in images:
    obj_points.append(objp)

# Calibrate Camera
K = np.eye(3,3, dtype=np.float32)
dist_coeff = np.zeros((4,1))
flags = cv2.CALIB_USE_EXTRINSIC_GUESS| cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (h,w), None, None,flags = flags)

# Report calibration results
print("## Camera Calibration Results")
print(f"* The number of applied images = {w}x{h}")
print(f"* RMS error = {rms}")
print(f"* Camera matrix (K) = \n{K}")
print(f"* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff}")

# Save as cam_config.yaml
file_name = "calib_result_chs.json"
cam_dict = {
    "Intrinsic": K.flatten().tolist(),
    "Distortion": dist_coeff.flatten().tolist(),
    "RMS": rms,
}
with open(file_name, 'w') as f:
    yaml.dump(cam_dict, f, sort_keys=False, default_flow_style=False)

print("End!")