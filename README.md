
# PURPOSE 
This repository is intended to reorganize "Calibration from pedestrian" and implement experiment code for evaluate it 

## Structure 

Codes in this repo can be classified in 3 categories 


<1> Calibration and outlier detection algorithms 
  - This category is the main part of this repo. 
    catergory included 

    - calib_lines.py :
      - algorithm strongly based on "Camera calibration using parallel line segments" written by Gaku Nakano. 
      - RANSAC used in a 2 way (2lines/nlines)
      - Zscore and IQR method included (from outlier_detect)

    - outlier_detect.py : 
      - IQR and Zscore method is written 


<2> Extract Real data from video sequences 
  - To evaluate that the algorithms has acceptable performance in a wild, experiments with real data is necessary. This category included 
    - chessboard.py 
      - to get groundtruth the camera
      - cv2.calibrateCamera is used.
      - output: *calib_result_chs.json*
    - from_pedestrian.py
      - Used mediapipe.
      - can get head point and bottom point [in image plane (u,v)]
      - output 
         : *line_segment.json*, *line_segment_center.json*, *line_segment_leftside.json*
    - take_picorvid.py
      - Records Video 
      - output : * *.mp4*

<3> Experiment code 
 - experiment.py 
   - Evaluates accuracy of calibration algorithms
   - save graphs of accuracy
   - multiprocessing required bc many iteration for smoothing needed
   - DONE : synthetic data
   - TODO : video, public dataset data
   - output : exp_result.json
 - viz_result.py 
   - auxilary file for result visualization file Just in case 
  