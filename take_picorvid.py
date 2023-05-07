import cv2


cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
i = 1
while cap.isOpened():
    ret, img = cap.read()
    cv2.imshow('image',img)
    
    if cv2.waitKey(1) &0xFF == 32:
        cv2.imwrite(f'calibration/images/calib{i}.png', img)
        i += 1
    elif cv2.waitKey(1) &0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    

cap.release()