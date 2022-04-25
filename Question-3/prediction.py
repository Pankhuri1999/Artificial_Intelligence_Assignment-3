# Creation of actual, measured and predicted distnace
import cv2
from detector import Detector
from kalmanfilter import KalmanFilter
from tracker import Tracker

cap = cv2.VideoCapture("second.mp4")

# Load detector
od = Detector()

# Load Kalman filter to predict the trajectory
kf = KalmanFilter()
 

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    bbox = od.detect(frame)
    x, y, x2, y2 = bbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    predicted = kf.predict()
    (x1, y1) = kf.update([cx, cy])
    #cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 10)
    cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)
    cv2.circle(frame, (x1, y1), 20, (255, 255, 255), 4)

    cv2.putText(frame, "Actual Position", (cx+15, cy+15), 0, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Predicted Position", (predicted[0]-20, predicted[1]-20), 0, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Measured Position", (x1-40, y1-40), 0, 0.5, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(150)
    if key == 27:
        break