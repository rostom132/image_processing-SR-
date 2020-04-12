import cv2
import numpy as np
import imutils
import sys

# xa1 left x coordiante A
# ya1 top y coordinate A
# xa2 right x coordinate A
# ya2 bottom y coordinate A

# xb1 left x coordinate B
# yb1 top y coordinate B
# xb2 right x coordinate B
# yb2 bottom y coordinate B
# def calculateOverlapping(xa1, ya1, xa2, ya2, xb1, yb1, xb2, yb2):
#     left = max(xa1, xb1)
#     right = min(xa2, xb2)
#     bottom = min(ya2, yb2)
#     top = max(ya1, yb1)
#     if left < right and bottom > top:
#         interSection = (right - left) * (bottom - top)
#         Area_1 = (xa2 - xa1) * (ya2 - ya1)
#         Area_2 = (xb2 - xb1) * (yb2 - yb1)
#         Percent = (interSection / (Area_1 + Area_2 - interSection)) * 100
#         return Percent
#     return 0

# elif(res < 4 and res > 0): 
            #     percent = calculateOverlapping(ROI[0], ROI[1], ROI[0] + ROI[2], ROI[1] + ROI[3], x[0], y[0], x[2], y[2])
            #     cv2.putText(img2, 'Label is {:.0f} %'.format(percent), (0, 50), font, 0.55, (0,255,0), 2)
            #     print(percent)
            # elif(res == 0):
            #     print("Label wrong size")
            #     cv2.putText(img2, 'Label wrong size', (0, 50), font, 0.55, (0,255,0), 2)

# image = cv2.imread('./input/lix.jpg')
# h,w,_ = image.shape
# cv2.line(image,(w // 2, 0), (w // 2, h), (255,255,0), 5)

# cv2.imshow('Test', image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
# tracker_type = tracker_types[2]

# if tracker_type == 'KCF':
#         tracker = cv2.TrackerKCF_create()

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
# Read video
vs = cv2.VideoCapture("test.mp4")

# Initialize tracker with first frame and bounding box
tracker = OPENCV_OBJECT_TRACKERS["kcf"]() #Using CSRT
initBB = None

while True:
    _, frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)

    H,W,_ = frame.shape
    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x,y,w,h) = [int(v) for v in box]
            cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        initBB = cv2.selectROI(frame, False)
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break

video.realease()

cv2.destroyAllWindows()
