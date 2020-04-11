import cv2
import imutils
import numpy as np
def centroid(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return (cx,cy)
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('test.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video  file")

# Read until video is completed\
#firstframe = None
subtractor = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=220, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
Flag = False
count = 0
while True:

  # Capture frame-by-frame
  ret, frame = cap.read()
  frame_h, frame_w = frame.shape[:2]
  if ret == True:

    # Display the resulting frame
    text = "No bottle detected"
    frame_copy = imutils.resize(frame, width=500)
    hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
    lower_green = np.array([0, 110, 0])
    upper_green = np.array([179, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_green, upper_green)
    copy_h, copy_w = frame_copy.shape[:2]
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(frame, (21,21), 0)
    mask = subtractor.apply(mask_hsv)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    #mask = cv2.erode(mask, kernel, iterations = 1)
    #mask = cv2.dilate(mask, kernel, iterations=1)
    cv2.line(frame_copy, (int(copy_w*0.5),0), (int(copy_w*0.5),copy_h), (0, 0, 255), 2)
    cv2.rectangle(frame_copy, (180, 21), (180 + 140, 21 + 187), (0, 255, 0), 2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 10000: #Bo may cai contour be hon 20000 pixel
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 100) and (h>=150):
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx,cy = centroid(x,y,w,h)
            cv2.circle(frame_copy, (cx,cy), 5, (0, 0, 255), -1)
            if (cx > 250) and (Flag == False):
                count += 1
                x_crop, y_crop, w_crop, h_crop = int(x), int(y), int(w), int(h)
                roi = frame[int(21*(frame_h/copy_h)):int(21*(frame_h/copy_h)+187*(frame_h/copy_h)), int(180*(frame_w/copy_w)):int(180*(frame_w/copy_w)+140*(frame_w/copy_w))]
                cv2.imwrite("crop.jpg", roi)
                Flag = True
            text = "Bottle detected"
    cv2.putText(frame_copy, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame_copy, "Count: {}".format(count), (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", mask)
    #cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow('Frame', frame_copy)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
