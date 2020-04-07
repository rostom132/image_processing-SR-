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
subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
Flag = False
count = 0
while True:

  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    text = "No bottle detected"
    frame = imutils.resize(frame, width=500)
    copy = frame.copy()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(frame, (21,21), 0)
    mask = subtractor.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (25,25), 0)
    # if firstframe is None:
    #     firstframe = gray
    #     continue
    # frameDelta = cv2.absdiff(firstframe, gray)
    # thresh = cv2.threshold(frameDelta, 40, 70, cv2.THRESH_BINARY)[1]
    #
    # thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.line(frame, (250,0), (250,250), (0, 0, 255), 2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 20000: #Bo may cai contour be hon 20000 pixel
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 100) and (h>=200):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx,cy = centroid(x,y,w,h)
            cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
            if (cx > 250) and (Flag == False):
                count += 1
                x_crop, y_crop, w_crop, h_crop = int(x), int(y), int(w), int(h)
                roi = copy[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
                cv2.imwrite("crop.jpg", roi)
                Flag = True
            text = "Bottle detected"
    cv2.putText(frame, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Count: {}".format(count), (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", mask)
    #cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow('Frame', frame)

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
