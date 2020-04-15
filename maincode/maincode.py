import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from objecttracking import CentroidTracker
from Tracker import TrackableObject
import os.path
import csv
import sys
import threading

INPUT_VIDEO = 'input/final.mp4'
INPUT_LABEL = './input/label_2.png'
SCALE_LABEL = 20
MIN_MATCH_COUNT = 50

ct = CentroidTracker()
(H, W) = (None, None)
trackers = []
trackableObjects = {}


initROI = True
ROI = []
ROI_LABEL = [0 for i in range(4)]
if len(sys.argv) > 1:
	if sys.argv[1] == "--ROI":
		initROI = False 
elif os.path.isfile('ROI.csv'):
	initROI = True
	with open('ROI.csv', 'rt') as csvfile: 
		csvreader = csv.reader(csvfile,delimiter='\t') 
		for row in csvreader: 
			ROI.append([int(i) for i in row])
else:
	initROI = False

# LABEL
font = cv2.FONT_HERSHEY_SIMPLEX 
label = cv2.imread(INPUT_LABEL)
scale_percent = SCALE_LABEL # percent of original size
width = int(label.shape[1] * scale_percent / 100)
height = int(label.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized_label = cv2.resize(label, dim, interpolation = cv2.INTER_AREA)
label_gray= cv2.cvtColor(resized_label, cv2.COLOR_BGR2GRAY)
h,w = label_gray.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(label_gray, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

def compareCoor(roi, x, y):
    logic = roi[0] < x < roi[0] + roi[2] and roi[1] < y < roi[1] + roi[3]
    return logic


def get_template(img2):
    global ROI, MIN_MATCH_COUNT, font, pts, kp1, des1, flann, count

    # Find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)

    print("Keypoints in 1st Image: " + str(len(kp1)))
    print("Keypoints in 2nd Image: " + str(len(kp2)))

    matches = flann.knnMatch(des1,des2,k=2)

    # Store all good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
                good.append(m)

    number_keypoints = 0
    if(len(kp1) <= len(kp2)):
        number_keypoints = len(kp1)
    else:
        number_keypoints = len(kp2)

    number_goodpoints = len(good)
    print("Good matches found: " + str(number_goodpoints))
    similariy_percentage = float(number_goodpoints) / number_keypoints * 100
    print(similariy_percentage)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
       

        dst = cv2.perspectiveTransform(pts,M)

        x = []
        y = []
        (rows, columns) = (dst.shape[0], dst.shape[1])
        for i in range(rows):
            array = dst[i][0]
            x.append(array[0])
            y.append(array[1])

        res = 0
        if(initROI == True):
            for i in range(rows):
                res = res + 1 if (compareCoor(ROI[1], x[i], y[i]) == True) else res + 0   
            if(res == 4):
                print("Label is correct")
                cv2.putText(img2, 'Label is pasted correct', (0, 50), font, 1, (0,255,0), 2)
                cv2.putText(img2, 'Input sample: {:.0f} %'.format(similariy_percentage), (0, 100), font, 1, (0,255,0), 2)
                count += 1
            else:
                print("Label is pasted wrong")
                cv2.putText(img2, 'Label is pasted wrong', (0, 50), font, 1, (0,255,0), 2)
                cv2.putText(img2, 'Input sample: {:.0f} %'.format(similariy_percentage), (0, 100), font, 1, (0,255,0), 2) 
        else:
            print("There is no ROI reference to check")
            cv2.putText(img2, 'No ROI found', (0, 50), font, 1, (0,255,0), 2)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
    else:
        cv2.putText(img2, 'No label found', (0, 50), font, 1, (0,255,0), 2) 
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None


    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                 singlePointColor = None,
    #                 matchesMask = matchesMask, # draw only inliers
    #                 flags = 2)
    # img3 = cv2.drawMatches(label_gray,kp1,img2,kp2,good,None,**draw_params)
    # plt.imshow(img3, 'gray'),plt.show()

    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyWindow('gray')

def check_label(img1):
    global initROI, label_gray, font
    
    sample = img1
    sample_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    # # Second step: Get ROI of zone for checking the position of the label
    cv2.rectangle(sample_gray, (ROI[1][0], ROI[1][1]), (ROI[1][0] + ROI[1][2], ROI[1][1] + ROI[1][3]), (0,255,0), 3)

    # Third step: Execute the function
    get_template (sample_gray)

    # key = cv2.waitKey(0) & 0xFF
    # if key == ord("q"):
    #     cv2.destroyAllWindows()

def check_bottle(frame):
    global ROI, objects, text_1, initROI
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame_copy, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame_copy, (centroid[0], centroid[1]), 4, (0, 255, 0), -1) 
        to = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
        elif (not to.counted) and (centroid[0] > 250):
            to.counted = True
            x_crop, y_crop, w_crop, h_crop = int(x), int(y), int(w), int(h)
            roi = frame[int(ROI[0][1]):int(ROI[0][1]+ROI[0][3]), int(ROI[0][0]):int(ROI[0][0]+ROI[0][2])]
            print("hello")
            check_label(roi)
            text_1 = "Bottle detected"
            text = "ID {}".format(objectID)
            print(text)
        trackableObjects[objectID] = to

def centroid_detect(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return (cx,cy)
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(INPUT_VIDEO)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video  file")

# Read until video is completed\

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
Flag = False
count = 0
text_1 = "No bottle detected"
firstframe = None
lower_green = np.array([30, 130, 0])
upper_green = np.array([179, 255, 255])
check_bottle_flag = False

while True:
  # Capture frame-by-frame
  
  ret, frame = cap.read()
  frame_h, frame_w = frame.shape[:2]
  propotion =  500 / frame_w

  if ret == True:
    rects = []
    # Display the resulting frame
    text = "No bottle detected"
    frame_copy = imutils.resize(frame, width=500)
    hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)

    mask_hsv = cv2.inRange(hsv, lower_green, upper_green)
    mask = mask_hsv.copy()
    copy_h, copy_w = frame_copy.shape[:2]
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(mask, (3,3), 0)
    if firstframe is None:
        firstframe = gray
        continue
    frameDelta = cv2.absdiff(firstframe, gray)
    mask = frameDelta
    # mask = subtractor.apply(mask_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    #mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    cv2.line(frame_copy, (int(copy_w*0.5),0), (int(copy_w*0.5),copy_h), (0, 0, 255), 2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 10000: #Bo may cai contour be hon 20000 pixel
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 100) and (h>=150):
            check_bottle_flag = True
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx,cy = centroid_detect(x,y,w,h) 
            rects.append((x,y,x+w,y+h))

            # loop over the tracked objects 
            if initROI == False and cx > 250:
                ROI = []
                ROI.append(cv2.selectROI("Select the zone for checking", frame, False))
                print(ROI)
                qlog = open( 'ROI.csv', 'w' )
                qlog.write('%d\t%d\t%d\t%d\n' %(ROI[0][0], ROI[0][1], ROI[0][2], ROI[0][3]))
                ROI_product = frame[int(ROI[0][1]):int(ROI[0][1]+ROI[0][3]), int(ROI[0][0]):int(ROI[0][0]+ROI[0][2])]
                ROI.append(cv2.selectROI("Select the label", ROI_product, False))
                qlog.write('%d\t%d\t%d\t%d\n' %(ROI[1][0], ROI[1][1], ROI[1][2], ROI[1][3]))
                qlog.close()
                initROI = True
                #cv2.destroyWindow("Select the zone for checking")

    objects = ct.update(rects)
    if (check_bottle_flag == True):       
        
            # draw both the ID of the object and the centroid of the
            # object on the output frame
        check_botle_flag = False
        t = threading.Thread(target=check_bottle, args=(frame, ))
        t.start()

    cv2.putText(frame_copy, "Status: {}".format(text_1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame_copy, "No errors: {}".format(count), (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.imshow("Security Feed", frame)
    if initROI == True:
        cv2.rectangle(frame_copy, ((int)(ROI[0][0]*propotion),(int)(ROI[0][1]*propotion)), ((int)((ROI[0][0] + ROI[0][2])*propotion), (int)((ROI[0][1] + ROI[0][3])*propotion)), (0,155,0), 3)
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