import numpy as np  
import cv2 
from matplotlib import pyplot as plt 
import os.path

font = cv2.FONT_HERSHEY_SIMPLEX 

# Defind ROI 
ROI = [0 for i in range(4)]
ROI_LABEL = [0 for i in range(4)]
initROI = False
# label = cv2.imread('./label.jpg', 0) # trainImage
# sample = cv2.imread('./input/lix.jpg', 0) # queryImage

def compareCoor(roi, x, y):
    logic = roi[0] < x < roi[0] + roi[2] and roi[1] < y < roi[1] + roi[3]
    return logic

def get_template(img1, img2):
    MIN_MATCH_COUNT = 20

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print("Keypoints in 1st Image: " + str(len(kp1)))
    print("Keypoints in 2nd Image: " + str(len(kp2)))

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

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
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

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
                res = res + 1 if (compareCoor(ROI, x[i], y[i]) == True) else res + 0   
            if(res == 4):
                print("Label is correct")
                cv2.putText(img2, 'Label is pasted correct', (0, 50), font, 1, (0,255,0), 2)
                cv2.putText(img2, 'Input sample: {:.0f} %'.format(similariy_percentage), (0, 100), font, 1, (0,255,0), 2)
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
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    #plt.imshow(img2, 'gray'),plt.show()

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()

# Read input image
#label = cv2.imread('./sunlight_label.png') # labelImage
sample = cv2.imread('./fixed_mask.jpg') # sampleImage

sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

# First step: Get ROI of the label
if not os.path.isfile("ROI_LABEL.jpg"):
    ROI_LABEL= cv2.selectROI("Select the label",sample, False)
    #cv2.destroyWindow("Select the label")
    roi = sample[ROI_LABEL[1]:ROI_LABEL[1] + ROI_LABEL[3], ROI_LABEL[0]:ROI_LABEL[0] + ROI_LABEL[2]]
    cv2.imwrite("ROI_LABEL.jpg", roi)

label = cv2.imread('./ROI_LABEL.jpg')
label_gray= cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

# # Second step: Get ROI of zone for checking the position of the label
if initROI == False:
    ROI = cv2.selectROI("Select the zone for checking", sample, False)
    initROI = True
    #cv2.destroyWindow("Select the zone for checking")
cv2.rectangle(sample_gray, (ROI[0], ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (0,255,0), 3)

# Third step: Execute the function
get_template(label_gray, sample_gray)

key = cv2.waitKey(0) & 0xFF
if key == ord("q"):
    cv2.destroyAllWindows()
