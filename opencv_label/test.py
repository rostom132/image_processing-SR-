import cv2 as cv    
import numpy as np
import configparser
import numpy as np
import argparse
import imutils
import glob

kernel = np.ones((5,5),np.uint8)

ROI_EXT = 0 


config = configparser.ConfigParser()
config.read("setting.ini")
IMAGES_MASK = config.get("path", "images_mask")
SOURCE_IMAGES_PATH = config.get("path", "source_dir")
RESULT_IMAGES_PATH = config.get("path", "result_dir")
CUT_BORDER = config.getfloat("geometry", "cut_border")
PRE_ROI_WIDTH = config.getfloat("geometry", "pre_roi_width")
ROI_WIDTH = config.getfloat("geometry", "roi_width")
ROI_EXT = config.getfloat("geometry", "roi_ext")

def show(pic, name):
   cv.namedWindow(name,cv.WINDOW_NORMAL) 
   cv.resizeWindow(name, 600,600)
   cv.imshow(name,pic)


def get_roi(img, left_border, right_border):
   img_h, img_w, _ = img.shape
   left_part = img[:, 0:left_border]
   right_part = img[:, right_border:img_w]
   return np.column_stack((left_part, right_part))

def find_background(img, roi):
   img_h, img_w, _ = img.shape
   hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
   hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
   roi_hist = cv.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
   mask = cv.calcBackProject([hsv_img], [0, 1], roi_hist, [0, 180, 0, 256], 1)

   ksize = int(0.0025 * img_h)
   kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
   mask = cv.filter2D(mask, -1, kernel)
   _, mask = cv.threshold(mask, 180, 255, cv.THRESH_BINARY)
   return mask

def get_pre_borders(mask):
   components = cv.connectedComponentsWithStats(mask, connectivity=8, ltype=cv.CV_32S)
   _, labelmap, stats, centers = components
   st = stats[:, 2]
   largest = np.argmax(st)
   st[largest] = 0
   second = np.argmax(st)
   left = stats[second, 0]
   right = left + stats[second, 2]
   top = stats[second, 1]
   bot = top + stats[second, 3]
   return left, right, top, bot

precessing_pic = 'input/bottle9.jpg'

original_img = cv.imread(precessing_pic)
original_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
orig_img_h, orig_img_w, _ = original_img.shape


border = int(orig_img_w * CUT_BORDER)
image = original_img[:, border:orig_img_w - border]
image_h, image_w, _ = image.shape

pre_roi = get_roi(image, int(image_w * PRE_ROI_WIDTH), int(image_w - image_w * PRE_ROI_WIDTH))
show(pre_roi, "roi")
pre_mask = cv.bitwise_not(find_background(image, pre_roi))

temp_mat = np.zeros(pre_mask.shape, dtype=np.uint8)

cnts = cv.findContours(pre_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv.contourArea, reverse=True)
for c in cnts:
    cv.drawContours(temp_mat, [c], -1, (255,255,255), -1)
    break

show(temp_mat, 'mask')

left,right, top, bot = get_pre_borders(temp_mat)
cut_img = image[top:bot , left:right]
# cut_img = image[:, (left_border - roi_width):(right_border + roi_width)]
# _, cut_img_w, _ = cut_img.shape
show(cut_img, 'mask_1')


cv.waitKey(0) # waits until a key is pressed
cv.destroyAllWindows() # destroys the window showing image