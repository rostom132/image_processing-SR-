# USAGE
# python detect_edges_image.py --edge-detector hed_model --image images/guitar.jpg

# import the necessary packages
import argparse
import cv2
import os
import numpy as np

class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

font = cv2.FONT_HERSHEY_COMPLEX
kernel = np.ones((2,2),np.uint8)
# construct the argument parser and parse the arguments
protoPath = os.path.sep.join(["hed_model",
	"deploy.prototxt"])
modelPath = os.path.sep.join(["hed_model",
	"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)



class edgeDetection:
	def __init__(self):
		print("qq")
	def checkShape(self, image):
		global net, kernel
		(H, W) = image.shape[:2]

		# construct a blob out of the input image for the Holistically-Nested
		# Edge Detector
		blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
			mean=(122.82124693, 148.45341493, 121.97431813),
			swapRB=False, crop=False)
		# print(blob)
		# set the blob as the input to the network and perform a forward pass
		# to compute the edges
		net.setInput(blob)
		hed = net.forward()
		hed = cv2.resize(hed[0, 0], (W, H))
		hed = (255 * hed).astype("uint8")
		# temp = cv2.resize(blob[0, 0], (W, H))
		blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1)

		# cv2.imwrite('houghlines5.jpg',hed_binary)
		
		hed_temp_1 = hed.copy()
		# hed_temp = cv2.erode(hed_temp,kernel,iterations = 5)
		hed_temp = hed_temp_1
		hed_temp[ hed_temp_1 < 40 ] = 0
		cnts = cv2.findContours(hed_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		print(len(cnts))
		for c in cnts:
			area = cv2.contourArea(c)
			if (area > 400):
				cv2.drawContours(hed_temp, [c], -1, (255,255,255), -1)
				cnts_ero = cv2.findContours(hed_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				approx = cv2.approxPolyDP(c,  0.026*cv2.arcLength(c, True), True)
				x = approx.ravel()[0]
				y = approx.ravel()[1]
				return len(approx)