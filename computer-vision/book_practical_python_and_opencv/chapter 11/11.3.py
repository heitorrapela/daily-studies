from __future__ import print_function
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11,11), 0)
cv2.imshow("Image", image)

edged = cv2.Canny(blurred,50,250)
cv2.imshow("Edges", edged)
cv2.waitKey(0)

(_,cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for (i,c) in enumerate(cnts):
	(x,y,w,h) = cv2.boundingRect(c)

	print("Cont #{}".format(i+1))
	cont = image[y:y+h,x:x+w]
	cv2.imshow("Cont", cont)
	mask = np.zeros(image.shape[:2], dtype="uint8")
	((centerX,centerY), radius) = cv2.minEnclosingCircle(c)
	cv2.circle(mask, (int(centerX),int(centerY)),int(radius),255,-1)
	mask = mask[y:y+h,x:x+w]

	cv2.imshow("Masked cont", cv2.bitwise_and(cont,cont,mask=mask))
	cv2.waitKey(0)
