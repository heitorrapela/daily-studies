import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

blurred = np.hstack([
	cv2.GaussianBlur(image,(3,3),0), 
	cv2.GaussianBlur(image,(5,5),0), 
	cv2.GaussianBlur(image,(7,7),0)])
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)