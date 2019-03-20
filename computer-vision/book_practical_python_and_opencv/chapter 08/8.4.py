import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

blurred = np.hstack([
	cv2.bilateralFilter(image,5,21,21), 
	cv2.bilateralFilter(image,7,31,31), 
	cv2.bilateralFilter(image,9,41,41)])
cv2.imshow("Bilateral", blurred)
cv2.waitKey(0)