from __future__ import print_function
import numpy as np 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

M = np.ones(image.shape, dtype = "uint8")*100
added = cv2.add(image,M)
cv2.imshow("Added", added)

M = np.ones(image.shape, dtype = "uint8")*50
subtracted = cv2.subtract(image,M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)