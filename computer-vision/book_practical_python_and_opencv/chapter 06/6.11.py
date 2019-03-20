from __future__ import print_function
import numpy as np 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

print("max of 255: {}".format(cv2.add(np.uint8([200]),np.uint8([100]))))
print("min of 0: {}".format(cv2.subtract(np.uint8([50]), np.uint8([100]))))
print("wrap around: {}".format(np.uint8([200] + np.uint8([100]))))
print("wrap around: {}".format(np.uint8([50]) - np.uint8([100])))


