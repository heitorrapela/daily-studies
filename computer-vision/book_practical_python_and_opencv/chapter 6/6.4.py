import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

rotated = imutils.rotate(image,angle=45,center=(image.shape[0]/2,image.shape[1]/2),scale=2.0)
cv2.imshow("Image rotated 45 degrees and rezised 2x", rotated)
cv2.waitKey(0)