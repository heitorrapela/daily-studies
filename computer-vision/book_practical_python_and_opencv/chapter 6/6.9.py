import numpy as np 
import cv2
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

print(image.shape)

cropped = image[20:150, 60:170]
cv2.imshow("Rapela's Face", cropped)
cv2.waitKey(0)