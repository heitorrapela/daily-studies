import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

translate_x = 10
translate_y = 10

shifted = imutils.translate(image,x=translate_x,y=translate_y)

cv2.imshow("Shifted Up and Left", shifted)
cv2.waitKey(0)