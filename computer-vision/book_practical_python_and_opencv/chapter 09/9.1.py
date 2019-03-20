import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
b,g,r = cv2.split(image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5,5), 0)
cv2.imshow("Image", image)

(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)

(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Threshold Binary", thresh)
cv2.imshow("Threshold Inv Binary", threshInv)
cv2.waitKey(0)

b = cv2.bitwise_and(b, b, mask=thresh)
g = cv2.bitwise_and(g, g, mask=thresh)
r = cv2.bitwise_and(r, r, mask=thresh)

cv2.imshow("Image Yep", cv2.merge((b,g,r)))
cv2.waitKey(0)