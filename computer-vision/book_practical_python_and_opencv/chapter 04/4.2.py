from __future__ import print_function
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

posx = 0
posy = 0
(b,g,r) = image[posy,posx]
print("Pixel at ({},{}) - Red: {}, Green: {}, Blue: {}".format(posy,posx,r,g,b))