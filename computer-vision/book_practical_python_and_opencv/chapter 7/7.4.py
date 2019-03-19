from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

chans = cv2.split(image)
colors = ("b", "g", "r")

hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])

print("3D histgoram shape: {}, with {} values".format(hist.shape,hist.flatten().shape[0]))

plt.show()