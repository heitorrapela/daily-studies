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

fig = plt.figure()

ax = fig.add_subplot(131)

hist = cv2.calcHist([chans[1],chans[0]], [0,1], None, [32,32], [0,256,0,256])

p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)


ax = fig.add_subplot(132)

hist = cv2.calcHist([chans[1],chans[2]], [0,1], None, [32,32], [0,256,0,256])

p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and z")
plt.colorbar(p)

ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0],chans[2]],[0,1], None, [32,32], [0,256,0,256])

p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)

print("2D histogram shape: {}, with {} values".format(hist.shape,hist.flatten().shape[0]))

plt.show()