import cv2
import numpy as np

canvas = np.zeros((300,300,3), dtype = 'uint8')
(centerX, centerY) = (canvas.shape[1]//2, canvas.shape[0]//2)
white = (255,255,255)

for r in range(0,200,25):
	#print(r)
	cv2.circle(canvas, (centerX, centerY),r,white)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)