import numpy as np
import cv2

def translate(image, x, y):
	M = np.float32([[1,0,x],[0,1,y]])
	shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
	return shifted

def rotate(image, angle, center = None, scale=1.0):
	(h,w) = image.shape[:2]

	if(center is None):
		center = (w/2,h/2)

	M = cv2.getRotationMatrix2D(center,angle, scale)
	rotated = cv2.warpAffine(image,M,(w,h))

	return rotated

def resize(image, width = None, height = None , inter =cv2.INTER_AREA):
	dim = None
	(h,w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = height/float(h)
		dim = (int(w*r),height)

	if height is None:
		r = width/float(w)
		dim = (width,int(h*r))

	if width is not None and height is not None:
		dim = (width,height)

	resized = cv2.resize(image, dim, interpolation = inter)
	return resized

def flip(image, axis=0):
	# axis = 0 horizontally
	# axis = 1 vertically
	# axis = -1 both
	flipped = cv2.flip(image,axis)
	return flipped

def crop(image, startx,endx,starty,endy):
	aux = image.copy()
	return aux[startx:endx,starty:endy]