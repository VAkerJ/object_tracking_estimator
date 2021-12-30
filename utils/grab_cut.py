import cv2
import time
import numpy as np

from .image_tools import crop_image

def grab_cut(base_image, selected_area, iterC, verbose=1):
	image, cropped_area, selected_area = crop_image(base_image, selected_area, factor=2)

	# initiera graphcut variabler
	mask = np.zeros(image.shape[:2], dtype="uint8")
	fgModel = np.zeros((1, 65), dtype="float")
	bgModel = np.zeros((1, 65), dtype="float")
	
	# applicera grab cut
	start = time.time()
	(mask, bgModel, fgModel) = cv2.grabCut(image, mask, cropped_area, \
	bgModel, fgModel, iterCount=iterC, mode=cv2.GC_INIT_WITH_RECT)
	end = time.time()
	print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

	outputMask = get_mask(mask)
	output = cv2.bitwise_and(image, image, mask=outputMask)
	if verbose > 0:
		cv2.imshow("Cropped image", image)
		cv2.imshow("GrabCut Mask", outputMask)
		cv2.imshow("GrabCut output", output)
		cv2.waitKey(1)
	return outputMask, output, selected_area

def get_mask(mask):

	## visualisera graph cut resultaten
	#values = (
	#	("Definite Background", cv2.GC_BGD),
	#	("Probable Background", cv2.GC_PR_BGD),
	#	("Definite Foreground", cv2.GC_FGD),
	#	("Probable Foreground", cv2.GC_PR_FGD),
	#)
	#for (name, value) in values:
	#	print("[INFO] showing mask for '{}'".format(name))
	#	valueMask = (mask == value).astype("uint8") * 255
	#	cv2.imshow(name, valueMask)
	#	cv2.waitKey(1)

	# maska så att prob bg och def bg blir 0
	outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0,1)
	
	# skala fr 0:1 till 0:255
	outputMask = (outputMask * 255).astype("uint8")

	return outputMask
