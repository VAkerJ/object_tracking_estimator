import cv2
import time
import numpy as np

from .image_tools import crop_image

def grab_cut(base_image, selected_area, iterC, verbose=0):
	cropped_image, cropped_area, selected_area = crop_image(base_image, selected_area, factor=2)

	# Initiate variables for foreground and background
	mask = np.zeros(cropped_image.shape[:2], dtype="uint8")
	fgModel = np.zeros((1, 65), dtype="float")
	bgModel = np.zeros((1, 65), dtype="float")
	
	# Apply grabCut
	start = time.time()
	(mask, bgModel, fgModel) = cv2.grabCut(cropped_image, mask, cropped_area, \
	bgModel, fgModel, iterCount=iterC, mode=cv2.GC_INIT_WITH_RECT)
	end = time.time()
	print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

	outputMask = get_mask(mask)
	output = cv2.bitwise_and(cropped_image, cropped_image, mask=outputMask)
	if verbose > 1:
		cv2.imshow("Cropped image", cropped_image)
		cv2.imshow("GrabCut mask", outputMask)
		cv2.imshow("GrabCut output", output)
		cv2.waitKey(1)

	outputIm = [cropped_image, outputMask, output]
	return outputMask, output, selected_area, outputIm

def get_mask(mask):
	# Convert probabilities to background/foreground and rescale
	outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0,1)
	outputMask = (outputMask * 255).astype("uint8")
	return outputMask
