import cv2
import time
import numpy as np

from .image_tools import crop_image

def contour_detection(base_image, selected_area, verbose=0):
	cropped_image, _, _ = crop_image(base_image, selected_area, factor=0)


	start = time.time()
	# hittar edges
	gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
	_, threshold = cv2.threshold(gray_image, np.mean(gray_image), 255, cv2.THRESH_BINARY_INV) # inte full kolla på hur denhär funkar
	edges = cv2.dilate(cv2.Canny(threshold, 0, 255), None) # eller denhär

	# skapar segment/områden av edges
	contours = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1] # verkl int koll
	outputMask = np.zeros(cropped_image.shape[:2], np.uint8)
	masked = cv2.drawContours(outputMask, [contours], -1, 255, -1)

	detected_edges = cv2.bitwise_and(cropped_image, cropped_image, mask=outputMask)
	output = cv2.cvtColor(detected_edges, cv2.COLOR_BGR2RGB)
	end = time.time()
	print("[INFO] applying Contour Detection took {:.2f} seconds".format(end - start))

	if verbose > 1:
		cv2.imshow("Cropped image", cropped_image)
		cv2.imshow("Contour_Detection detected_edges", detected_edges)
		cv2.imshow("Contour_Detection mask", outputMask)
		cv2.imshow("Contour_Detection output", output)
		cv2.waitKey(1)

	outputIm = [cropped_image, outputMask, output, detected_edges]
	return outputMask, output, selected_area, outputIm