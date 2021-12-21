import cv2
import argparse
import os
import numpy as np
from copy import copy

import time

class Rectangle():
	def __init__(self):
		self.p0, self.p1, self.ptemp = None, None, None

	def get_points(self):
		p0, p1 = self.p0, self.p1
		if p1 is None: p1 = self.ptemp
		return p0, p1

	def is_active(self):
		return self.p0 is not None

	def is_finished(self):
		return self.p1 is not None

	def clear(self):
		self.p0, self.p1, self.ptemp = None, None, None

	def mouse_test(self, event, x, y, flags, param):
		if event == 1: #mouse1 click
			self.p0 = (x,y)

		elif event == 4: #mouse1 release
			self.p1 = (x,y)

		if self.p0 is not None: self.ptemp = (x,y)

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



if __name__=="__main__":

	# arg parser, spelar ingen roll just nu
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str,
		default=os.path.sep.join(["test_data", "frodo_1.mp4"]),
		help="path to input video")
	ap.add_argument("-c", "--iter", type=int, default=10,
		help="# of GrabCut iterations")
	args = vars(ap.parse_args())

	# initiera fönstret och rektangeln som används för val av area
	rect = Rectangle()
	windowName = "TEST"
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, rect.mouse_test)

	# ta första bilden ur videon
	vidcap = cv2.VideoCapture(args["input"])
	success,base_image = vidcap.read()
	image = base_image

	# initiera graphcut variabler
	mask = np.zeros(image.shape[:2], dtype="uint8")
	fgModel = np.zeros((1, 65), dtype="float")
	bgModel = np.zeros((1, 65), dtype="float")

	count = 0
	while success:
		# kolla om rektangeln börjat bli vald och rita isf ut den
		if rect.is_active():
			p0, p1 = rect.get_points()
			image = cv2.rectangle(copy(base_image), p0, p1, (0,0,255), 1)
			if rect.is_finished(): # kolla om rektangeln är klar (m1 släppt)
				selected_area = tuple(list(p0) + list(p1))
				rect.clear()

				# applicera grab cut
				start = time.time()
				(mask, bgModel, fgModel) = cv2.grabCut(base_image, mask, selected_area,
				bgModel, fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_RECT)
				end = time.time()
				print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

				outputMask = get_mask(mask)
				output = cv2.bitwise_and(base_image, base_image, mask=outputMask)
				cv2.imshow("GrabCut Mask", outputMask)
				cv2.imshow("GrabCut output", output)
				cv2.waitKey(1)

		cv2.imshow(windowName, image) # visa bilden med eller utan rektangel

		# stoppa eller gå till nästa frame beroende på space eller esc
		k = cv2.waitKey(1)
		if k%256 == 27:
			print("Excape hit, closing...")
			break

		if k%256 == 32:
			success,base_image = vidcap.read()
			image = base_image
			print('Read a new frame: ', success)
			count += 1

	cv2.destroyAllWindows()