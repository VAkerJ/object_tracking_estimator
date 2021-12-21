import cv2
import argparse
import os
import numpy as np
from copy import copy

from utils import Rectangle, Grab_Cut


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

				outputMask, output = Grab_Cut(base_image, mask, selected_area, bgModel, fgModel, args["iter"])
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