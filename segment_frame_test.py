import cv2
import argparse
import os
import numpy as np
from copy import copy

from utils import Rectangle, Grab_Cut, Kmeans, Contour_Detection


if __name__=="__main__":
	

	# arg parser, nyttja text 'python3 segment_frame_test.py -s Kmeans' för att köra med Kmeans metoden
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str,
		default=os.path.sep.join(["test_data", "frodo_1.mp4"]),
		help="path to input video")
	ap.add_argument("-c", "--iter", type=int, default=10,
		help="# of GrabCut or Kmeans iterations")
	ap.add_argument("-s", "--segment", type=str, default="Grab_Cut",
		help="type of segmentation method")
	ap.add_argument("-v", "--verbose", type=int, default=1,
		help="determine verbosity")
	ap.add_argument("-k", "--clusters", type=int, default=2,
		help="# of Kmeans clusters")
	args = vars(ap.parse_args())

	segmentaion_methods = {"Grab_Cut" : [Grab_Cut, [args["iter"], args["verbose"]]],
							"Kmeans" : [Kmeans, [args["iter"], args["clusters"], args["verbose"]]],
							"Contour_Detection" : [Contour_Detection, []]}
	segment, segArgs = segmentaion_methods[args["segment"]]
	Segment = lambda image, area: segment(image, area, *segArgs)

	# initiera fönstret och rektangeln som används för val av area
	rect = Rectangle()
	windowName = "TEST"
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, rect.mouse_test)

	# ta första bilden ur videon
	vidcap = cv2.VideoCapture(args["input"])
	success,base_image = vidcap.read()
	image = base_image

	count = 0
	while success:
		# kolla om rektangeln börjat bli vald och rita isf ut den
		if rect.is_active():
			p0, p1 = rect.get_points()
			image = cv2.rectangle(copy(base_image), p0, p1, (0,0,255), 1)
			if rect.is_finished(): # kolla om rektangeln är klar (m1 släppt)
				#selected_area = tuple(list(p0) + list(p1))
				selected_area = rect.get_rec()
				rect.clear()

				#outputMask, output = Grab_Cut(base_image, selected_area, args["iter"])
				
				Segment(base_image, selected_area)
				

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
