import cv2
import argparse
import os
import numpy as np
from copy import copy

from utils import Rectangle, Grab_Cut, Kmeans, Contour_Detection, CV2_Tracker, Kalman_Tracker, Kalman_Filter

def main(filepath, Segment, wait, verbose):
	# initiera input fönstret och rektangeln som används för val av area
	rect = Rectangle()
	windowName = "Input window"
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, rect.mouse_test)

	# ta första bilden ur videon
	video = cv2.VideoCapture(filepath)
	success,base_image = video.read()
	image = copy(base_image)

	while True:
		cv2.imshow(windowName, image) # visa bilden med eller utan rektangel
		
		#TODO med tillhörande värden
		input_check = check_input(Segment, base_image, rect, verbose=verbose)
		if input_check is not None:
			if type(input_check) is tuple:
				image, selected_area, measurements, delta_measurements = input_check
			else:
				image = input_check				

		# stoppa eller starta spårning beroende på space eller esc
		k = cv2.waitKey(1)
		if k%256 == 27:
			success = False
			print("Excape hit, closing...")
			break

		if k%256 == 32:
			try:
				selected_area
			except NameError:
				print('No rectangle drawn, try again')
				continue
			else:
				break

	# Initiera Kalmanfilter och tracker
	kalman_tracker = Kalman_Tracker(Segment, Kalman_Filter(measurements, delta_measurements), selected_area, verbose)

	# initiera input fönstret och cv2 trackern att jämföra med
	cv2_tracker = CV2_Tracker(base_image, selected_area)
	outputWindow = "Output window"
	cv2.namedWindow(outputWindow)	

	c = int(not wait) # vänta på tangenttryck mellan varje bild ifall wait=true
	while success:
		

		success, image = kalman_tracker.update(copy(base_image))
		if not success: break

		_, cv2_tracker_im = cv2_tracker.update(copy(base_image))

		output_images = np.vstack([image, cv2_tracker_im])
		cv2.imshow(outputWindow, output_images)

		# stoppa eller gå till nästa frame beroende på space eller esc
		k = cv2.waitKey(c)
		if k%256 == 27:
			success = False
			print("Excape hit, closing...")
			break

		if k%256 == 32 or not wait:
			success,base_image = video.read()
			image = copy(base_image)
			print('Read a new frame: ', success)


	cv2.waitKey(0)
	on_exit(video)




def check_input(Segment, base_image, rect, prev_measurements=None, verbose=1):
	# kolla om rektangeln börjat bli vald och rita isf ut den i bilden
	if rect.is_active():
		p0, p1 = rect.get_points()
		image = cv2.rectangle(copy(base_image), p0, p1, (0,0,255), 1)
		if rect.is_finished(): # kolla om rektangeln är klar (m1 släppt)
			selected_area = rect.get_rec()
			rect.clear()
			
			#applicera segmenteringsmetoden
			outputMask, _, new_selected_area, outputIm = Segment(base_image, selected_area)
			
			# bara liten varning ifall man använde kmeans
			if args["segment"] == "Kmeans":
				print("Method not fully implemented for 'Kmeans'")
				return image

			# skapar preliminära mätvärden att testa illustrering med
			success, measurements, delta_measurements = Kalman_Tracker.get_measurements(outputMask, new_selected_area, prev_measurements) # döp till prev measurements på direkten?
			if not success:
				print('No foreground found, try again')
				return image

			# illustrerar mätvärdena
			if verbose > 0:
				measurement_im = Kalman_Tracker.illustrate_measurements(copy(outputIm[0]), measurements, new_selected_area[0:2])
				outputIm.append(measurement_im)
				Kalman_Tracker.show_segmentation(outputIm)

			return image, selected_area, measurements, delta_measurements

		# ifall rektangeln inte är klar
		return image

	# ifall rektangeln inte excisterar
	return None

def on_exit(video):
	video.release()
	cv2.destroyAllWindows()

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
	ap.add_argument("-w", "--wait", type=bool, default=False,
		help="# if the method should wait between predictions")
	args = vars(ap.parse_args())

	segmentaion_methods = {"Grab_Cut" : [Grab_Cut, [args["iter"], args["verbose"]]],
							"Kmeans" : [Kmeans, [args["iter"], args["clusters"], args["verbose"]]],
							"Contour_Detection" : [Contour_Detection, []]}
	seg_meth, segArgs = segmentaion_methods[args["segment"]]
	segmentation_method = lambda image, area: seg_meth(image, area, *segArgs)

	main(args["input"], segmentation_method, args["wait"], args["verbose"])