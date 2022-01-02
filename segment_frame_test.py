import cv2
import argparse
import os
import numpy as np
import filterpy.kalman as kf 
import filterpy.common as co
from copy import copy
from math import pi

from utils import Rectangle, Grab_Cut, Kmeans, Contour_Detection

def get_measurement(mask, rectangle, prev_measurements):

	mask_info = get_mask_info(mask, rectangle)

	# räkna ut föränding i mätningar
	# prev bör alltså inte innehålla förra hastigheterna
	if prev_measurements is not None:
		delta_measurements = [cur - prev for cur, prev in zip(mask_info, prev_measurements)]
		delta_measurements = np.asarray(delta_measurements, dtype=np.float32)
	else:
		delta_measurements = None

	mask_info = np.asarray(mask_info, dtype=np.float32)
	

	return mask_info, delta_measurements # borde kanske slå ihop?
	

def get_indices(mask):
	index_list = []

	X = np.shape(outputMask)[1]
	Y = np.shape(outputMask)[0]
	index_tot = [0,0]
	for x in range(X):
		for y in range(Y):
			if outputMask[y, x] > 0:
				index_list.append((x, y))

				index_tot[0] += x
				index_tot[1] += y

	index_amount = len(index_list)

	index_x, index_y = index_tot[0]//index_amount, index_tot[1]//index_amount
	i_min, i_max = min(index_list), max(index_list)

	return index_list, index_x, index_y, i_min, i_max, index_amount

def get_mask_info(mask, rectangle):

	index_list, index_x, index_y, i_min, i_max, index_amount = get_indices(mask)

	mask_height = i_max[1] - i_min[1]
	mask_width = i_max[0] - i_min[0]
	mask_density = index_amount *4/(pi*mask_height*mask_width) # formel för ellipse, kanske bör ändras för att hantera outliers i pixlar?

	index_x, index_y = index_x+rectangle[0], index_y+rectangle[1] # nu jobbar man inte med croppade koordinater längre
	
	return index_x, index_y, index_amount, mask_height, mask_width, mask_density

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
	seg_meth, segArgs = segmentaion_methods[args["segment"]]
	Segment = lambda image, area: seg_meth(image, area, *segArgs)

	# initiera fönstret och rektangeln som används för val av area
	rect = Rectangle()
	windowName = "TEST"
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, rect.mouse_test)

	# ta första bilden ur videon
	vidcap = cv2.VideoCapture(args["input"])
	success,base_image = vidcap.read()
	image = base_image

	# Initiera Kalmanfilter
	k_fil = kf.KalmanFilter(dim_x = 2, dim_z = 2, dim_u = 0) # Vill vi ha med hastighet som state också? isf ska dim_x = 4
	# x0, y0 = getMeanPosition(mask) 
	#k_fil.x = np.array([[x0],[y0],[vx0],[vy0]])	# Startposition och hastighet (x0,y0,vx0,vy0)
	k_fil.F = np.array([[1.,1.],[0.,1.]])   # state transition matrix
	prev_measurements = None

	dt = 0.1

	k_fil.H = np.array([[1.,0.]])           # Measurement function
	k_fil.P *= 1000.                        # covariance matrix
	k_fil.R = 5                             # state uncertainty
	k_fil.Q = co.Q_discrete_white_noise(2, dt, .1) # process uncertainty

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
				
				outputMask, _, new_selected_area, outputIm = Segment(base_image, selected_area)

				#outputIm = np.concatenate(outputIm, axis=1)
				if args["verbose"] > 0:
					output_images = []
					for im in outputIm:
						if len(np.shape(im)) < 3:
							im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
						output_images.append(im)

					output_images = np.hstack(output_images)
					cv2.imshow("output images", output_images)
					cv2.waitKey(1)

				if args["segment"] != "Kmeans":
					measurements, delta_measurements = get_measurement(outputMask, new_selected_area, prev_measurements) # döp till prev measurements på direkten?
					print("Measurements\n", measurements, "\nx coor,    y coor,    nr of indices,    height    ,width,    density")
				else:
					print("Method not fully implemented for 'Kmeans'")
				

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

			k_fil.predict()
			#z = getMeasurement(image, mask)
			#k_fil.update(z)
			X = k_fil.x # New state estimate
			P = k_fil.P # Covariance matrix

			#plotEstimate(image,X,P)

			count += 1

	cv2.destroyAllWindows()
