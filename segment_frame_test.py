import cv2
import argparse
import os
import numpy as np
import filterpy.kalman as kf 
import filterpy.common as co
from copy import copy
from math import pi

from utils import Rectangle, Grab_Cut, Kmeans, Contour_Detection


def main():
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
	image = copy(base_image)

	# Initiera Kalmanfilter
	dt = 0.1
	k_fil = kf.KalmanFilter(dim_x = 4, dim_z = 2, dim_u = 2) 
	k_fil.F = np.array([[1.,0.,dt,0.], 		# state transition matrix
						[0.,1.,0.,dt],
						[0.,0.,0.,0.],
						[0.,0.,0.,0.]])   
	k_fil.H = np.array([[1.,0.,0.,0.],
						[0.,1.,0.,0.]])    	# Measurement function
	k_fil.P *= 1000.                        # Covariance matrix
	k_fil.R = 5                             # State uncertainty
	k_fil.Q = co.Q_discrete_white_noise(4, dt, .1) # Process uncertainty
	prev_measurements = None
	#print(k_fil)

	count = 0
	while success:
		cv2.imshow(windowName, image) # visa bilden med eller utan rektangel
		
		# kolla om rektangeln börjat bli vald och rita isf ut den
		if rect.is_active():
			p0, p1 = rect.get_points()
			image = cv2.rectangle(copy(base_image), p0, p1, (0,0,255), 1)
			if rect.is_finished(): # kolla om rektangeln är klar (m1 släppt)
				selected_area = rect.get_rec()
				rect.clear()
				
				outputMask, _, new_selected_area, outputIm = Segment(base_image, selected_area)

				# TODO: gör till egen funktion
				if args["verbose"] > 0:
					output_images = []
					for im in outputIm:
						if len(np.shape(im)) < 3:
							im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
						output_images.append(im)

					output_images = np.hstack(output_images)
					cv2.imshow("output images", output_images)
					cv2.waitKey(1)

				if args["segment"] == "Kmeans":
					print("Method not fully implemented for 'Kmeans'")
					continue

				try:
					measurements, delta_measurements = get_measurement(outputMask, new_selected_area, prev_measurements) # döp till prev measurements på direkten?
				except ZeroDivisionError:
					print('No foreground found, try again')
					continue
				#print("Measurements\n", measurements, "\nx coor,    y coor,    nr of indices,    height    ,width,    density")
				#print(delta_measurements)
				x0, y0 = measurements[0], measurements[1]

				if delta_measurements is not None: 
					v_x, v_y = delta_measurements[0], delta_measurements[1]
				else: v_x, v_y = 0, 0

				k_fil.x = np.array([[x0],[y0],[v_x],[v_y]])
				#print(k_fil.x)

				
		

		# stoppa eller gå till nästa frame beroende på space eller esc
		k = cv2.waitKey(1)
		if k%256 == 27:
			print("Excape hit, closing...")
			break

		if k%256 == 32:
			success,base_image = vidcap.read()
			image = copy(base_image)
			print('Read a new frame: ', success)
			
			try:
				measurements, delta_measurements = get_measurement(outputMask, new_selected_area, prev_measurements) # döp till prev measurements på direkten?
			except:
				print('No rectangle drawn')
				continue
			z = measurements[0:2]

			k_fil.predict()
			k_fil.update(z)
			X = k_fil.x # New state estimate
			P = k_fil.P # Covariance matrix
			X = X.astype(int)
			# print(X, type(X))
			# print(type(X[0]))
			center_est = (int(X[0]),int(X[1]))
			# print(center_est)
			image = cv2.circle(image,center_est,20,(255,0,0),2)

			count += 1

	cv2.destroyAllWindows()


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
	X = np.shape(mask)[1]
	Y = np.shape(mask)[0]
	index_tot = [0,0]
	for x in range(X):
		for y in range(Y):
			mask[y,x] > 0
			if (mask[y,x] > 0).any(): # Oklart varför .any() behövs när mask[y,x] är 1x1
				index_list.append((x,y))

				index_tot[0] += x
				index_tot[1] += y

	index_amount = len(index_list)
	#if index_amount == 0: raise Exception('No foreground found :(')
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
	main()

