import cv2
import argparse
import os
import numpy as np
import filterpy.kalman as kf 
import filterpy.common as co
from copy import copy
from math import pi

from utils import Rectangle, Grab_Cut, Kmeans, Contour_Detection, Kalman_Tracker
Get_Measurements = Kalman_Tracker.get_measurements # för att det ska funka som tidigare


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
	P0 = 1000.
	k_fil.P *= P0                        # Covariance matrix
	k_fil.R = 5                             # State uncertainty
	k_fil.Q = co.Q_discrete_white_noise(4, dt, .1) # Process uncertainty
	prev_measurements = None
	#print(k_fil)

	count = 1
	tracking = False
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
					_, measurements, delta_measurements = Get_Measurements(outputMask, new_selected_area, prev_measurements) # döp till prev measurements på direkten?
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

				
		
		k = cv2.waitKey(1)

		if k%256 == 27:
			print("Excape hit, closing...")
			break
		
		if k%256 == 32 or tracking:
			success,base_image = vidcap.read()
			image = copy(base_image)
			print('Read frame ', count, ' : ', success)
			count += 1
			
		if not tracking:
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
						measurements, delta_measurements = Get_Measurements(outputMask, new_selected_area, prev_measurements) # döp till prev measurements på direkten?
					except ZeroDivisionError:
						print('No foreground found, try again')
						tracking = False
						continue
					#print("Measurements\n", measurements, "\nx coor,    y coor,    nr of indices,    height    ,width,    density")
					#print(delta_measurements)
					tracking = True
					x0, y0 = measurements[0], measurements[1]

					if delta_measurements is not None: 
						v_x, v_y = delta_measurements[0], delta_measurements[1]
					else: v_x, v_y = 0, 0

					k_fil.x = np.array([[x0],[y0],[v_x],[v_y]])
					#print(k_fil.x)

		if tracking:
			prev_measurements = measurements
			try:
				_, measurements, delta_measurements = Get_Measurements(outputMask, new_selected_area, prev_measurements) # döp till prev measurements på direkten?
			except:
				print('No foreground found in the rectangle')
				continue
			
			z = measurements[0:2]
			k_fil.predict()
			k_fil.update(z)
			P = [] # Covariance matrix
			X = [] # State matrix
			for i in range(len(k_fil.x)):
				X.append(int(k_fil.x[i]))
				P.append(float(k_fil.P.diagonal()[i]))
			center_est = X[0:2]
			image = cv2.circle(image,tuple(center_est),20,(255,0,0),1)

			# Update rectangle (foreground area of interest)
			(x_min, y_min, x_len, y_len) = selected_area
			# x_len = int(x_len*P[0])
			# y_len = int(y_len*1.1)
			x_min = int(center_est[0] - x_len/2)
			y_min = int(center_est[1] - y_len/2)
			selected_area = (x_min, y_min, x_len, y_len)
			print(selected_area)
			p0 = (x_min, y_min + y_len)
			p1 = (x_min + x_len, y_min)
			image = cv2.rectangle(image, p0, p1, (0,0,255), 1)
			
		
		cv2.imshow(windowName, image) # visa bilden med eller utan rektangel
		
	cv2.destroyAllWindows()





if __name__=="__main__":
	main()

