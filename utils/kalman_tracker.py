import numpy as np
from copy import copy
import cv2

class Tracker():
	def __init__(self, Segment, Filter, selected_area, verbose, setup = 2):
		self.filter = Filter
		self.prev_measurements = None
		self.segment = Segment
		self.verbose = verbose
		self.selected_area = selected_area
		self.setup = setup

		self.segment_window = "Segmentation"
		cv2.namedWindow(self.segment_window, cv2.WINDOW_NORMAL)

	def update(self, base_image):
		selected_area = self.selected_area
		
		# segmentera och hämta mätvärden
		outputMask, _, cropped_selected_area, outputIm = self.segment(base_image, selected_area)
		success, measurements, delta_measurements = Tracker.get_measurements(outputMask, cropped_selected_area, self.prev_measurements, self.setup)

		# skriver ut sista resultaten ifall error upstår, bör kanske skrivas om i framtiden så att den kan hantera att den tappar objektet
		if not success:
			Tracker.show_error_image(base_image, outputIm[:2], selected_area, cropped_selected_area)
			return success, base_image

		# skapar en liten ruta med de olika små output-bilderna
		if self.verbose > 0:
			measurement_im = Tracker.illustrate_measurements(copy(outputIm[0]), measurements, cropped_selected_area[0:2])
			outputIm.append(measurement_im)
			Tracker.show_segmentation(outputIm, self.segment_window)

		# ritar ut rektangeln i stora output-bilden tsm med punkt för center och estimerat center
		center_measured = (int(measurements[0]), int(measurements[1]))
		image = self.draw_estimate(copy(base_image), center_measured)

		# updatera filtret?
		self.update_filter(measurements, delta_measurements)
		# updatera var rektangeln är
		self.update_selected_area(measurements)

		self.prev_measurements = measurements

		return success, image

	def update_filter(self, measurements, delta_measurements):
		self.filter.update(measurements, delta_measurements)

	def update_selected_area(self, measurements): # metod för att flytta den valda rutan
		x, y, width, height = self.selected_area
		x = int(measurements[0] - width/2)
		y = int(measurements[1] - height/2)
		self.selected_area = (x, y, width, height)
		self.filter.set_selected_area(self.selected_area)
		self.selected_area = self.filter.get_new_area()

	#-----------------------------------------------------------------------------
	# get image info/measurements metoder

	@staticmethod
	def get_measurements(mask, rectangle, prev_measurements, setup = 2):
		success = True

		try:
			measurements = Tracker.get_mask_info(mask, rectangle)
		except ZeroDivisionError:
			# ändra dehär för att hantera när bilden försvinner
			print("[Alert] Empty mask using rectangle={}, aborting".format(rectangle))
			success = False
			return success, None, None

		if setup == 1: measurements = np.append(np.append(measurements[0:2],measurements[8:10]),measurements[6:8])
	
		# räkna ut föränding i mätningar
		# prev bör alltså inte innehålla förra hastigheterna
		if prev_measurements is not None:
			delta_measurements = [cur - prev for cur, prev in zip(measurements, prev_measurements)]
			delta_measurements = np.asarray(delta_measurements, dtype=np.float32)
		else:
			delta_measurements = np.zeros(np.shape(measurements))

		measurements = np.asarray(measurements, dtype=np.float32)

		return success, measurements, delta_measurements

	@staticmethod
	def get_indices(mask):
		index_list = []
		X = np.shape(mask)[1]
		Y = np.shape(mask)[0]
		index_tot = [0,0]
		for x in range(X):
			for y in range(Y):
				if (mask[y,x] > 0).any(): # Oklart varför .any() behövs när mask[y,x] är 1x1
					index_list.append((x,y))

					index_tot[0] += x
					index_tot[1] += y

		index_amount = len(index_list)

		if index_amount == 0: raise ZeroDivisionError

		index_x, index_y = index_tot[0]//index_amount, index_tot[1]//index_amount

		i_min = (min(index_list, key=lambda x:x[0])[0], min(index_list, key=lambda x:x[1])[1])
		i_max = (max(index_list, key=lambda x:x[0])[0], max(index_list, key=lambda x:x[1])[1])

		return index_list, index_x, index_y, i_min, i_max, index_amount

	@staticmethod
	def get_mask_info(mask, rectangle):

		index_list, index_x, index_y, i_min, i_max, index_amount = Tracker.get_indices(mask)

		
		mask_height = i_max[0] - i_min[0]
		mask_width = i_max[1] - i_min[1]

		x_min, x_max = i_min[0]+rectangle[0], i_max[0]+rectangle[0]
		y_min, y_max = i_min[1]+rectangle[1], i_max[1]+rectangle[1]
		index_x, index_y = index_x+rectangle[0], index_y+rectangle[1] # nu jobbar man inte med croppade koordinater längre

		mask_density = index_amount/((x_max-x_min)*(y_max-y_min)) # formel för rektangel

		return index_x, index_y, x_min, y_min, x_max, y_max, index_amount, mask_density, mask_width, mask_height

	#-----------------------------------------------------------------------------
	# illustrativa metoder
	def draw_estimate(self, image, center_measured):
		# for drawing the box with estimated and measured center in the large outputwindow
		selected_area = self.filter.get_new_area()
		p0 = selected_area[0:2]
		p1 = (selected_area[0]+selected_area[2], selected_area[1]+selected_area[3])
		estimated_area = (p0, p1)

		X = self.filter.X
		if self.setup == 1:
			(x_min, y_min, x_len, y_len) = selected_area
			p0 = (int(x_min),int(y_min))
			p1 = (int(x_min + x_len),int(y_min + y_len))
			center_estimate = (int(X[0]),int(X[1]))
		else:
			p0 = (int(X[2]), int(X[3]))
			p1 = (int(X[4]), int(X[5]))
			center_estimate = self.filter.get_center_est()
			box_color = (0,0,255)
			dot_color = (255,0,0)
			Tracker.draw_box_with_dot(image, estimated_area, center_measured, box_color, dot_color)
		
		mask_area = (p0, p1)
		
		box_color = (0,255,0)
		dot_color = (0,255,0)
		Tracker.draw_box_with_dot(image, mask_area, center_estimate, box_color, dot_color)
		

		return image

	@staticmethod
	def illustrate_measurements(image, measurements, offset=[0,0]):
		# for drawing drawing the cropped images and masks in the small window
		p0 = (int(measurements[2] - offset[0]), int(measurements[3] - offset[1]))
		p1 = (int(measurements[4] - offset[0]), int(measurements[5] - offset[1]))

		cv2.rectangle(image, p0, p1, (0,0,255), 1)

		data = str(measurements[-1])[0:4]
		cv2.putText(image, "Psi:{}".format(data), p0, cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),1) # TODO: kom på bra sak att plotta här
		return image

	@staticmethod
	def show_segmentation(outputIm, windowName="Segmentation"):
		output_images = []
		for im in outputIm:
			if len(np.shape(im)) < 3:
				im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
			output_images.append(im)

		output_images = np.hstack(output_images)
		cv2.imshow(windowName, output_images)
		cv2.waitKey(1)

	@staticmethod
	def draw_box_with_dot(image, area, dot, box_color, dot_color):
		p0, p1 = area
		image = cv2.rectangle(image, p0, p1, box_color, 1)

		radius = 1
		thickness = 1
		image = cv2.circle(image, dot, radius, dot_color, thickness)

		return image

	@staticmethod
	def show_error_image(image, outputIm, selected_area, cropped_selected_area):
		x, y = selected_area[0], selected_area[1]
		x2, y2 = x+selected_area[2], y+selected_area[3]

		image = cv2.rectangle(copy(image), (x,y), (x2, y2), (0,0,255), 1)
		cv2.imshow("ERROR INPUT", image)

		x -= cropped_selected_area[0]
		y -= cropped_selected_area[1]
		x2 -= cropped_selected_area[0]
		y2 -= cropped_selected_area[1]
		cropped_image = cv2.rectangle(copy(outputIm[0]), (x,y), (x2, y2), (0,0,255), 1)

		Tracker.show_segmentation([cropped_image, outputIm[1]], "ERROR OUTPUT")
	