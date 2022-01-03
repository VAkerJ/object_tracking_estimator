import numpy as np
from copy import copy
import cv2

class Tracker():
	def __init__(self, Segment, Filter, selected_area, verbose):
		self.filter = Filter

		self.prev_measurements = None
		self.segment = Segment
		self.verbose = verbose
		self.selected_area = selected_area

	def update(self, base_image):
		selected_area = self.selected_area

		# segmentera och hämta mätvärden
		outputMask, _, cropped_selected_area, outputIm = self.segment(base_image, selected_area)
		success, measurements, delta_measurements = Tracker.get_measurements(outputMask, cropped_selected_area, self.prev_measurements)

		# skriver ut sista resultaten ifall error upstår, bör kanske skrivas om i framtiden så att den kan hantera att den tappar objektet
		if not success:
			x, y = selected_area[0], selected_area[1]
			x2, y2 = x+selected_area[2], y+selected_area[3]

			image = cv2.rectangle(copy(base_image), (x,y), (x2, y2), (0,0,255), 1)
			cv2.imshow("ERROR INPUT", image)

			x -= cropped_selected_area[0]
			y -= cropped_selected_area[1]
			x2 -= cropped_selected_area[0]
			y2 -= cropped_selected_area[1]
			cropped_image = cv2.rectangle(copy(outputIm[0]), (x,y), (x2, y2), (0,0,255), 1)

			Tracker.show_segmentation([cropped_image, outputIm[1]], "ERROR OUTPUT")
			return success, base_image

		# skapar en liten ruta med de olika små output-bilderna
		if self.verbose > 0:
			measurement_im = Tracker.illustrate_measurements(copy(outputIm[0]), measurements, cropped_selected_area[0:2])
			outputIm.append(measurement_im)
			Tracker.show_segmentation(outputIm)

		# ritar ut rektangeln i stora output-bilden tsm med punkt för center och estimerat center
		center_measured = (int(measurements[0]), int(measurements[1]))
		center_estimate = self.filter.get_center_est()

		image = Tracker.draw_box_with_center(copy(base_image), selected_area, center_measured, center_estimate)

		# updatera filtret?
		self.update_filter(measurements, delta_measurements) # TODO: anderberg, gör din grej
		# updatera var rektangeln är
		self.update_selected_area(measurements, delta_measurements) # TODO: gör bättre?

		self.prev_measurements = measurements

		return success, image

	def update_filter(self, measurements, delta_measurements):
		self.filter.update(measurements, delta_measurements)

	def update_selected_area(self, measurements, delta_measurements): # metod för att flytta den valda rutan
		#x, y, width, height = self.selected_area
		#x = int(measurements[0] - width/2 + delta_measurements[0])
		#y = int(measurements[1] - height/2 + delta_measurements[1])
		#self.selected_area = (x, y, width, height)
		self.filter.set_selected_area(self.selected_area)
		self.selected_area = self.filter.get_new_area()

	#-----------------------------------------------------------------------------
	# get image info metoder

	@staticmethod
	def get_measurements(mask, rectangle, prev_measurements):
		success = True

		try:
			measurements = Tracker.get_mask_info(mask, rectangle)
		except ZeroDivisionError:
			# ändra dehär för att hantera när bilden försvinner
			print("[Alert] Empty mask using rectangle={}, aborting".format(rectangle))
			success = False
			return success, None, None


		# räkna ut föränding i mätningar
		# prev bör alltså inte innehålla förra hastigheterna
		if prev_measurements is not None:
			delta_measurements = [cur - prev for cur, prev in zip(measurements, prev_measurements)]
			delta_measurements = np.asarray(delta_measurements, dtype=np.float32)
		else:
			delta_measurements = np.zeros(np.shape(measurements))

		measurements = np.asarray(measurements, dtype=np.float32)

		return success, measurements, delta_measurements # borde kanske slå ihop?

	@staticmethod
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

		# rektangel blir nog bättre än ellipse till en början, illustreringsfunktionen måste ändras om detta ändras
		#mask_density = index_amount *4/(pi*mask_height*mask_width) # formel för ellipse, kanske bör ändras för att hantera outliers i pixlar?
		mask_density = index_amount/(mask_height*mask_width) # formel för rektangel

		index_x, index_y = index_x+rectangle[0], index_y+rectangle[1] # nu jobbar man inte med croppade koordinater längre

		return index_x, index_y, index_amount, mask_height, mask_width, mask_density

	#-----------------------------------------------------------------------------
	# illustrativa metoder
	@staticmethod
	def illustrate_measurements(image, measurements, offset=[0,0]):
		# for drawing drawing the cropped images and masks in the small window
		x = int(measurements[0] - offset[0])
		y = int(measurements[1] - offset[1])
		size = tuple([int(c) for c in measurements[3:5]])

		# valde att byta till rektangel istället för ellipse
		#cv2.ellipse(image, (x,y), size, 0,0,360, color=(0,0,255), thickness=1)
		x = x - size[0]//2
		y = y - size[1]//2
		cv2.rectangle(image, (x,y), (x+size[0], y+size[1]), (0,0,255), 1)

		data = str(measurements[-1])[0:4]
		cv2.putText(image, "Psi:{}".format(data), (x,y), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),1) # TODO: kom på bra sak att plotta här
		return image

	@staticmethod
	def show_segmentation(outputIm, windowName="output images"):
		output_images = []
		for im in outputIm:
			if len(np.shape(im)) < 3:
				im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
			output_images.append(im)

		output_images = np.hstack(output_images)
		cv2.imshow(windowName, output_images)
		cv2.waitKey(1)

	@staticmethod
	def draw_box_with_center(image, selected_area, center1, center2):
		# for drawing the box with estimated and measured center in the large outputwindow
		p0 = selected_area[0:2]
		p1 = (selected_area[0]+selected_area[2], selected_area[1]+selected_area[3])

		image = cv2.rectangle(image, p0, p1, (0,0,255), 1)

		radius = 1
		thickness = 1
		color = (0,255,0)

		image = cv2.circle(image, center1, radius, color, thickness)

		color = (255,0,0)
		image = cv2.circle(image, center2, radius, color, thickness)

		return image

	