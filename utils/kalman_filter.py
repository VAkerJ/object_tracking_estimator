import filterpy.kalman as kf 
import filterpy.common as co
import numpy as np


class Filter():

	def __init__(self, measurements, delta_measurements): # TODO: flytta till egen skript (kalman.py?)
		dt = 0.1
		k_fil = kf.KalmanFilter(dim_x = 4, dim_z = 2, dim_u = 2) 
		k_fil.F = np.array([[1.,0.,dt,0.], 		# state transition matrix
							[0.,1.,0.,dt],
							[0.,0.,0.,0.],
							[0.,0.,0.,0.]])   
		k_fil.H = np.array([[1.,0.,0.,0.],		# Measurement function
							[0.,1.,0.,0.]])    	
		k_fil.B = np.array([[0.,0.]				# Control transition matrix
							[0.,0.]
							[1.,0.]
							[0.,1.]])
		k_fil.P *= 1000.                   		# Covariance matrix
		k_fil.Q = co.Q_discrete_white_noise(4,dt,.1) 	# Process uncertainty/noise
		k_fil.R = np.array([[1.,0.],			# Measurment uncertainty/noise
							[0.,1.]]).multiply(5)		
		
		self.dt, self.k_fil, self.prev_measurements = dt, k_fil, None
		self.set_x(measurements, delta_measurements)

	def set_x(self, measurements, delta_measurements):
		x0, y0 = measurements[0], measurements[1]
		v_x, v_y = delta_measurements[0], delta_measurements[1]
		self.k_fil.x = np.array([[x0],[y0],[v_x],[v_y]])
		self.center_est = (x0, y0)

	def update(self, measurements, delta_measurements):
		z = measurements[0:2]
		self.k_fil.predict()
		self.k_fil.update(z)

	def set_selected_area(self, selected_area):
		P = []
		X = []
		for i in range(len(self.k_fil.x)):
			X.append(int(self.k_fil.x[i]))
			P.append(float(self.k_fil.P.diagonal()[i]))
		center_est = X[0:2]

		(x_min, y_min, x_len, y_len) = selected_area
		x_min = int(center_est[0] - x_len/2)
		y_min = int(center_est[1] - y_len/2)
		selected_area = (x_min, y_min, x_len, y_len)

		self.selected_area = selected_area
		self.center_est = tuple(center_est)

	def get_new_area(self):
		return self.selected_area

	def get_center_est(self):
		return self.center_est

		