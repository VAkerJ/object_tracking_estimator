import filterpy.kalman as kf 
import numpy as np


class Filter():

	def __init__(self, measurements, delta_measurements): 
		dt = 0.1
		process_var = 1
		measurement_var = 0.1
		k_fil = kf.KalmanFilter(dim_x = 4, dim_z = 2, dim_u = 2) 
		k_fil.F = np.array([[1.,0.,dt,0.], 		# state transition matrix
							[0.,1.,0.,dt],
							[0.,0.,1.,0.],
							[0.,0.,0.,1.]])   
		k_fil.H = np.array([[1.,0.,0.,0.],		# Measurement function
							[0.,1.,0.,0.]])    	
		k_fil.B = np.array([[0.,0.],			# Control transition matrix
							[0.,0.],
							[1.,0.],
							[0.,1.]])
		k_fil.P *= 1000.                   		# Covariance matrix
		# k_fil.Q = np.diag(np.random.standard_normal(4))*process_var 	# Process uncertainty/noise (variance)
		# k_fil.R = np.diag(np.random.standard_normal(2))*measurement_var	# Measurment uncertainty/noise (variance)
		k_fil.Q = np.eye(4)*process_var 	# Process uncertainty/noise (variance)
		k_fil.R = np.eye(2)*measurement_var	# Measurment uncertainty/noise (variance)
		self.dt, self.k_fil, self.prev_measurements = dt, k_fil, None
		self.delta_measurements = delta_measurements
		self.set_x(measurements, delta_measurements)
		# print(k_fil)

	def set_x(self, measurements, delta_measurements):
		x0, y0 = measurements[0], measurements[1]
		v_x, v_y = delta_measurements[0], delta_measurements[1]
		self.k_fil.x = np.array([[x0],[y0],[v_x],[v_y]])
		self.center_est = (int(x0), int(y0))

	def update(self, measurements, delta_measurements):
		u = delta_measurements[0:2]
		u = np.array([u]).T
		self.k_fil.predict(u)
		z = measurements[0:2]
		self.k_fil.update(z)
		# print(self.k_fil.x,'\n')
		# print(self.k_fil.K)
		self.delta_measurements = delta_measurements

	def set_selected_area(self, selected_area):
		P = []
		X = []
		for i in range(len(self.k_fil.x)):
			X.append(int(self.k_fil.x[i]))
			P.append(float(self.k_fil.P.diagonal()[i]))
		center_est = X[0:2]
		(x_min, y_min, x_len, y_len) = selected_area

		delta_density = self.delta_measurements[5]
		print(delta_density)
		dxy = 1
		lim = 0
		if delta_density > lim:
			x_len += dxy
			y_len += dxy
		elif delta_density < lim:
			x_len -= dxy
			y_len -= dxy
		
		print(x_len,y_len)

		x_min = int(center_est[0] - x_len/2)
		y_min = int(center_est[1] - y_len/2)
		selected_area = (x_min, y_min, x_len, y_len)

		self.selected_area = selected_area
		self.center_est = tuple(center_est)

	def get_new_area(self):
		return self.selected_area

	def get_center_est(self):
		return self.center_est

		