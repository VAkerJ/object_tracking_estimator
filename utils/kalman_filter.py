import filterpy.kalman as kf 
import numpy as np


class Filter():

	def __init__(self, measurements, delta_measurements, selected_area, dt = 0.1, process_var = 10, measurement_var = .5, P = 1000.): 
		#dt = 0.1
		#process_var = 10
		#measurement_var = 1
		x, z, u = len(measurements)+len(delta_measurements), len(measurements), len(delta_measurements)
		k_fil = kf.KalmanFilter(dim_x = x, dim_z = z, dim_u = u)
		A = np.eye(x//2)

		k_fil.F = np.vstack([np.hstack([A,A*dt]),np.hstack([A*0,A*0])]) # state transition matrix	
		k_fil.H = np.hstack([A, A*0]) 									# Measurement function
		k_fil.B = np.vstack([A*0, A]) 									# Control transition matrix
		k_fil.P *= P                 									# Covariance matrix
		k_fil.Q = np.diag(np.random.standard_normal(x))*process_var 	# Process uncertainty/noise (variance)
		k_fil.R = np.diag(np.random.standard_normal(z))*measurement_var	# Measurment uncertainty/noise (variance)

		self.dt, self.k_fil, self.prev_measurements = dt, k_fil, None
		self.selected_area = selected_area
		self.margin = .5
		self.set_x(measurements, delta_measurements)


	def set_x(self, measurements, delta_measurements):

		X = np.hstack([measurements, delta_measurements])
		self.k_fil.x = X
		self.center_est = (int(X[0]), int(X[1]))
		self.X = X

	def update(self, measurements, delta_measurements):

		M = measurements
		D = delta_measurements

		#u = np.asarray(delta_measurements)
		u = delta_measurements
		self.k_fil.predict(u)

		#z = np.asarray(M)
		z = measurements
		self.k_fil.update(z)
		self.set_est_X()

	def set_est_X(self):

		P = list(self.k_fil.P.diagonal())
		X = list(self.k_fil.x[:-1].astype(int))
		X.append(self.k_fil.x[-1])
		self.X = X

	def set_selected_area(self, selected_area):
		sigmoid = lambda t: 1/(1+np.exp(-abs(t))) -.5

		X = self.X
		center_est = X[0:2]
		p0 = (int(X[2]), int(X[3]))
		p1 = (int(X[4]), int(X[5])) # hämtar de estimerade gränserna för masken

		mask_len = (p1[0]- p0[0], p1[1]- p0[1])

		target_area = (p0[0]-mask_len[0]*self.margin, p0[1]-mask_len[1]*self.margin, \
		p1[0]+mask_len[0]*self.margin, p1[1]+mask_len[1]*self.margin) # sätter ut en marginal runt masken

		(x_min, y_min, x_len, y_len) = selected_area # definerar det nuvarande området
		x_max = x_min + x_len
		y_max = y_min + y_len
		current_area = [x_min, y_min, x_max, y_max]

		delta = [t-c for t,c in zip(target_area, current_area)] # tar ut skillnaden

		current_area = [c+d*sigmoid(d) for c,d in zip(current_area, delta)] # modifierar koordinaterna

		x_min, y_min = int(current_area[0]), int(current_area[1])
		x_len, y_len = int(current_area[2])-x_min, int(current_area[3])-y_min

		selected_area = (x_min, y_min, x_len, y_len)

		self.selected_area = selected_area
		self.center_est = tuple(center_est)

	def get_new_area(self):
		return self.selected_area

	def get_center_est(self):
		return self.center_est

		