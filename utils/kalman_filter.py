import filterpy.kalman as kf 
import numpy as np
import matplotlib.pyplot as plt


class Filter():

	def __init__(self, measurements, delta_measurements, selected_area, dt = 0.1, process_var = 10, measurement_var = .5, P = 1000., setup = 2): 
		if setup == 1:
			k_fil = kf.KalmanFilter(dim_x = 4, dim_z = 2, dim_u = 2) 
			k_fil.F = np.array([[1.,0.,dt,0.], 	# State transition matrix
								[0.,1.,0.,dt],
								[0.,0.,0.,0.],
								[0.,0.,0.,0.]])   
			k_fil.H = np.array([[1.,0.,0.,0.],	# Measurement function
								[0.,1.,0.,0.]])    	
			k_fil.B = np.array([[0.,0.],		# Control transition matrix
								[0.,0.],
								[1.,0.],
								[0.,1.]])
			k_fil.P *= 1000.                   	# Covariance matrix
			k_fil.Q = np.eye(4)*process_var 	# Process uncertainty/noise (variance)
			k_fil.R = np.eye(2)*measurement_var	# Measurment uncertainty/noise (variance)
		
		else:
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
		self.delta_measurements = delta_measurements
		self.selected_area = selected_area
		self.data = [[],[]]	# Covariance and Kalman gain
		self.setup = setup
		self.margin = .5
		self.set_x(measurements, delta_measurements)


	def set_x(self, measurements, delta_measurements):
		if self.setup == 1: 
			X = np.append(measurements[0:2],delta_measurements[0:2])			
		else:
			X = np.hstack([measurements, delta_measurements])
			self.center_est = (int(X[0]), int(X[1]))
		self.k_fil.x = X	
		self.X = X

	def update(self, measurements, delta_measurements):
		# M = measurements
		# D = delta_measurements
		
		#u = np.asarray(delta_measurements)
		u = delta_measurements
		#z = np.asarray(M)
		z = measurements

		if self.setup == 1: 
			u = u[0:2]
			z = z[0:2]

		self.k_fil.predict(u)
		self.k_fil.update(z)

		self.set_est_X()

		self.data[0].append(list(np.diag(self.k_fil.P_post)[0:1])[0])
		self.data[1].append(self.k_fil.K[0][0])
		self.delta_measurements = delta_measurements
		

	def set_est_X(self):
		P = list(self.k_fil.P.diagonal())
		X = list(self.k_fil.x[:-1].astype(int))
		X.append(self.k_fil.x[-1])
		self.X = X

	def set_selected_area(self, selected_area):
		sigmoid = lambda t: 1/(1+np.exp(-abs(t))) -.5

		X = self.X
		center_est = X[0:2]
		(x_min, y_min, x_len, y_len) = selected_area

		if self.setup == 1:
			print(self.delta_measurements)
			delta_density = self.delta_measurements[5]
			dxy = 1
			lim = 0
			if delta_density > lim:
				x_len += dxy
				y_len += dxy
			elif delta_density < lim:
				x_len -= dxy
				y_len -= dxy

		else:
			x_min = int(center_est[0] - x_len/2)
			y_min = int(center_est[1] - y_len/2)
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

	def plotData(self):
		data = self.data
		iterations = range(len(data[0]))
		fig, axs= plt.subplots(ncols=2)
		plt.grid()
		axs[0].scatter(iterations, data[0], linewidth=1.0)
		axs[1].plot(iterations, data[1], linewidth=1.0)
		axs[0].set(ylim=(0, 1.1))
		axs[1].set(ylim=(0, 1.1))
		axs[0].set_xlabel('Number of iterations')
		axs[0].set_ylabel('Covariance')
		axs[1].set_xlabel('Number of iterations')
		axs[0].set_ylabel('Kalman gain')
		plt.show()

		