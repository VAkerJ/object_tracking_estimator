import numpy as np
import filterpy.kalman as kf
import filterpy.common as co
import time


def main():

    my_filter = kf.KalmanFilter(dim_x = 2, dim_z = 1, dim_u = 0)
    print(my_filter)
    my_filter.x = np.array([[2.],[0.]])         # initial state (location and velocity)
    my_filter.F = np.array([[1.,1.],[0.,1.]])   # state transition matrix

    dt = 0.1

    my_filter.H = np.array([[1.,0.]])           # Measurement function
    my_filter.P *= 1000.                        # covariance matrix
    my_filter.R = 5                             # state uncertainty
    my_filter.Q = co.Q_discrete_white_noise(2, dt, .1) # process uncertainty

    print(my_filter)
    t = 0
    while True:
        my_filter.predict()
        my_filter.update(getMeasurement(t))

        # do something with the output
        x = my_filter.x
        print("t = {:.1f}\t\t {}".format(t,my_filter.x.round(1).T))
        t += dt
        time.sleep(dt)

def getMeasurement(t):
    return 2*t


if __name__ == "__main__":
    main()

