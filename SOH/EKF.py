import numpy as np
import filterpy
from filterpy.kalman import ExtendedKalmanFilter

def hx(x):
    return np.array([x[0]])  # measurement function

def H_jacobian(x):
    return np.array([[1, 0]])  # Jacobian of hx

ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
ekf.x = np.array([0, 1])      # initial state
ekf.P *= 1000                 # covariance
ekf.R *= 5                    # measurement noise
ekf.F = np.array([[1, 1],     # state transition
                  [0, 1]])
ekf.Q = np.eye(2)             # process noise

z = np.array([2.0])
ekf.predict()
ekf.update(z, H_jacobian, hx)

print("Post-update state:", ekf.x)