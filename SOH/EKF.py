import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


class UKF:
    def __init__(self, state_dim, x, F, H, jac, window_size=10):
        self.N = state_dim
        self.x = x
        self.F = F
        self.H = H
        self.jac = jac
        self.Pxx = np.diag([1e-3, 1e-4])

        self.alpha = 1
        self.beta = 1
        self.kappa = 0
        self.lamb = self.alpha ** 2 * (self.N + self.kappa) - self.N
        self.weight = self.N + self.lamb

        self.Wm = np.full(2 * self.N + 1, 1 / (2 * self.weight))
        self.Wc = np.full(2 * self.N + 1, 1 / (2 * self.weight))
        self.Wm[0] = self.lamb / self.weight
        self.Wc[0] = self.lamb / self.weight + (1 - self.alpha ** 2 + self.beta)

        self.Q = np.diag([1e-6, 1e-6])
        self.R = np.array([[1e-2]])

        self.Cd = np.zeros((1, 1))
        self.Cr = np.zeros((1, 1))

        self.d_history = deque(maxlen=window_size)
        self.r_history = deque(maxlen=window_size)

    def sigma_points(self):
        # enforce symmetry
        self.Pxx = (self.Pxx + self.Pxx.T) / 2
        # add jitter
        eps = 1e-8
        A = self.weight * (self.Pxx + eps * np.eye(self.N))
        sigma_points = np.zeros((2 * self.N + 1, self.N))
        sigma_points[0] = self.x
        sqrt_P = np.linalg.cholesky(A)
        for i in range(self.N):
            sigma_points[i + 1] = self.x + sqrt_P[:, i]
            sigma_points[i + 1 + self.N] = self.x - sqrt_P[:, i]
        return sigma_points

    def predict(self, control):
        sigma_points = self.sigma_points()
        sigma_points = np.array([self.F(x, control) for x in sigma_points])
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)
        self.Pxx = np.sum(
            [self.Wc[i] * np.outer((sigma_points[i] - self.x), (sigma_points[i] - self.x))
             for i in range(2 * self.N + 1)], axis=0) + self.Q
        return sigma_points

    def update(self, control, sigma_points, y_actual):
        y_preds = np.array([self.H(x, control) for x in sigma_points])
        y = np.sum(self.Wm[:, np.newaxis] * y_preds, axis=0)

        Pyy = np.sum(self.Wc[:, np.newaxis] * np.array([np.dot(a - y, (a - y).T) for a in y_preds])) + self.R
        Pxy = np.sum(
            self.Wc[:, np.newaxis] * np.array([np.dot((a - self.x)[:, np.newaxis], (b - y).T) for a,
            b in zip(sigma_points, y_preds)]), axis=0)

        K = Pxy[:, np.newaxis] @ np.linalg.inv(Pyy)
        innovation = y_actual - y

        self.x = self.x + K @ innovation
        self.Pxx = self.Pxx - K @ Pyy @ K.T
        self.Pxx = (self.Pxx + self.Pxx.T) / 2
        self.Pxx += 1e-8 * np.eye(self.N)

        self.d_history.append(innovation)
        residual = y_actual - self.H(self.x, control)
        self.r_history.append(residual)

        self.adapt_Q(K)
        self.adapt_R()

        return self.x

    def adapt_Q(self, K):
        if len(self.d_history) == 0:
            return
        self.Cd = np.mean([np.outer(d, d) for d in self.d_history], axis=0)
        self.Q = K @ self.Cd @ K.T
        self.Q = (self.Q + self.Q.T) / 2
        self.Q += 1e-8 * np.eye(self.N)

    def adapt_R(self):
        if len(self.r_history) == 0:
            return
        self.Cr = np.mean([np.outer(d, d) for d in self.r_history], axis=0)
        H = self.jac(self.x)
        self.R = self.Cr + H @ self.Pxx @ H.T
        self.R = (self.R + self.R.T) / 2
        self.R += 1e-8 * np.eye(self.R.shape[0])


def F(x: np.ndarray, control: np.ndarray) -> np.ndarray:
    c = 2.6
    sigma = -0.01325 if control.item() > 0 else -0.0135
    R_transient = 0.005378
    tau_transient = 102.5149
    k = np.exp(-1 / tau_transient)
    return np.array([x[0] - ((control.item() * sigma) / (36 * c)), x[1] * k - (R_transient * (1 - k) * control.item())])


def H(x: np.ndarray, control: np.ndarray) -> np.ndarray:
    V_nom = 3
    BAC = 0.1
    R_series = 0.05378
    V_OC = V_nom * (x[0] / (1 - BAC * (1 - x[0])))  #
    V_terminal = V_OC - x[1] - R_series * control.item()
    return np.array([V_terminal])


def jac_H(x: np.ndarray) -> np.ndarray:
    SoC, Vt = x
    V_nom = 3
    BAC = 0.1

    denom = (1 - BAC * (1 - SoC))
    dVoc_dSoC = V_nom * (1 + BAC) / (denom ** 2)

    return np.array([[dVoc_dSoC, -1]])


if __name__ == '__main__':
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/battery_log_processed.csv")

    initial_state = np.array([65.5, -0.11])
    ukf = UKF(state_dim=2, x=initial_state, F=F, H=H, jac=jac_H)

    current_series = df['Rack Current[A]'].values
    voltage_series = df['Avg. Cell V[V]'].values

    predicted_states = []

    for k in range(len(df)):
        I_batt = current_series[k]
        V_meas = voltage_series[k]

        X_pred = ukf.predict(control=I_batt)

        ukf.update(control=I_batt, sigma_points=X_pred, y_actual=V_meas)

        predicted_states.append(ukf.x.copy())

    predicted_df = pd.DataFrame(predicted_states, columns=['SoC', 'V_transient'])
    predicted_df.to_csv("ukf_state_predictions.csv", index=False)
    print("Prediction complete.")
    # Load data

    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/ukf_state_predictions.csv")
    df2 = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/battery_log_processed.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(df['SoC'], label="SOC")
    plt.plot(df2['SOC[%]'], label="SOC Real")
    plt.legend()
    plt.grid(True)
    plt.savefig('Fuuuuuuck4.png', dpi=450)
    plt.show()

