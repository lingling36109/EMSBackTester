import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


class AUKF:
    def __init__(self, x, R, Pxx, N=2, window_size=5, alpha=1, beta=1, kappa=0):
        self.N = N
        self.x = x
        self.Pxx = Pxx

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lamb = self.alpha ** 2 * (2 + self.kappa) - self.N
        self.weight = self.N + self.lamb

        self.Wm = np.full(2 * self.N + 1, 1 / (2 * self.weight))
        self.Wm[0] = self.lamb / self.weight
        self.R = R
        self.K = np.array([[0], [0]])

        self.d_history = deque(maxlen=window_size)
        self.r_history = deque(maxlen=window_size)

    def symmetrize(self, matrix):
        n = np.shape(matrix)[0]
        matrix = (matrix + matrix.T) / 2
        matrix = (matrix + 1e-8 * np.eye(n))
        return matrix

    def F(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def H(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac(self, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sigma_points(self):
        self.Pxx = self.symmetrize(self.Pxx)
        sigma_points = np.zeros((2 * self.N + 1, 2))
        sigma_points[0] = self.x
        sqrt_P = np.linalg.cholesky(self.Pxx)
        for i in range(self.N):
            sigma_points[i + 1] = self.x + np.sqrt((self.N + self.lamb)) * sqrt_P[:, i]
            sigma_points[i + 1 + self.N] = self.x - np.sqrt((self.N + self.lamb)) * sqrt_P[:, i]
        return sigma_points

    def predict(self, control):
        sigma_points = self.sigma_points()
        sigma_points = np.array([self.F(x=x, control=control) for x in sigma_points])
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)
        self.Pxx = np.sum([self.Wm[i] * np.outer((sigma_points[i] - self.x), (sigma_points[i] - self.x))
             for i in range(2 * self.N + 1)], axis=0)
        return sigma_points

    def correct(self, control, sigma_points, y_actual):
        y_preds = np.array([self.H(x=x, control=control) for x in sigma_points])
        y = np.sum(self.Wm[:, np.newaxis] * y_preds, axis=0)

        Pyy = np.sum(self.Wm[:, np.newaxis] * np.array([np.dot(a - y, (a - y).T) for a in y_preds])) + self.R
        Pxy = np.sum(
            self.Wm[:, np.newaxis] * np.array([np.dot((a - self.x)[:, np.newaxis], (b - y).T) for a,
            b in zip(sigma_points, y_preds)]), axis=0)

        self.K = Pxy[:, np.newaxis] @ np.linalg.inv(Pyy[:, np.newaxis])
        innovation = y_actual - y
        self.x = self.x + (self.K @ innovation).flatten()

        self.d_history.append(innovation)
        residual = y_actual - self.H(x=self.x, control=control)
        self.r_history.append(residual)

        self.adapt_R(control=control)
        return self.x

    def adapt_R(self, control):
        if len(self.r_history) == 0:
            return
        Cr = np.mean([r @ r.T for r in self.r_history], axis=0)
        H = self.jac(control=control)
        self.R = np.array([Cr + H @ self.Pxx @ H.T])
        self.R = self.symmetrize(self.R)

class BatteryConstants:
    def __init__(self, R_transient=0.005, tau=100, capacity=2.6, SOC=0.655, V_transient=0, R_series=0.005, K=0.5):
        self.R_transient = R_transient
        self.tau = tau
        self.capacity = capacity
        self.SOC = SOC
        self.V_transient = V_transient
        self.R_series = R_series
        self.K = K

    def OCV(self, SOC):
        return 3.81

    def update_parameters(self, arr: np.ndarray):
        self.capacity = arr[0].item()
        self.R_series = arr[1].item()

    def update_states(self, arr: np.ndarray):
        self.V_transient = arr[0].item()
        self.SOC = arr[1].item()


class BatteryModel(AUKF):
    def __init__(self, x: np.ndarray, R: np.ndarray, Pxx: np.ndarray, batteryConstants: BatteryConstants, window_size=25, alpha=0.001, beta=10, kappa=0):
        super().__init__(x, R, Pxx, window_size=window_size, alpha=alpha, beta=beta, kappa=kappa)
        self.batteryConstants = batteryConstants

    def correct(self, control, sigma_points, y_actual):
        x = super().correct(control=control, sigma_points=sigma_points, y_actual=y_actual)
        self.batteryConstants.update_parameters(x)
        return x

    def F(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        return x

    def H(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        capacity, R_series = x
        SOC = self.batteryConstants.SOC
        current = control.item()
        sigma = (1 - (self.batteryConstants.K * R_series)) if current < 0 else (
                    0.9 - (self.batteryConstants.K * R_series))
        SOC += (sigma * current / (3600 * capacity))
        return np.array([SOC])

    def jac(self, control: np.ndarray) -> np.ndarray:
        capacity, R_series = self.x
        current = control.item()
        sigma = (1 - (self.batteryConstants.K * R_series)) if current < 0 else (1 - (self.batteryConstants.K * R_series))
        return np.array([(sigma * current * -1)/(3600 * capacity * capacity), (self.batteryConstants.K * current * -1)/(3600 * capacity)])


class SOCModel(AUKF):
    def __init__(self, x: np.ndarray, R: np.ndarray, Pxx: np.ndarray, batteryConstants: BatteryConstants, window_size=25, alpha=1, beta=1, kappa=0.5):
        super().__init__(x, R, Pxx,  window_size=window_size, alpha=alpha, beta=beta, kappa=kappa)
        self.batteryConstants = batteryConstants

    def correct(self, control, sigma_points, y_actual):
        x = super().correct(control=control, sigma_points=sigma_points, y_actual=y_actual)
        self.batteryConstants.update_states(x)
        return x

    def F(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        V_transient, SOC = x
        current = control.item()
        sigma = (1 - (self.batteryConstants.K * self.batteryConstants.R_series)) if current < 0 else (1 - (self.batteryConstants.K * self.batteryConstants.R_series))
        SOC += (sigma * current / (3600 * self.batteryConstants.capacity))
        V_transient *= np.exp(-1 / self.batteryConstants.tau)
        V_transient += self.batteryConstants.R_transient * (1 - np.exp(-1 / self.batteryConstants.tau)) * current
        return np.array([V_transient, min(max(SOC.item(), 0),1)])

    def H(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        V_transient, SOC = x
        current = control.item()
        V_open = self.batteryConstants.OCV(SOC)
        V_terminal = V_open - current * self.batteryConstants.R_series - V_transient
        return np.array([V_terminal])

    def jac(self, control: np.ndarray) -> np.ndarray:
        return np.array([-1, 0])


class DAUKF:
    def __init__(self, x1, R1, Pxx1, x2, R2, Pxx2, batteryConstants=BatteryConstants()):
        self.batteryConstants = batteryConstants
        self.AUKF1 = BatteryModel(x=x2, R=R2, Pxx=Pxx2, batteryConstants=batteryConstants)
        self.AUKF2 = SOCModel(x=x1, R=R1, Pxx=Pxx1, batteryConstants=batteryConstants)

    def forward(self, current, voltage, SOC):
        points1 = self.AUKF1.predict(control=None)
        result1 = self.AUKF1.correct(control=np.array([current]), sigma_points=points1, y_actual=np.array([SOC]))
        points2 = self.AUKF2.predict(control=np.array([current]))
        result2 = self.AUKF2.correct(control=np.array([current]), sigma_points=points2, y_actual=np.array([voltage]))
        return np.array([result1[0], result1[1], result2[0], result2[1]])


if __name__ == '__main__':
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/training/processed/battery_log_processed.csv")

    df['Rack Current[A]'] = df['Rack Current[A]'] / 60
    df['SOC[%]'] = df['SOC[%]'] / 100
    current_series = df['Rack Current[A]'].values
    voltage_series = df['Avg. Cell V[V]'].values
    SOC_series = df['SOC[%]'].values

    model = DAUKF(x1=np.array([0, SOC_series[0]]), R1=np.array([0.25]), Pxx1=np.array([[1e-4, 0], [0, 1e-8]]),
                  x2=np.array([2.8, 0.05]), R2=np.array([1]), Pxx2=np.array([[1e-10, 0], [0, 1e-10]]))

    results = []

    for k in range(len(current_series)):
        current = current_series[k]
        V_measured = voltage_series[k]
        SOC_measured = SOC_series[k]

        result = model.forward(current, V_measured, SOC_measured)

        results.append(result)

    df_out = pd.DataFrame(results, columns=["Capacity", "Res", "V_transient", "SOC"])
    df_out.to_csv("dual_ukf_predictions.csv", index=False)

    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/dual_ukf_predictions.csv")
    df2 = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/training/processed/battery_log_processed.csv")
    fig, axs = plt.subplots(2, figsize=(20, 10))
    axs[0].plot(df['SOC'], label="SOC")
    axs[0].plot((df2['SOC[%]']/100), label="SOC Real")
    axs[1].plot(df['Res'], label="Resistance")
    fig.legend()
    fig.savefig('Fuuuuuuck5.png', dpi=450)
    fig.show()

