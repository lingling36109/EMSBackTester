import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import mean_squared_error


class UKF:
    def __init__(self, x, R, Q, Pxx, N=2, window_size=5, alpha=1, beta=1, kappa=0):
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
        self.Q = Q
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

    def sigma_points(self):
        self.Pxx = self.symmetrize(self.Pxx)
        sigma_points = np.zeros((2 * self.N + 1, self.N))
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
        self.Pxx = (np.sum([self.Wm[i] * np.outer((sigma_points[i] - self.x), (sigma_points[i] - self.x))
             for i in range(2 * self.N + 1)], axis=0)) + self.Q
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

        return self.x


class BatteryConstants:
    def __init__(self, R_transient=0.005, tau=100, capacity=2.8, SOC=0.655, V_transient=0, R_series=0.05, K=1.0, sigma1=1, sigma2=0.9):
        self.R_transient = R_transient
        self.tau = tau
        self.capacity = capacity
        self.SOC = SOC
        self.V_transient = V_transient
        self.R_series = R_series
        self.K = K
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def OCV(self, SOC):
        raise NotImplementedError

    def CE(self, current, R_series):
        raise NotImplementedError

    def update_parameters(self, arr: np.ndarray):
        self.capacity = arr[0].item()
        # self.R_series = arr[1].item()

    def update_states(self, arr: np.ndarray):
        self.V_transient = arr[0].item()
        self.SOC = arr[1].item()


class BatteryV1(BatteryConstants):
    def OCV(self, SOC):
        return 3.81

    def CE(self, current, R_series):
        return (self.sigma1 - (self.K * R_series)) if current < 0 else (
                    self.sigma2 - (self.K * R_series))


class BatteryV2(BatteryConstants):
    def __init__(self, R_transient=0.005, tau=100, capacity=4.0, SOC=1.0, V_transient=0, R_series=0.05, K=1.0, sigma1=1.0, sigma2=1.0):
        super().__init__(R_transient=R_transient, tau=tau, capacity=capacity, SOC=SOC, V_transient=V_transient, R_series=R_series, K=K, sigma1=sigma1, sigma2=sigma2)

    def OCV(self, SOC):
        if 0.15 <= SOC:
            return 0.9/0.85 * (SOC - 0.15) + 3.3
        else:
            return 0.8/0.15 * SOC + 2.5

    def CE(self, current, R_series):
        return (self.sigma1 - (self.K * R_series)) if current < 0 else (
                    self.sigma2 - (self.K * R_series))


class BatteryModel(UKF):
    def __init__(self, x: np.ndarray, R: np.ndarray, Q: np.ndarray, Pxx: np.ndarray, batteryConstants: BatteryConstants, window_size=25, alpha=1, beta=1, kappa=0, N=2):
        super().__init__(x=x, R=R, Q=Q, Pxx=Pxx, N=N, window_size=window_size, alpha=alpha, beta=beta, kappa=kappa)
        self.batteryConstants = batteryConstants

    def correct(self, control, sigma_points, y_actual):
        x = super().correct(control=control, sigma_points=sigma_points, y_actual=y_actual)
        self.batteryConstants.update_parameters(x)
        return x

    def F(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        return x

    def H(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        current, _ = control
        SOC = self.batteryConstants.SOC
        V_terminal = self.batteryConstants.OCV(SOC) + self.batteryConstants.V_transient + current * self.batteryConstants.R_series
        return np.array([V_terminal])


class SOCModel(UKF):
    def __init__(self, x: np.ndarray, R: np.ndarray, Q:np.ndarray, Pxx: np.ndarray, batteryConstants: BatteryConstants, window_size=25, alpha=1, beta=1, kappa=0.5):
        super().__init__(x=x, R=R, Pxx=Pxx, Q=Q, window_size=window_size, alpha=alpha, beta=beta, kappa=kappa)
        self.batteryConstants = batteryConstants

    def correct(self, control, sigma_points, y_actual):
        x = super().correct(control=control, sigma_points=sigma_points, y_actual=y_actual)
        self.batteryConstants.update_states(x)
        return x

    def F(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        V_transient, SOC = x
        current, time = control
        sigma = self.batteryConstants.CE(current, self.batteryConstants.R_series)
        SOC += (sigma * current / (3600 * self.batteryConstants.capacity))
        V_transient *= np.exp(-time / self.batteryConstants.tau)
        V_transient += self.batteryConstants.R_transient * (1 - np.exp(-time / self.batteryConstants.tau)) * current
        return np.array([V_transient, min(max(SOC.item(), 0), 1)])

    def H(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        V_transient, SOC = x
        current, time = control
        V_terminal = self.batteryConstants.OCV(SOC) + current * self.batteryConstants.R_series + V_transient
        return np.array([V_terminal])


class DUKF:
    def __init__(self, x1, R1, Q1, Pxx1, x2, R2, Q2, Pxx2, batteryConstants=BatteryConstants()):
        self.batteryConstants = batteryConstants
        self.UKF1 = BatteryModel(N=1, x=x2, R=R2, Q=Q2, Pxx=Pxx2, batteryConstants=batteryConstants)
        self.UKF2 = SOCModel(x=x1, R=R1, Q=Q1, Pxx=Pxx1, batteryConstants=batteryConstants)

    def forward(self, current, voltage, dt):
        points1 = self.UKF1.predict(control=None)
        points2 = self.UKF2.predict(control=np.array([current, dt]))
        result1 = self.UKF1.correct(control=np.array([current, dt]), sigma_points=points1, y_actual=np.array([voltage]))
        result2 = self.UKF2.correct(control=np.array([current, dt]), sigma_points=points2, y_actual=np.array([voltage]))
        return np.array([result1[0], result2[0], result2[1]])


class CapacityUKF(UKF):
    def __init__(self, x: np.ndarray, R: np.ndarray, Q:np.ndarray, Pxx: np.ndarray, batteryConstants: BatteryConstants, window_size=25, alpha=1, beta=1, kappa=0.5):
        super().__init__(N=1, x=x, R=R, Pxx=Pxx, Q=Q, window_size=window_size, alpha=alpha, beta=beta, kappa=kappa)
        self.batteryConstants = batteryConstants

    def F(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        return x

    def H(self, x: np.ndarray, control: np.ndarray) -> np.ndarray:
        capacity = x.item()
        current, time, prevSOC, currSOC = control
        sigma = self.batteryConstants.CE(current, self.batteryConstants.R_series)
        d = currSOC - prevSOC + ((sigma * current * time) / capacity)
        return np.array([d])


if __name__ == '__main__':
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/simulation_output3.csv")

    # df['Rack Current[A]'] = df['Rack Current[A]'] / 60
    # df['SOC[%]'] = 1.0 - (df['Discharge capacity [A.h]'] /(5.0 - df['Total capacity lost to side reactions [A.h]']))
    df['SOC[%]'] = 1.0 - (df['Discharge capacity [A.h]'] / 5.0)
    df['Delta t [s]'] = (df['Time [s]'].shift(-1) - df['Time [s]']).fillna(0)
    current_series = df['Current [A]'].values
    voltage_series = df['Terminal voltage [V]'].values
    SOC_series = df['SOC[%]'].values
    dt_series = df['Delta t [s]'].values

    # model = DUKF(x1=np.array([0, SOC_series[0]]), R1=np.array([1e-2]), Q1=np.array([[1e-5, 0], [0, 1e-10]]), Pxx1=np.array([[1e-1, 0], [0, 1e-3]]),
    #               x2=np.array([5.0]), R2=np.array([1e-8]), Q2=np.array([1e-4]), Pxx2=np.array([1e-4]), batteryConstants=BatteryV2())

    model = CapacityUKF(x=np.array([5.0]), R=np.array([1e8]), Q=np.array([1e2]), Pxx=np.array([1e-8]), batteryConstants=BatteryV2())

    results = []

    for k in range(1, len(current_series)):
        current = current_series[k]
        # V_measured = voltage_series[k]
        dt = dt_series[k]
        SOC_series1 = SOC_series[k]
        SOC_series2 = SOC_series[k-1]

        result = model.predict(control=None)
        result = model.correct(control=np.array([current, dt, SOC_series2, SOC_series1]), sigma_points=result, y_actual=np.array([0]))

        results.append(result)

    results.append([4.98,0,0])

    df_out = pd.DataFrame(results, columns=["Capacity", "V_transient", "SOC"])
    df_out.to_csv("dual_ukf_predictions.csv", index=False)
    fig, axs = plt.subplots(1, figsize=(90, 15))
    # axs[0].plot(df['Time [s]'], df_out['SOC'], label="SOC")
    # axs[0].plot(df['Time [s]'], (df['SOC[%]']), label="SOC Real")
    #
    # print(mean_squared_error(df_out['SOC'], df['SOC[%]']))

    # axs[1].plot(df['Time [s]'], df_out['Res'], label="Resistance")
    axs.plot(df['Time [s]'], df_out['Capacity'], label="Capacity")
    axs.plot(df['Time [s]'], 5.0 - df['Total capacity lost to side reactions [A.h]'], label="Capacity")
    fig.legend()
    fig.savefig('Fuuuuuuck.png', dpi=300)
    fig.show()

