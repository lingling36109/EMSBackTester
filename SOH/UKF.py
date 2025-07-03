import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from functools import partial
from sklearn.metrics import root_mean_squared_error


class UKF:
    def __init__(self, state_dim, x, F, H, jac, Q, R, Pxx, window_size=5, alpha=1, beta=1, kappa=1):
        self.N = state_dim
        self.x = x
        self.F = F
        self.H = H
        self.jac = jac
        self.Pxx = Pxx

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lamb = self.alpha ** 2 * (self.N + self.kappa) - self.N
        self.weight = self.N + self.lamb

        self.Wm = np.full(2 * self.N + 1, 1 / (2 * self.weight))
        self.Wc = np.full(2 * self.N + 1, 1 / (2 * self.weight))
        self.Wm[0] = self.lamb / self.weight
        self.Wc[0] = self.lamb / self.weight + (1 - self.alpha ** 2 + self.beta)

        self.Q = Q
        self.R = R

        self.Cd = np.zeros((1, 1))
        self.Cr = np.zeros((1, 1))

        self.d_history = deque(maxlen=window_size)
        self.r_history = deque(maxlen=window_size)

    def sigma_points(self):
        self.Pxx = (self.Pxx + self.Pxx.T) / 2
        self.Pxx = self.weight * (self.Pxx + 1e-8 * np.eye(self.N))
        sigma_points = np.zeros((2 * self.N + 1, self.N))
        sigma_points[0] = self.x
        sqrt_P = np.linalg.cholesky(self.Pxx)
        for i in range(self.N):
            sigma_points[i + 1] = self.x + sqrt_P[:, i]
            sigma_points[i + 1 + self.N] = self.x - sqrt_P[:, i]
        return sigma_points

    def predict(self, control):
        sigma_points = self.sigma_points()
        sigma_points = np.array([self.F(x=x, control=control) for x in sigma_points])
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)
        self.Pxx = np.sum(
            [self.Wc[i] * np.outer((sigma_points[i] - self.x), (sigma_points[i] - self.x))
             for i in range(2 * self.N + 1)], axis=0) + self.Q
        return sigma_points

    def update(self, control, sigma_points, y_actual):
        y_preds = np.array([self.H(x=x, control=control) for x in sigma_points])
        y = np.sum(self.Wm[:, np.newaxis] * y_preds, axis=0)

        Pyy = np.sum(self.Wc[:, np.newaxis] * np.array([np.dot(a - y, (a - y).T) for a in y_preds])) + self.R
        Pxy = np.sum(
            self.Wc[:, np.newaxis] * np.array([np.dot((a - self.x)[:, np.newaxis], (b - y).T) for a,
            b in zip(sigma_points, y_preds)]), axis=0)

        K = Pxy[:, np.newaxis] @ np.linalg.inv(Pyy)
        innovation = y_actual - y

        # innovation = np.array([0.001])
        #
        # print(K)
        # print(innovation)

        self.x = self.x + K @ innovation
        self.Pxx = self.Pxx - K @ Pyy @ K.T
        self.Pxx = (self.Pxx + self.Pxx.T) / 2
        self.Pxx += 1e-8 * np.eye(self.N)

        self.d_history.append(innovation)
        residual = y_actual - self.H(x=self.x, control=control)
        self.r_history.append(residual)

        self.adapt_Q(K)
        self.adapt_R(control)

        return self.x

    def getX(self):
        return self.x

    def adapt_Q(self, K):
        self.Cd = np.mean([np.outer(d, d) for d in self.d_history], axis=0)
        self.Q = K @ self.Cd @ K.T
        self.Q = (self.Q + self.Q.T) / 2
        self.Q += 1e-8 * np.eye(self.N)

    def adapt_R(self, control):
        self.Cr = np.mean([np.outer(d, d) for d in self.r_history], axis=0)
        H = self.jac(x=self.x, control=control)
        self.R = self.Cr + H @ self.Pxx @ H.T
        self.R = (self.R + self.R.T) / 2
        self.R += 1e-8 * np.eye(self.R.shape[0])


class DUKF:
    def __init__(self, initial1, initial2, func1, func2, func3, func4, func5, func6, c, sigma1, sigma2, R_transient,
                 tau_transient, V_nom, BAC, alpha1, alpha2, beta1, beta2, kappa1, kappa2, R1, Q1, Q2, P1, P2, R2, Q3, P3):
        func1 = partial(func1, c=c, sigma1=sigma1, sigma2=sigma2, R_transient=R_transient, tau_transient=tau_transient)
        func2 = partial(func2, V_nom=V_nom, BAC=BAC)
        func3 = partial(func3, V_nom=V_nom, BAC=BAC)
        func5 = partial(func5, V_nom=V_nom, BAC=BAC)
        self.UKF1 = UKF(state_dim=2, x=initial1, F=func1, H=func2, jac=func3, alpha=alpha1, beta=beta1, kappa=kappa1, R=np.array([[R1]]), Q=np.diag([Q1, Q2]), Pxx=np.diag([P1, P2]), window_size=50)
        self.UKF2 = UKF(state_dim=1, x=initial2, F=func4, H=func5, jac=func6, alpha=alpha2, beta=beta2, kappa=kappa2, R=np.array([[R2]]), Q=np.diag([Q3]), Pxx=np.diag([P3]), window_size=50)
        self.UKF2.adapt_Q = lambda *_: None
        self.UKF2.adapt_R = lambda *_: None

    def step(self, control1, control2, y1, y2):
        pred1 = self.UKF1.predict(control1)
        control1 = np.append(control1, self.UKF2.getX())
        output1 = self.UKF1.update(control1, pred1, y1)
        pred2 = self.UKF2.predict(control2)
        control2 = np.append(control2, output1)
        output2 = self.UKF2.update(control2, pred2, y2)
        return output1, output2


def F1(c, sigma1, sigma2, R_transient, tau_transient, x: np.ndarray, control: np.ndarray) -> np.ndarray:
    sigma = sigma1 if control.item() > 0 else sigma2
    SoC_new = x[0] + ((control.item() * sigma) / (36 * c))
    k = np.exp(-1 / tau_transient)
    Vt_new = x[1] * k - (R_transient * (1 - k) * control.item())
    return np.array([SoC_new, Vt_new])


def H1(V_nom, BAC, x: np.ndarray, control: np.ndarray) -> np.ndarray:
    R_series = control[1].item()
    SOC = x[0].item()
    V_OC = V_nom * (SOC / (1 - BAC * (1 - SOC)))
    V_terminal = V_OC - x[1] - R_series * control[0].item()
    return np.array([V_terminal])


def F2(x: np.ndarray, control: np.ndarray) -> np.ndarray:
    return x


def H2(V_nom, BAC, x: np.ndarray, control: np.ndarray) -> np.ndarray:
    R_series = x[0].item()
    SoC = control[1].item()
    V_transient = control[2].item()

    V_OC = V_nom * (SoC / (1 - (BAC * (1 - SoC))))
    V_terminal = V_OC - V_transient - R_series * control[0]
    return np.array([V_terminal])


def jac_H1(V_nom, BAC, x: np.ndarray, control: np.ndarray) -> np.ndarray:
    SoC = x[0]
    denom = (1 - BAC * (1 - SoC))
    dVoc_dSoC = V_nom * (1 + BAC) / (denom ** 2)

    return np.array([[dVoc_dSoC, -1]])


def jac_H2(x: np.ndarray, control: np.ndarray) -> np.ndarray:
    return np.array([[-control[0].item()]])


def objective_function(solution):
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/battery_log_processed.csv")
    alpha1 = solution[0]
    beta1 = solution[1]
    kappa1 = solution[2]
    alpha2 = solution[3]
    beta2 = solution[4]
    kappa2 = solution[5]
    P1 = solution[6]
    P2 = solution[7]
    P3 = solution[8]
    R1 = solution[9]
    R2 = solution[10]
    Q1 = solution[11]
    Q2 = solution[12]
    Q3 = solution[13]

    initial_soc = 0.655
    initial_Vt = 0
    initial_R_series = 0.05

    # Initialize DUKF
    dukf = DUKF(
        initial1=np.array([initial_soc, initial_Vt]),
        initial2=np.array([initial_R_series]),
        func1=F1,
        func2=H1,
        func3=jac_H1,
        func4=F2,
        func5=H2,
        func6=jac_H2,
        c=2.6,
        sigma1=-0.015,
        sigma2=-0.015,
        R_transient=0.005,
        tau_transient=100,
        V_nom=4,
        BAC=0.1,
        alpha1=alpha1,
        beta1=beta1,
        kappa1=kappa1,
        alpha2=alpha2,
        beta2=beta2,
        kappa2=kappa2,
        Q1=Q1,
        Q2=Q2,
        Q3=Q3,
        P1=P1,
        P2=P2,
        P3=P3,
        R1=R1,
        R2=R2,
    )

    current_series = df['Fuck2'].values
    voltage_series = df['Cell Sum V[V]'].values

    state_preds = []
    resistance_preds = []

    try:
        for k in range(len(current_series)):
            I = current_series[k]
            V_measured = np.array([voltage_series[k]])

            output1, output2 = dukf.step(control1=np.array([I]), control2=np.array([I]), y1=V_measured, y2=V_measured)

            soc_est, vt_est = output1
            R_series_est = output2[0]

            state_preds.append([soc_est, vt_est])
            resistance_preds.append(R_series_est)

        df_out = pd.DataFrame(state_preds, columns=["SOC[%]", "V_transient[V]"])
        df_out["R_series[A]"] = resistance_preds
        rms1 = root_mean_squared_error(df['Fuck3'], df_out['SOC[%]'])
        return rms1
    except Exception as e:
        return 1_000_000


if __name__ == '__main__':
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/battery_log_processed.csv")

    initial_soc = 0.655
    initial_Vt = 0
    initial_R_series = 0.05

    params = {
        "c": 2,
        "sigma1": 1,
        "sigma2": 1,
        "R_transient": 0.005,
        "tau_transient": 100,
        "V_nom": 3.7,
        "BAC": 0.1,
        "alpha1": 1, "beta1": 1, "kappa1": 0,
        "alpha2": 1, "beta2": 1, "kappa2": 0
    }

    # Initialize DUKF
    dukf = DUKF(
        initial1=np.array([initial_soc, initial_Vt]),
        initial2=np.array([initial_R_series]),
        func1=F1,
        func2=H1,
        func3=jac_H1,
        func4=F2,
        func5=H2,
        func6=jac_H2,
        Q1=1e-6,
        Q2=1e-6,
        Q3=1e-4,
        P1=1e-1,
        P2=1e-4,
        P3=1e-3,
        R1=0.1,
        R2=0.1,
        **params
    )

    df['Fuck2'] = df['Fuck2']/240

    current_series = df['Fuck2'].values
    voltage_series = df['Avg. Cell V[V]'].values

    predicted_states = []

    state_preds = []
    resistance_preds = []

    for k in range(len(current_series)):
    # for k in range(5):
        I = current_series[k]
        V_measured = np.array([voltage_series[k]])

        output1, output2 = dukf.step(control1=np.array([I]), control2=np.array([I]), y1=V_measured, y2=V_measured)

        soc_est, vt_est = output1
        R_series_est = output2[0]

        state_preds.append([soc_est, vt_est])
        resistance_preds.append(R_series_est)

    df_out = pd.DataFrame(state_preds, columns=["SoC", "V_transient"])
    df_out["R_series"] = resistance_preds
    df_out.to_csv("dual_ukf_predictions.csv", index=False)
    rms1 = root_mean_squared_error(df['Fuck3'], df_out['SoC'])
    print(rms1)
    # Load data
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/dual_ukf_predictions.csv")
    df2 = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/battery_log_processed.csv")

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create figure and axes
    fig, axes = plt.subplots(2, figsize=(20, 10))

    # First plot: R_series
    sns.lineplot(ax=axes[0], data=df, x=df.index, y='R_series', label='R_series', color='blue')

    # Second plot: SoC estimated and real
    sns.lineplot(ax=axes[1], data=df, x=df.index, y='SoC', label='SOC', color='green')
    sns.lineplot(ax=axes[1], data=df2, x=df2.index, y='Fuck3', label='SOC Real', color='red')

    # Add legend
    axes[0].legend()
    axes[1].legend()

    # Save figure
    fig.savefig('Fuuuuuuck5.png', dpi=450)
    plt.show()