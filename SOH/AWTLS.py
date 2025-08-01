import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIGMA_CURRENT = 0.1
ALPHA = 0.05
FORGETTING_FACTOR = 0.99
SOC_NOISE_SCALE = 0.01
CLIP_SOC_DELTA = 0.2
CLIP_AH = 0.5


def compute_sigma(x_i, y_i, delta_t):
    sigma_yi = np.sqrt((SIGMA_CURRENT * delta_t)**2 + (ALPHA * y_i)**2)
    sigma_xi = SOC_NOISE_SCALE * abs(x_i) + 1e-6
    return sigma_xi, sigma_yi


def compute_jacobian(Q, x, y, sigma_x, sigma_y):
    Qx_minus_y = Q * x - y
    num = 2 * Qx_minus_y * (Q * y * sigma_x**2 + x * sigma_y**2)
    denom = np.maximum((Q**2 * sigma_x**2 + sigma_y**2)**2, 1e-12)
    return np.sum(num / denom)


def compute_hessian(Q, x, y, sigma_x, sigma_y):
    sx2 = sigma_x**2
    sy2 = sigma_y**2
    Q2 = Q**2
    Qx_minus_y = Q * x - y
    denom = np.maximum((Q2 * sx2 + sy2)**3, 1e-12)

    term1 = 4 * Qx_minus_y * (y * sx2 + Q * x * sy2)
    term2 = 2 * (Q * y * sx2 + x * sy2)**2
    numerator = term1 + term2
    return np.sum(numerator / denom) + 1e-8

def newton_wtls_solver(x, y, sigma_x, sigma_y, Q_init=5.0, max_iter=10, tol=1e-6):
    Q = Q_init
    for _ in range(max_iter):
        jac = compute_jacobian(Q, x, y, sigma_x, sigma_y)
        hess = compute_hessian(Q, x, y, sigma_x, sigma_y)
        if abs(hess) < 1e-12:
            print("Warning: Hessian near zero — numerical instability")
            break
        Q_new = Q - jac / hess
        if abs(Q_new - Q) < tol:
            break
        Q = Q_new
    return Q, hess


def main():
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/training/processed/battery_log_processed.csv")
    time = df['Counter']
    soc = df['SOC[%]'] / 100
    current = df['Rack Current[A]'] / -240

    # df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/simulation_output.csv")
    # df['SOC [%]'] = 100 * (
    #             1.0 - (df['Discharge capacity [A.h]'] / (5.0 - df['Total capacity lost to side reactions [A.h]'])))
    # time = df['Time [s]']
    # soc = df['SOC [%]']
    # current = df['Current [A]']

    x_vals, y_vals = [], []
    sigma_x_vals, sigma_y_vals = [], []
    time_series, Q_estimates, Q_sigma_bounds = [], [], []

    for i in range(1, len(df)):
        dt = time[i] - time[i - 1]
        delta_soc = soc[i] - soc[i - 1]
        if current[i] >= 0 or abs(delta_soc) < 1e-4:
            continue

        # Outlier rejection
        x_i = np.clip(delta_soc, -CLIP_SOC_DELTA, CLIP_SOC_DELTA)
        y_i = np.clip(-current[i] * dt, -CLIP_AH, CLIP_AH)

        sigma_xi, sigma_yi = compute_sigma(x_i, y_i, dt)

        # Apply fading memory
        x_vals = [FORGETTING_FACTOR * xj for xj in x_vals] + [x_i]
        y_vals = [FORGETTING_FACTOR * yj for yj in y_vals] + [y_i]
        sigma_x_vals = [FORGETTING_FACTOR * sj for sj in sigma_x_vals] + [sigma_xi]
        sigma_y_vals = [FORGETTING_FACTOR * sj for sj in sigma_y_vals] + [sigma_yi]

        time_series.append(time[i])

        x_np = np.array(x_vals)
        y_np = np.array(y_vals)
        sx_np = np.array(sigma_x_vals)
        sy_np = np.array(sigma_y_vals)

        Q_init = np.sum(x_np * y_np / sy_np**2) / np.sum(x_np**2 / sy_np**2)

        Q_hat, hess = newton_wtls_solver(x_np, y_np, sx_np, sy_np, Q_init)
        Q_estimates.append(Q_hat)

        print(hess)

        sigma_Q = np.sqrt(1.0 / hess) if hess > 0 else np.nan
        Q_sigma_bounds.append(sigma_Q)

    results = pd.DataFrame({
        'Time [s]': time_series,
        'Estimated Capacity [Ah]': Q_estimates,
        'Sigma Capacity': Q_sigma_bounds
    })
    results['Upper Bound'] = results['Estimated Capacity [Ah]'] + 3 * results['Sigma Capacity']
    results['Lower Bound'] = results['Estimated Capacity [Ah]'] - 3 * results['Sigma Capacity']
    results['TimeDelta'] = pd.to_timedelta(results['Time [s]'], unit='s')
    results.set_index('TimeDelta', inplace=True)
    results['Smoothed'] = results['Estimated Capacity [Ah]'].rolling('3600s').mean()
    results.to_csv("approx_wtls_results_with_bounds.csv")

    # === Plot Results ===
    plt.figure(figsize=(20, 6))
    time_sec = results.index.total_seconds()

    # Confidence bounds first (underlay)
    plt.fill_between(
        time_sec,
        results['Lower Bound'],
        results['Upper Bound'],
        color='gray',
        alpha=0.3,
        label='±3σ Confidence Interval',
        zorder=1
    )

    plt.plot(
        time_sec,
        results['Estimated Capacity [Ah]'],
        label='WTLS Estimate',
        alpha=0.6,
        color='tab:blue',
        linewidth=1.0,
        zorder=2
    )

    plt.plot(
        time_sec,
        results['Smoothed'],
        label='Smoothed (1hr)',
        linewidth=2.0,
        color='tab:orange',
        zorder=3
    )

    plt.title("AWTLS Capacity Estimation with Confidence Bounds")
    plt.xlabel("Time [s]")
    plt.ylabel("Estimated Capacity [Ah]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("approx_wtls_capacity_bounds.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
