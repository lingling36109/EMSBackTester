import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/simulation_output.csv")
# df['SOC [%]'] = 100 * (1.0 - (df['Discharge capacity [A.h]'] / (5.0 - df['Total capacity lost to side reactions [A.h]'])))
# # df['SOC [%]'] = df['SOC [%]'].clip(lower=0, upper=100)
# time_series = df['Time [s]']
# SOC_series = df['SOC [%]']
# current_series = df['Current [A]']

df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/training/processed/battery_log_processed.csv")
time_series = df['Counter']
SOC_series = df['SOC[%]']
current_series = df['Rack Current[A]'] / 480

Q_nominal = 5
sigma_current = 1e-8
var_y0 = 1e-8
alpha = 0.05

c1 = 1.0 / var_y0
c2 = Q_nominal / var_y0

capacity_estimates = [{'time_s': 0, 'estimated_capacity':  5}]

for i in range(1, len(df)):
    delta_SOC = SOC_series.iloc[i] - SOC_series.iloc[i - 1]
    delta_t = time_series.iloc[i] - time_series.iloc[i - 1]
    current = current_series.iloc[i]
    sigma = 1 if current > 0 else 0.85

    if current <= 0 or abs(delta_SOC) < 1e-2:
        continue

    x_i = delta_SOC / 100.0
    y_i = sigma * -current * delta_t/ 3600.0

    sigma_add = (sigma_current / 3600.0) * delta_t
    sigma_yi = np.sqrt(sigma_add ** 2 + (alpha * y_i) ** 2)
    var_yi = sigma_yi ** 2

    weight = 1.0 / (var_yi + 1e-12)
    c1 *= 0.99
    c2 *= 0.99
    c1 += (x_i ** 2) * weight
    c2 += x_i * y_i * weight

    Q_hat = c2 / c1
    if Q_hat > Q_nominal:
        Q_hat = Q_nominal

    capacity_estimates.append({
        'time_s': time_series.iloc[i],
        'estimated_capacity': Q_hat
    })

results_df = pd.DataFrame(capacity_estimates)
results_df['TimeDelta'] = pd.to_timedelta(results_df['time_s'], unit='s')
results_df.set_index('TimeDelta', inplace=True)
results_df['smoothed'] = results_df['estimated_capacity'].rolling('3600s').mean()

results_df.to_csv("WLS_capacity_estimate.csv")

if __name__ == "__main__":
    fig, axs = plt.subplots(1, figsize=(75, 15))

    axs.plot(
        results_df.index.total_seconds(),
        results_df['estimated_capacity'],
        label="Raw Capacity Estimate",
        color='tab:blue',
        linewidth=0.5,
        alpha=0.5
    )
    axs.plot(
        results_df.index.total_seconds(),
        results_df['smoothed'],
        label="Smoothed (1hr Rolling Avg)",
        color='tab:orange',
        linewidth=2.0
    )
    # axs.plot(
    #     df['Time [s]'],
    #     (5.0 - df['Total capacity lost to side reactions [A.h]']),
    #     label="Real Deal",
    #     color='tab:pink',
    #     linewidth=2.0
    # )
    axs.set_title("Weighted Least Squares Capacity Estimate (Recursive)")
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Capacity Estimate [Ah]")
    axs.grid(True)
    fig.legend()
    plt.savefig("WLS_Capacity.png", dpi=300)
    plt.show()
