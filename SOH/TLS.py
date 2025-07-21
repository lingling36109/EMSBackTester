import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/training/processed/battery_log_processed.csv")
df['SOC [%]'] = df['SOC[%]']
df['Current [A]'] = df['Rack Current[A]'] / 60
time_series = df['Counter']

# df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/simulation_output.csv")
# df['SOC [%]'] = 1.0 - (df['Discharge capacity [A.h]'] / (5.0 - df['Total capacity lost to side reactions [A.h]']))
# time_series = df['Time [s]']

SOC_series = df['SOC [%]']/100
current_series = df['Current [A]']


window_size = 10000
results = [{'slope': 3.5, 'time': 0, 'x_variance': 0, 'y_variance': 0, 'r_squared':0}]

for i in range(len(df) - window_size):
    X = []
    y = []
    y_prev = 0
    for j in range(i+1, i + window_size):
        delta_soc = SOC_series[j] - SOC_series[i]
        X.append(delta_soc)

        delta_t = time_series[j] - time_series[j - 1]
        sigma = 0
        y_prev += delta_t * current_series[j] * 0.995/3600
        y.append(y_prev)

    X_array = np.array(X).reshape(-1, 1)
    y_array = np.array(y)

    # if np.var(y_array) <= 7 * 1e-3 and np.var(X_array) <= 5 * 1e-6:
    #     continue

    model = LinearRegression(fit_intercept=False).fit(X_array, y_array)
    slope = model.coef_[0]

    if slope <= 0 or model.score(X_array, y_array) <= 0.9:
        results.append({
            'slope': results[-1]['slope'],
            'time': time_series[int((i + i + window_size) / 2)],
            'x_variance': np.var(X_array),
            'y_variance': np.var(y_array),
            'r_squared': model.score(X_array, y_array)
        })
        continue

    print(f"Run: {i}")
    print(f"Slope: {slope}")
    print(f"R² score: {model.score(X_array, y_array)}")

    results.append({
        'slope': slope,
        'time': time_series[int((i + i + window_size) / 2)],
        'x_variance': np.var(X_array),
        'y_variance': np.var(y_array),
        'r_squared': model.score(X_array, y_array)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    results_df = pd.read_csv("results.csv")
    results_df['TimeDelta'] = pd.to_timedelta(results_df['time'], unit='s')
    results_df.set_index('TimeDelta', inplace=True)
    results_df['slope_smooth'] = results_df['slope'].rolling('3600s').mean()

    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/simulation_output.csv")
    # df = df[df['Time [s]'] <= 20000]

    fig, axs = plt.subplots(3, figsize=(75, 15))
    axs[0].plot(
        results_df.index.total_seconds(),
        results_df['slope'],
        label="Raw Slope",
        color='tab:blue',
        linewidth=0.5,
        alpha=0.4
    )
    axs[0].plot(
        results_df.index.total_seconds(),
        results_df['slope_smooth'],
        label="Simple Moving Avg",
        color='tab:orange',
        linewidth=1.5
    )
    axs[1].plot(
        results_df.index.total_seconds(),
        results_df['x_variance'],
        label="Variance X",
        color='tab:purple',
        linewidth=1.5
    )
    axs[1].plot(
        results_df.index.total_seconds(),
        results_df['y_variance'],
        label="Variance Y",
        color='tab:pink',
        linewidth=1.5
    )
    axs[2].plot(
        results_df.index.total_seconds(),
        results_df['r_squared'],
        label="R Squared",
        color='tab:red',
        linewidth=1.5
    )
    axs[2].set_ylim(0, 1)
    # axs.plot(
    #     df['Time [s]'],
    #     5.0 - df['Total capacity lost to side reactions [A.h]'],
    #     label="Model's Capacity",
    #     color='tab:green',
    #     linewidth=1.0,
    #     alpha=0.6
    # )
    fig.tight_layout()
    fig.legend()
    plt.savefig('Smoothed_Capacity.png', dpi=300)
    plt.show()