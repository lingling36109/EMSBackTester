import os
import argparse
import datetime
import pandas as pd
from scipy.io import loadmat

def load_data(battery_path: str):
    mat = loadmat(battery_path + ".mat")
    battery = battery_path.split("/")[-1]
    print("Total data in dataset: ", len(mat[battery][0, 0]["cycle"][0]))
    counter = 0
    dataset = []

    capacity_data = []

    for i in range(len(mat[battery][0, 0]["cycle"][0])):
        row = mat[battery][0, 0]["cycle"][0, i]
        if row["type"][0] == "discharge":
            ambient_temperature = row["ambient_temperature"][0][0]
            date_time = datetime.datetime(
                int(row["time"][0][0]),
                int(row["time"][0][1]),
                int(row["time"][0][2]),
                int(row["time"][0][3]),
                int(row["time"][0][4]),
            ) + datetime.timedelta(seconds=int(row["time"][0][5]))
            data = row["data"]
            capacity = data[0][0]["Capacity"][0][0]
            for j in range(len(data[0][0]["Voltage_measured"][0])):
                voltage_measured = data[0][0]["Voltage_measured"][0][j]
                current_measured = data[0][0]["Current_measured"][0][j]
                temperature_measured = data[0][0]["Temperature_measured"][0][j]
                current_load = data[0][0]["Current_load"][0][j]
                voltage_load = data[0][0]["Voltage_load"][0][j]
                time = data[0][0]["Time"][0][j]
                dataset.append(
                    [
                        counter + 1,
                        ambient_temperature,
                        date_time,
                        capacity,
                        voltage_measured,
                        current_measured,
                        temperature_measured,
                        current_load,
                        voltage_load,
                        time,
                    ]
                )
            capacity_data.append(
                [counter + 1, ambient_temperature, date_time, capacity]
            )
            counter = counter + 1

    max_capacity = max([row[3] for row in dataset])
    return [
        pd.DataFrame(
            data=dataset,
            columns=[
                "cycle",
                "ambient_temperature",
                "datetime",
                "capacity",
                "voltage_measured",
                "current_measured",
                "temperature_measured",
                "current_load",
                "voltage_load",
                "time",
            ],
        ),
        pd.DataFrame(
            data=capacity_data,
            columns=["cycle", "ambient_temperature", "datetime", "capacity"],
        ),
    ], max_capacity


if __name__ == "__main__":
    mat_file_path = "/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/B0005"  # Without .mat extension
    output_dir = "/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/output.csv"
    battery_name = mat_file_path.split("/")[-1]

    # Load data from .mat
    (battery_df, capacity_df), max_capacity = load_data(mat_file_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # File paths for CSVs
    full_csv_path = f"{output_dir}/{battery_name}_full_data.csv"
    capacity_csv_path = f"{output_dir}/{battery_name}_capacity.csv"

    # Save as CSV
    battery_df.to_csv(full_csv_path, index=False)
    capacity_df.to_csv(capacity_csv_path, index=False)

    print(f"CSV saved: {full_csv_path}")
    print(f"CSV saved: {capacity_csv_path}")
