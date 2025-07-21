import csv
import os
import sys
import re
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def read_csv_to_dict(path):
    """
    Read a CSV file and return headers list and values (list of rows), skipping empty lines and commented lines.
    """
    headers = []
    values = []

    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Skip empty rows
            if len(row) == 0:
                continue

            # Skip rows starting with '#'
            if row[0].startswith('#'):
                continue

            # First non-comment row is assumed to be header
            if not headers:
                headers = row
                continue

            # Append data rows
            values.append(row)

    return headers, values


def build_column_dict(headers, values):
    """
    Build a dictionary mapping each header to a list of column values.
    """
    df = {header: [] for header in headers}
    for row in values:
        for i, header in enumerate(headers):
            df[header].append(row[i])
    return df


def partition_by_rack(df):
    """
    Partition the flat dictionary of column data into nested structure by rack number.
    Returns: (df_new, rack_count)
    - df_new: dict where keys are rack numbers (as strings) mapping to sub-dicts of their columns.
    - rack_count: total number of detected racks (int)
    """
    # Detect how many racks exist based on header prefixes
    rack_count = 0
    for i in range(0, 100):
        key_head = f"[Rack#{i + 1}]"
        keys = [k for k in df.keys() if k.startswith(key_head)]
        if not keys:
            rack_count = i
            break

    # Create nested structure
    df_new = {}
    for key in df.keys():
        if key.startswith('[Rack#'):
            # Extract rack number between '[Rack#' and ']'
            rack_num = key.split(']')[0][6:]
            if rack_num not in df_new:
                df_new[rack_num] = {}
            # Extract the sub-key (after '[Rack#N]')
            sub_key = key.split(f'[Rack#{rack_num}]')[1]
            df_new[rack_num][sub_key] = df[key]
        else:
            # Non-rack columns remain at top level
            df_new[key] = df[key]

    return df_new, rack_count


def compute_averages(df_new, rack_count, values_length):
    """
    Compute averaged metrics across all racks for each timestamp.
    - For numeric values, compute mean across racks.
    - For "Min" columns, take the minimum across racks.
    - For "Max" columns, take the maximum across racks.
    - For enumerated states ("Open", "Closed", etc.), map to numeric via enum_dict and then average.
    Returns: df_avg, a dict mapping each column name to its averaged time series list.
    """
    # Initialize df_avg with always-present counters
    df_avg = {}
    df_avg['Counter'] = df_new['Counter']
    df_avg['Time'] = df_new['Time']

    # Mapping for enumerated string states → numeric
    enum_dict = {
        'Open': 1,
        'Closed': 0,
        'On': 1,
        'Off': 0,
        'Idle': 0,
        'Discharging': -1,
        'Charging': 1,
    }

    # Iterate over each column name in rack #1 to discover common metrics
    for key in tqdm(df_new['1'].keys(), desc='Computing averages for keys'):
        # Attempt to parse the first value as float to check if numeric column
        first_val = df_new['1'][key][0]
        try:
            float(first_val)
        except ValueError:
            # Skip non-numeric and non-enumerated columns
            if first_val not in enum_dict:
                continue

        # Initialize averaged list for this key
        df_avg[key] = [0.0] * values_length

        # For each timestamp index i, aggregate across racks
        for i in range(values_length):
            agg_val = 0.0
            current_min = None
            current_max = None
            numeric_sum = 0.0

            # Sum or compare across each rack
            for rack in range(1, rack_count + 1):
                # Fetch string repr and convert enumerations if needed
                raw_val = df_new[str(rack)][key][i]
                if raw_val in enum_dict:
                    val = enum_dict[raw_val]
                else:
                    val = float(raw_val)

                # Handle Min/Max columns separately
                if 'Min' in key:
                    if (current_min is None) or (val < current_min):
                        current_min = val
                elif 'Max' in key:
                    if (current_max is None) or (val > current_max):
                        current_max = val
                else:
                    # Sum numeric values for averaging
                    numeric_sum += val

            # Assign aggregated result based on key type
            if 'Min' in key:
                df_avg[key][i] = current_min if (current_min is not None) else 0.0
            elif 'Max' in key:
                df_avg[key][i] = current_max if (current_max is not None) else 0.0
            else:
                # Compute mean across all racks
                df_avg[key][i] = round(numeric_sum / rack_count, 2)

    return df_avg


def normalize_and_smooth(df_avg):
    """
    Normalize voltage columns to [0,1] range and smooth voltage/current columns by
    removing outliers via Z-score > 3 and interpolating NaN values.
    Mutates df_avg by adding normalized columns and smoothing in-place.
    """
    # Identify all keys ending with '[V]' as voltage columns
    keys_voltages = [k for k in df_avg.keys() if k.endswith('[V]')]

    # Normalize each voltage column to [0,1]
    for key in keys_voltages:
        values = np.array(df_avg[key], dtype=float)
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        if max_val - min_val == 0:
            # Skip constant columns
            continue
        norm_key = key.replace('[V]', '_Normalized')
        df_avg[norm_key] = [round((v - min_val) / (max_val - min_val), 4) for v in values]

    # Identify keys (voltages and currents) to smooth
    keys_currents = [k for k in df_avg.keys() if k.endswith('[A]')]
    keys_to_smooth = keys_voltages + keys_currents

    # For each key in smoothing targets:
    for key in tqdm(keys_to_smooth, desc='Smoothing keys'):
        series = np.array(df_avg[key], dtype=float)

        # Compute Z-scores and mark outliers as NaN
        z_scores = stats.zscore(series, nan_policy='omit')
        outlier_mask = np.abs(z_scores) > 3
        series[outlier_mask] = np.nan

        # Use pandas Series for linear interpolation and fill at ends
        s = pd.Series(series)
        s_interp = s.interpolate(method='linear', limit_direction='both')
        s_filled = s_interp.fillna(method='bfill').fillna(method='ffill')

        # Assign smoothed values back
        df_avg[key] = s_filled.tolist()

    return df_avg


def save_to_csv(df_avg, output_path):
    """
    Save the dictionary-of-lists df_avg to a CSV file.
    Each key becomes a header, and each row is written in order.
    """
    keys = list(df_avg.keys())
    n_rows = len(df_avg[keys[0]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(keys)

        # Write each data row by index
        for i in range(n_rows):
            row = [df_avg[key][i] for key in keys]
            writer.writerow(row)

    print(f"Averaged data saved to {output_path}")


def datacleaner(in_directory_name, out_directory_name):
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    directory = os.fsencode(in_directory_name)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and re.search(".+Rack.+", filename):
            input = filename
            output = "processed_" + filename
            input = os.path.join(modpath, in_directory_name + "/" + input)
            output = os.path.join(modpath, out_directory_name + output)

            # Step 1: Read CSV file
            headers, values = read_csv_to_dict(input)
            print(f"Read {len(values)} rows (excluding header) from {output}")

            # Step 2: Build dictionary mapping each header to list of column values
            df = build_column_dict(headers, values)
            print(f"Loaded columns: {list(df.keys())[:5]} ... (+{len(df) - 5} more)")

            # Step 3: Partition columns by rack; detect how many racks exist
            df_new, rack_count = partition_by_rack(df)
            print(f"Detected {rack_count} racks in dataset.")

            # Step 4: Compute averages across racks
            df_avg = compute_averages(df_new, rack_count, len(values))
            print(f"Computed averaged metrics for {len(df_avg) - 2} keys (excluding 'Counter' & 'Time').")

            # Step 5: Normalize voltages and smooth signals
            df_avg = normalize_and_smooth(df_avg)
            print("Normalization and smoothing complete.")

            # Step 6: Save averaged result to output path
            save_to_csv(df_avg, output)


if __name__ == "__main__":
    df = pd.read_csv("/SOH/simulation_output.csv")
    df_out = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/dual_ukf_predictions.csv")
    fig, axs = plt.subplots(2, figsize=(90, 15))
    # axs[0].plot(df['Time [s]'], df_out['SOC'], label="SOC")
    # axs[0].plot(df['Time [s]'], (df['SOC[%]']), label="SOC Real")

    axs[0].plot(df['Time [s]'], df_out['Res'], label="Resistance")
    # axs[1].plot(df['Time [s]'], df_out['Capacity'], label="Capacity")
    axs[1].plot(df['Time [s]'], df['Total capacity lost to side reactions [A.h]'], label="Capacity")
    fig.legend()
    fig.savefig('Fuuuuuuck.png', dpi=300)
    fig.show()