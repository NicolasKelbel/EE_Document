import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Parameters
number_of_faces_of_bluff_body = 9
cutoff = 1.0  # Seconds
prominence_threshold = 0.3

# Load C_L data 
filename = rf"..."
time, Cl = [], []

os.chdir(r"...")

with open(filename, "r") as f:
    for line in f:
        if line.strip().startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 5:
            time.append(float(parts[0]))
            Cl.append(float(parts[4]))

time = np.array(time)
Cl = np.array(Cl)

# Remove startup phase
mask = time > cutoff
time = time[mask]
Cl = Cl[mask]

# Normalize C_L
Cl_norm = Cl - np.mean(Cl)
Cl_norm = Cl_norm / np.max(np.abs(Cl_norm))

# Detect peaks
peak_indices, _ = find_peaks(Cl_norm, prominence=prominence_threshold)

# Find troughs between each pair of peaks
trough_indices = []
for i in range(len(peak_indices) - 1):
    left = peak_indices[i]
    right = peak_indices[i + 1]
    if right > left + 1:
        trough_region = Cl_norm[left:right+1]
        trough_local_index = np.argmin(trough_region)
        trough_global_index = left + trough_local_index
        trough_indices.append(trough_global_index)

trough_indices = np.array(trough_indices)

# Calculate average period from peaks and troughs separately
peak_times = time[peak_indices]
trough_times = time[trough_indices]

# Only compute periods if enough points exist
if len(peak_times) >= 2:
    peak_periods = np.diff(peak_times)
else:
    peak_periods = np.array([])

if len(trough_times) >= 2:
    trough_periods = np.diff(trough_times)
else:
    trough_periods = np.array([])

# Combine both for average period
all_periods = np.concatenate([peak_periods, trough_periods])
if len(all_periods) >= 1:
    avg_full_period = np.mean(all_periods)
else:
    avg_full_period = np.nan

# Print results 
print(f"Number of peaks:    {len(peak_indices)}")
print(f"Number of troughs:  {len(trough_indices)}")
print(f"\nAverage full period: {avg_full_period:.5f} s")

