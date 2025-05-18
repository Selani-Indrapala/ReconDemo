import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

import numpy as np
import pandas as pd
import mpmath
import math
import time as t

from scipy.signal import firwin, lfilter, freqz, filtfilt

from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def SB(bws, n, t):
    if n == 0 and t == 0:
        return 1
    elif n > 0 and t == 0:
        return 0
    else:
        return np.real(np.sign(bws * t) * np.emath.sqrt((2 * n + 1) / (2 * bws * t)) * mpmath.besselj(n+0.5, bws * np.pi * t)) 
 

# Your existing functions
def RemoveSamples(signal, time, remove_percentage):
    np.random.seed(42)
    total_points = len(signal)
    num_to_remove = int(total_points * remove_percentage)
    keep_indices = np.sort(np.random.choice(total_points, total_points - num_to_remove, replace=False))
    return signal[keep_indices], time[keep_indices]

def concatenate_signals_with_overlap(signals, times):
    concatenated_signal = np.array([])
    final_times = np.array([])
    
    for i, signal in enumerate(signals):
        signal = signal.flatten()
        current_times = times[i].flatten()
        if i == 0:
            concatenated_signal = signal
            final_times = current_times
        else:
            previous_times = final_times
            common_times = np.intersect1d(previous_times, current_times)
            if len(common_times) > 0:
                prev_indices = np.searchsorted(previous_times, common_times)
                curr_indices = np.searchsorted(current_times, common_times)
                concatenated_signal[prev_indices] = (
                    concatenated_signal[prev_indices] + signal[curr_indices]
                ) / 2
                non_overlap_indices = np.where(current_times > previous_times[-1])[0]
                if len(non_overlap_indices) > 0:
                    concatenated_signal = np.concatenate([concatenated_signal, signal[non_overlap_indices]])
                    final_times = np.concatenate([final_times, current_times[non_overlap_indices]])
            else:
                concatenated_signal = np.concatenate((concatenated_signal, signal))
                final_times = np.concatenate((final_times, current_times))
    return concatenated_signal, final_times

def get_samples_at_timestamps(time_array, signal_array, target_timestamps):
    indices = np.where(np.isin(time_array, target_timestamps))[0]
    return signal_array[indices]

num_t =501
DEG = 100
legendre_B = np.zeros((DEG+1, num_t))
bws = 0.2

for i in range(num_t):
    for deg in range(DEG+1):
        legendre_B[deg, i] =SB(bws, deg, i-num_t//2)

    
def reconstruct_signal_using_windows(full_signal,full_time, full_range_array, window_size, recon_size, overlap_size):
    global num_t
    
    regula = 14.5
    mu = 10 ** -regula  # regularization factor for least squares
    unit_time = 1
    DEG = 100
    
    app_signal_windows = []
    app_time_windows = []
    
    pos = full_time[0]
    start_index = 0
    start_index_range = 0
    step_size = window_size - overlap_size
    
    total_duration = full_time[-1] - full_time[0]  # Calculate the total time duration of the signal
    step_size = (window_size - overlap_size)  # Calculate step size based on window size and overlap
    number_of_windows = int(total_duration / (step_size * unit_time))  # Find the number of windows
    
    window = 0
    
    # print("number of windows is", number_of_windows)
    total_time =0
    while pos + window_size*unit_time <= full_time[-1]:
        
        window+=1
        
        start_time = t.time()
        
        ##Getting the window to reconstruct
        end_index = start_index
        while full_time[end_index] <= pos + window_size*unit_time :
            
            end_index+=1
        if start_index == end_index:
            end_index = len(full_signal)
            
        start_index_range = 0
        while full_range_array[start_index_range] <= pos + (window_size//2-recon_size//2)*unit_time :
            start_index_range +=1
            
        signal= full_signal[start_index:end_index]
        time = full_time[start_index:end_index]
        
        #Centering the time around zero
        sub = time[len(time)//2]
        time = time - sub
        
        #Getting the time window to reconstruct
        new_start_index = start_index
        while full_time[new_start_index] <= pos + step_size*unit_time :
            new_start_index+=1
        start_index = new_start_index-1
    
        lh = len(signal)
        
        #Input matrix
        A = np.zeros((lh, DEG + 1))
        for k in range(DEG + 1):
            for j in range(lh):
                A[j, k] = legendre_B[k, time[j]+int(num_t//2)]

        #LSF
        CDs = np.linalg.solve(np.dot(A.T, A) + mu * np.eye(DEG + 1), np.dot(A.T, signal))

        
        
        range_array = []
        

        #Getting the next startpoint for the window
        next_time = start_index_range
    
        while full_range_array[next_time]<= (pos + (window_size//2+recon_size//2 )*unit_time) :
            next_time+=1

        ##Getting the window to reconstruct
        range_array = full_range_array[start_index_range: next_time]
        original_range_array = range_array
        range_array = range_array - sub
        original_range_array = original_range_array[(range_array>time[0]) & (range_array<time[-1])]
        range_array = range_array[(range_array>time[0]) & (range_array<time[-1])] ##To remove points outside the known range
        
        ##Calculating reconstruction matrix
        N = len(range_array)//2
        if (len(range_array)%2):
            B = np.zeros((DEG + 1, 2 * N+1 ))
        else: 
            B = np.zeros((DEG + 1, 2 * N ))
        for k in range(DEG + 1):
            for j in range(-N, N+1):
                if (j==N and len(range_array)%2==0):
                    break
                B[k, N + j] = legendre_B[k, range_array[N+j] +int(num_t//2)]

        app = np.dot(CDs, B) ##Calculating the approximation
        app = app.reshape(-1,1)
        app_signal_windows.append(app.reshape(1, -1))
        app_time_windows.append(original_range_array.reshape(1, -1))
        
        pos = pos + step_size*unit_time
        end_time = t.time()
        total_time+= (end_time-start_time)
        # print("amount of time for window", window,"is", end_time-start_time)
        
    # print("number of windows is :",len(app_signal_windows))
    # print("total time taken is", total_time)
    return app_signal_windows, app_time_windows

def calculate_metrics(real, approx):
    rmse = np.sqrt(np.mean((real - approx) ** 2))
    mae = np.mean(np.abs(real - approx))
    r_squared = 1 - (np.sum((real - approx) ** 2) / np.sum((real - np.mean(real)) ** 2))
    return rmse, mae, r_squared

# === Streamlit App ===
st.title("Signal Reconstruction Demo")

# User Inputs
remove_percentage = st.slider("Percentage of Data to Remove", min_value=0.1, max_value=0.7, step=0.1)

# Load data
save_path = 'SignalSegments\Signal03.npz'
data = np.load(save_path)
temperature_array = data["filtered"]
temperature_array = temperature_array[500:600]
Time = np.arange(0, len(temperature_array))
# original = data["original"]

# Parameters
nyint = 1
bws = 1  # if not defined earlier
window_size = 50 / bws
recon_size = 0.25 * window_size
overlap_size = 0.8 * window_size

# Sample removal
temp_arr, time = RemoveSamples(temperature_array, Time, remove_percentage)

# Dummy reconstruction for UI (replace with your real method)
# from your_reconstruction_module import reconstruct_signal_using_windows  # replace with actual import
range_array = Time
app_signal_windows, app_time_windows = reconstruct_signal_using_windows(temp_arr, time, range_array, window_size, recon_size, overlap_size)
app_signal, app_time = concatenate_signals_with_overlap(app_signal_windows, app_time_windows)

# Missing timestamps
missing_time_stamps = np.setdiff1d(Time, time)
missing_time_stamps = missing_time_stamps[(missing_time_stamps >= app_time[0]) & (missing_time_stamps <= app_time[-1])]
real_arr = get_samples_at_timestamps(Time, temperature_array, missing_time_stamps)

# Methods
methods = {
    "CDs": app_signal,
    "Linear": interp1d(time, temp_arr, kind='linear', fill_value='extrapolate')(app_time),
    "Spline": interp1d(time, temp_arr, kind='cubic', fill_value='extrapolate')(app_time),
    "Nearest Neighbour": interp1d(time, temp_arr, kind='nearest', fill_value=np.nan, bounds_error=False)(app_time),
}

placeholder = st.empty() 

fig, ax = plt.subplots()
ax.plot(Time, temperature_array, label="Original", color='orange', alpha=0.6)
ax.scatter(missing_time_stamps, real_arr, color='blue', label='Missing')
ax.set_xlabel("Time")
ax.set_ylabel("Temperature")
ax.grid(True)
ax.set_xlim(20,75)
ax.set_ylim(30.3, 31.2)
ax.legend()

# Prepare scatter plot for missing points, initially empty
# missing_scatter = ax.scatter([], [], color='blue', label='Missing')
recon_line_linear, = ax.plot([], [], '-*', label="Linear", color='green')
recon_line_spline, = ax.plot([], [], '-*', label="Spline", color='purple')
recon_line_nearest, = ax.plot([], [], '-*', label="Nearest Neighbour", color='blue')
recon_line_CD, = ax.plot([], [], '-*', label="CDs", color='red')

# st.pyplot(fig)

# Animate by progressively revealing reconstructed signal and missing points
# if st.button("▶ Play Reconstruction"):
for i in range(1, len(app_time) + 1):
        recon_line_linear.set_data(app_time[:i], methods["Linear"][:i])
        recon_line_spline.set_data(app_time[:i], methods["Spline"][:i])
        recon_line_nearest.set_data(app_time[:i], methods["Nearest Neighbour"][:i])
        recon_line_CD.set_data(app_time[:i], app_signal[:i])

        # current_missing_mask = missing_time_stamps <= app_time[i - 1]
        # missing_scatter.set_offsets(np.c_[missing_time_stamps[current_missing_mask], real_arr[current_missing_mask]])

        ax.legend()

        placeholder.pyplot(fig)  # This updates the same plot area
        t.sleep(0.005)  # Pause for animation effect

# Optional: keep final plot after loop ends
placeholder.pyplot(fig)

# Plotting
fig, ax = plt.subplots()
ax.scatter(missing_time_stamps, real_arr, color='blue', label='Missing')
ax.plot(app_time, methods["Linear"], '*', label="Linear")
ax.plot(app_time, methods["Spline"], '*', label="Spline", color='purple')
ax.plot(app_time, methods["Nearest Neighbour"], '*', label="Nearest Neighbour", color='blue')
ax.plot(app_time, app_signal, '*', label="CDs", color='red')
ax.plot(Time, temperature_array, '*', label="Original", color='orange', alpha=0.6)
ax.set_xlabel("Time")
ax.set_ylabel("Temperature")
ax.legend()
ax.set_xlim(40, 60)
CD_avg = np.mean(app_signal)
Lin_avg = np.mean(methods["Linear"])
Spline_avg = np.mean(methods["Spline"])
Nearest_avg = np.mean(methods["Nearest Neighbour"])
max_avg = max(CD_avg, Lin_avg, Spline_avg, Nearest_avg)
min_avg = min(CD_avg, Lin_avg, Spline_avg, Nearest_avg)
ax.set_ylim(max_avg - 0.3, max_avg + 0.3)
ax.grid(True)
st.pyplot(fig)

# Display metrics
st.subheader("Reconstruction Metrics")
for method, y in methods.items():
    app_arr = get_samples_at_timestamps(app_time, y, missing_time_stamps)
    rmse, mae, r2 = calculate_metrics(real_arr, app_arr)
    st.write(f"**{method}** - RMSE: {rmse:.2e}, MAE: {mae:.2e}, R²: {r2:.2e}")