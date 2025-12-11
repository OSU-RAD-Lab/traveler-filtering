import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt


BUFFER_SIZE = 3
MIN_STEP = 0.01
NOTCH_COUNT = 756
KV = 85
MAX_ALLOWANCE = 0.05


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def noise_subtraction(t, tau, theta):
    slope_sign = np.gradient(savgol_filter(theta, window_length=700, polyorder=3), t) > 0
    smooth_theta_a = savgol_filter(theta, window_length=500, polyorder=3)
    if PHASE_FLIP:
        diffs = np.array([(theta[i] - smooth_theta_a[i]) for i in range(len(theta))])
    else:
        diffs = np.array([(smooth_theta_a[i] - theta[i]) for i in range(len(theta))])

    # Find crossings based on slope_sign direction
    below_to_above = (theta[:-1] < smooth_theta_a[:-1]) & (theta[1:] >= smooth_theta_a[1:])
    above_to_below = (theta[:-1] > smooth_theta_a[:-1]) & (theta[1:] <= smooth_theta_a[1:])
    crossings = np.where(slope_sign[1:], below_to_above, above_to_below)
    crossing_indices = np.where(crossings)[0] + 1

    # Calculate range of each zone for tau_a and diffs
    zone_boundaries = np.concatenate([[0], crossing_indices, [len(theta)]])
    scaled_diffs = np.copy(diffs)
    print("Zone ranges:")
    for i in range(len(zone_boundaries) - 1):
        start = zone_boundaries[i]
        end = zone_boundaries[i + 1]
        tau_zone = tau[start:end]
        diffs_zone = diffs[start:end]
        tau_range = np.percentile(tau_zone, 95) - np.percentile(tau_zone, 5)
        diffs_range = np.max(diffs_zone) - np.min(diffs_zone)
        
        # Scale diffs by ratio of diffs_range / tau_range
        if tau_range > 0:
            ratio = tau_range / diffs_range
            scaled_diffs[start:end] = diffs[start:end] * ratio
    
    # subtract motor noise from tau
    denoised = tau - scaled_diffs

    return denoised, scaled_diffs

def endpoints(t, tau, theta):
    slope_sign = np.gradient(savgol_filter(theta, window_length=700, polyorder=3), t) > 0
    smooth_theta_a = savgol_filter(theta, window_length=500, polyorder=3)
    if PHASE_FLIP:
        diffs = np.array([(theta[i] - smooth_theta_a[i]) for i in range(len(theta))])
    else:
        diffs = np.array([(smooth_theta_a[i] - theta[i]) for i in range(len(theta))])

    # Find crossings based on slope_sign direction
    below_to_above = (theta[:-1] < smooth_theta_a[:-1]) & (theta[1:] >= smooth_theta_a[1:])
    above_to_below = (theta[:-1] > smooth_theta_a[:-1]) & (theta[1:] <= smooth_theta_a[1:])
    crossings = np.where(slope_sign[1:], below_to_above, above_to_below)
    crossing_indices = np.where(crossings)[0] + 1

    # Create zone boundaries
    zone_boundaries = np.concatenate([[0], crossing_indices, [len(theta) - 1]])
    
    # Create straight lines between start and end values of each zone (contiguous)
    endpoint_line = np.zeros_like(diffs)
    for i in range(len(zone_boundaries) - 1):
        start = zone_boundaries[i]
        end = zone_boundaries[i + 1]
        start_val = tau[start]
        end_val = tau[end]
        endpoint_line[start:end + 1] = np.linspace(start_val, end_val, end - start + 1)
    
    return endpoint_line

# data = np.loadtxt('data/raw/horizontal_in.csv', delimiter=',', encoding='utf-8-sig', skiprows=4)
# data = np.loadtxt('data/raw/horizontal_out.csv', delimiter=',', encoding='utf-8-sig', skiprows=4)
# data = np.loadtxt('data/raw/diag_out.csv', delimiter=',', encoding='utf-8-sig', skiprows=4)
# data = np.loadtxt('data/10cmshearnoleg.csv', delimiter=',', encoding='utf-8-sig', skiprows=4)
data = np.loadtxt('data/nolegpentration.csv', delimiter=',', encoding='utf-8-sig', skiprows=4)
PHASE_FLIP = True
START = 3000
END = -7500

START = 3000
END = -7000

t = data[START:END, 0]
theta_a = data[START:END, 9]
theta_b = data[START:END, 10]
tau_a = data[START:END, 11]
tau_b = data[START:END, 12]

denoised, noise = noise_subtraction(t, tau_a, theta_a)
# denoised = endpoints(t, tau_a, theta_a)

plt.plot(t[800:], tau_a[800:], label='raw torque')
plt.plot(t[800:], denoised[800:], label='denoised torque')
# plt.plot(t[800:], noise[800:], label='motor noise')

# t = data[START:END, 0]
# theta_a = data[START:END, 9]
# theta_b = data[START:END, 10]
# tau_a = data[START:END, 11]
# tau_b = data[START:END, 12]

# # denoised, noise = noise_subtraction(t, tau_a, theta_a)
# # # denoised = endpoints(t, tau_a, theta_a)

# plt.plot(t, tau_a, label='torque')
# plt.plot(t, theta_a, label='angle')
# # plt.plot(t[800:], denoised[800:], label='denoised torque')
# # plt.plot(t[800:], noise[800:], label='motor noise')


plt.plot()
plt.legend()
plt.grid(True)
plt.show()