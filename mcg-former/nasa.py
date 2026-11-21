"""NASA PCoE MATLAB loading and original per-cycle feature extraction."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy import interpolate


def moving_average(x, k=3):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < k or k <= 1:
        return x.copy()

    pad_left = k // 2
    pad_right = k - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode='edge')
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x_pad, kernel, mode='valid')


def compute_dynamic_eol_cycle(capacity_ah, threshold_ah, smooth_k=3, stable_k=2):
    cap = np.asarray(capacity_ah, dtype=np.float32)
    cap_smooth = moving_average(cap, k=smooth_k)

    n = len(cap_smooth)
    for i in range(n):
        if cap_smooth[i] <= threshold_ah:
            end_j = min(n, i + stable_k)
            if np.all(cap_smooth[i:end_j] <= threshold_ah):
                return i + 1  # 1-based cycle

    return n


def resample_cycle(voltage, current, temperature, target_len=100):
    n = len(voltage)
    if n <= 1:
        return np.zeros((target_len, 4), dtype=np.float32)

    t_orig = np.linspace(0, 1, n)
    t_target = np.linspace(0, 1, target_len)

    v = interpolate.interp1d(
        t_orig, voltage, kind='linear', fill_value='extrapolate'
    )(t_target)
    i = interpolate.interp1d(
        t_orig, current, kind='linear', fill_value='extrapolate'
    )(t_target)
    temp = interpolate.interp1d(
        t_orig, temperature, kind='linear', fill_value='extrapolate'
    )(t_target)

    q = np.cumsum(np.abs(i))
    dV = np.gradient(v) + 1e-6
    ic = np.clip(np.gradient(q) / dV, -50, 50)

    v = v / 4.2
    i = i / 2.0
    temp = temp / 60.0
    ic = ic / 50.0

    feat = np.stack([v, i, temp, ic], axis=1).astype(np.float32)
    return feat


def load_battery_data(file_path):
    mat = loadmat(file_path)
    file_key = Path(file_path).stem
    cycles = mat[file_key][0][0]['cycle'][0]

    data_dict = {"discharge": []}
    for cycle in cycles:
        if cycle['type'][0] == 'discharge':
            data = cycle['data'][0][0]
            data_dict["discharge"].append({
                "voltage": data['Voltage_measured'][0].flatten(),
                "current": data['Current_measured'][0].flatten(),
                "temperature": data['Temperature_measured'][0].flatten(),
                "capacity": float(data['Capacity'][0][0]),
            })
    return data_dict
