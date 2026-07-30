import os
import random
import numpy as np
import torch
from scipy.io import loadmat
from scipy import interpolate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    min_len = min(len(y_true), len(y_pred))
    if min_len == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2)))


def calculate_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    min_len = min(len(y_true), len(y_pred))
    if min_len == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[:min_len] - y_pred[:min_len])))


def calculate_ae(true_rul, pred_rul):
    return int(abs(int(true_rul) - int(pred_rul)))


def get_battery_eol_capacity(battery_name, config):
    if battery_name in config.EOL_CAPACITY_MAP:
        return float(config.EOL_CAPACITY_MAP[battery_name])
    return 1.40


def get_battery_eol_soh(battery_name, config):
    return float(get_battery_eol_capacity(battery_name, config) / config.INITIAL_CAPACITY)


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
    file_key = os.path.splitext(os.path.basename(file_path))[0]
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


def extract_cycles_and_soh(battery_data, config, battery_name=None):
    cycles, soh_list, capacity_list = [], [], []

    for discharge in battery_data["discharge"]:
        cycle = resample_cycle(
            discharge["voltage"],
            discharge["current"],
            discharge["temperature"],
            config.CYCLE_LEN
        )
        cap = float(discharge["capacity"])
        cycles.append(cycle)
        capacity_list.append(cap)
        soh_list.append(cap / config.INITIAL_CAPACITY)

    cycles = np.array(cycles, dtype=np.float32)
    soh = np.array(soh_list, dtype=np.float32)
    capacity_ah = np.array(capacity_list, dtype=np.float32)

    eol_cycle = len(capacity_ah)
    if battery_name is not None:
        threshold_ah = get_battery_eol_capacity(battery_name, config)
        eol_cycle = compute_dynamic_eol_cycle(
            capacity_ah=capacity_ah,
            threshold_ah=threshold_ah,
            smooth_k=config.EOL_SMOOTH_K,
            stable_k=config.EOL_STABLE_K,
        )
        cycles = cycles[:eol_cycle]
        soh = soh[:eol_cycle]
        capacity_ah = capacity_ah[:eol_cycle]

    return cycles, soh, capacity_ah, eol_cycle


def get_battery_data(data_dir, battery_name, config):
    battery_data = load_battery_data(os.path.join(data_dir, f"{battery_name}.mat"))
    cycles, soh, capacity_ah, eol_cycle = extract_cycles_and_soh(
        battery_data, config, battery_name=battery_name
    )
    return cycles, soh, eol_cycle


def build_handcrafted_features(history_soh):
    history_soh = np.asarray(history_soh, dtype=np.float32)
    recent_5 = history_soh[-5:] if len(history_soh) >= 5 else history_soh
    recent_10 = history_soh[-10:] if len(history_soh) >= 10 else history_soh

    return np.array([
        history_soh[-1],
        (recent_5[-1] - recent_5[0]) / max(1, len(recent_5) - 1),
        (recent_10[-1] - recent_10[0]) / max(1, len(recent_10) - 1),
        recent_5[0] - recent_5[-1],
        recent_10[0] - recent_10[-1],
        float(np.std(recent_5)),
        float(np.std(recent_10)),
    ], dtype=np.float32)


def left_pad_window(cycles, soh, end_idx, win_len):
    if end_idx >= win_len:
        hist_cycles = cycles[end_idx - win_len:end_idx]
        hist_soh = soh[end_idx - win_len:end_idx]
    else:
        pad_len = win_len - end_idx
        cycle_pad = np.repeat(cycles[:1], pad_len, axis=0)
        soh_pad = np.repeat(soh[:1], pad_len, axis=0)
        hist_cycles = np.concatenate([cycle_pad, cycles[:end_idx]], axis=0)
        hist_soh = np.concatenate([soh_pad, soh[:end_idx]], axis=0)

    return hist_cycles.astype(np.float32), hist_soh.astype(np.float32)


def build_train_sample(cycles, soh, idx, config):
    hist_cycles, hist_soh = left_pad_window(cycles, soh, idx, config.TRAIN_WINDOW_L)

    future = soh[idx: idx + config.WINDOW_H]
    valid_h = min(config.WINDOW_H, max(0, len(soh) - idx))

    if len(future) < config.WINDOW_H:
        pad_val = future[-1] if len(future) > 0 else soh[-1]
        future = np.concatenate(
            [future, np.full(config.WINDOW_H - len(future), pad_val, dtype=np.float32)],
            axis=0
        )

    future = future.astype(np.float32)
    traj_mask = np.zeros(config.WINDOW_H, dtype=np.float32)
    traj_mask[:valid_h] = 1.0

    true_rul = max(1, len(soh) - idx)
    handcrafted = build_handcrafted_features(hist_soh)

    return {
        "hist_cycles": hist_cycles,
        "hist_soh": hist_soh,
        "future_soh": future,
        "future_mask": traj_mask,
        "rul": np.array([true_rul], dtype=np.float32),
        "handcrafted": handcrafted,
    }


def build_test_window(cycles, soh, start_idx, config):
    hist_cycles, hist_soh = left_pad_window(cycles, soh, start_idx, config.TRAIN_WINDOW_L)
    handcrafted = build_handcrafted_features(hist_soh)
    true_future = soh[start_idx:].astype(np.float32)
    true_rul = int(len(soh) - start_idx)
    return hist_cycles, hist_soh, handcrafted, true_future, true_rul
