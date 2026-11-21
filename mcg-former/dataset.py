"""Battery records and shared MCG-Former training windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Battery:
    name: str
    cycles: np.ndarray
    soh: np.ndarray
    eol_cycle: int


def load_nasa_arrays(path, cfg):
    from .nasa import compute_dynamic_eol_cycle, load_battery_data, resample_cycle

    discharges = load_battery_data(path)["discharge"]
    if not discharges:
        raise ValueError(f"{path}: no discharge cycles were found.")
    capacity = np.asarray([cycle["capacity"] for cycle in discharges], dtype=np.float64)
    threshold = 1.44 if path.stem == "B0007" else 1.40
    # The moving average defines the EOL index only. The SOH uses raw capacity.
    eol = compute_dynamic_eol_cycle(capacity, threshold)
    features = np.stack([
        resample_cycle(cycle["voltage"], cycle["current"], cycle["temperature"], cfg.CYCLE_LEN)
        for cycle in discharges[:eol]
    ])
    return features, capacity[:eol], eol


def load_calce_arrays(path):
    from .prepare_calce import eol_cycle

    with np.load(path, allow_pickle=False) as archive:
        capacity = archive["capacity_ah"]
        features = archive["features"]
        eol = int(archive["eol_cycle"].item())
    if capacity.ndim != 1 or eol != eol_cycle(capacity):
        raise ValueError(
            f"{path}: saved EOL disagrees with the two-consecutive-cycle rule "
            "on raw capacity at 0.880 Ah. Run python -m mcg_former.prepare_calce again."
        )
    return features, capacity, eol


def load_battery(cfg, name):
    directory = Path(cfg.DATA_DIR)
    if cfg.DATASET == "nasa":
        path = directory / f"{name}.mat"
        cycles, capacity, eol = load_nasa_arrays(path, cfg)
    else:
        path = directory / f"{name}.npz"
        cycles, capacity, eol = load_calce_arrays(path)
    if capacity.ndim != 1 or cycles.shape != (len(capacity), cfg.CYCLE_LEN, cfg.INPUT_DIM):
        raise ValueError(f"{path}: unexpected capacity or feature shape.")
    if (not 0 < eol <= len(capacity)
            or not np.isfinite(capacity).all()
            or not np.isfinite(cycles).all()):
        raise ValueError(f"{path}: invalid capacity, cycle features or EOL index.")
    soh = (capacity[:eol] / cfg.INITIAL_CAPACITY).astype(np.float32)
    return Battery(name, cycles[:eol], soh, eol)


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


class CrossBatteryGraphDataset(Dataset):
    def __init__(self, batteries, cfg):
        self.cfg = cfg
        self.samples = [
            (battery, idx)
            for battery in batteries
            for idx in range(5, len(battery.soh) - 5)
        ]
        if not self.samples:
            raise ValueError("The selected training cells have no training windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        battery, start = self.samples[idx]
        item = build_train_sample(battery.cycles, battery.soh, start, self.cfg)
        keys = ("hist_cycles", "hist_soh", "future_soh", "future_mask", "rul", "handcrafted")
        return tuple(torch.from_numpy(item[key]) for key in keys)


def make_loader(batteries, cfg):
    return DataLoader(
        CrossBatteryGraphDataset(batteries, cfg),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
