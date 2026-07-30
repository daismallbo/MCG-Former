from typing import List
import torch
from torch.utils.data import Dataset, DataLoader

from utils import get_battery_data, build_train_sample


class CrossBatteryGraphDataset(Dataset):
    def __init__(self, data_dir: str, battery_names: List[str], cfg):
        self.samples = []
        self.cfg = cfg

        for b in battery_names:
            cycles, soh, eol_cycle = get_battery_data(data_dir, b, cfg)
            max_idx = max(5, len(soh) - 5)
            for idx in range(5, max_idx):
                item = build_train_sample(cycles, soh, idx, cfg)
                item["battery"] = b
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["hist_cycles"]),
            torch.from_numpy(s["hist_soh"]),
            torch.from_numpy(s["future_soh"]),
            torch.from_numpy(s["future_mask"]),
            torch.from_numpy(s["rul"]),
            torch.from_numpy(s["handcrafted"]),
        )


def make_loader(data_dir: str, battery_names: List[str], cfg, shuffle: bool) -> DataLoader:
    ds = CrossBatteryGraphDataset(data_dir, battery_names, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
