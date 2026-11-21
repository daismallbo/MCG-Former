"""Fixed settings used for the two final MCG-Former experiments."""

from dataclasses import dataclass, replace


NASA_CELLS = ("B0005", "B0006", "B0007", "B0018")
CALCE_CELLS = ("CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38")

# Mapping from outer test cell to the settings already selected on its
# development cells. Values are graph depth, similarity Top-k and history length.
NASA_FOLD_SETTINGS = {
    "B0005": (3, 2, 20),
    "B0006": (2, 1, 20),
    "B0007": (6, 4, 20),
    "B0018": (3, 3, 20),
}


@dataclass(frozen=True)
class Config:
    DATASET: str
    BATTERY_NAMES: tuple
    TEST_STARTS: tuple
    INPUT_DIM: int
    INITIAL_CAPACITY: float
    DATA_DIR: str
    CYCLE_LEN: int = 100
    TRAIN_WINDOW_L: int = 20
    WINDOW_H: int = 180
    SHORT_H: int = 30

    CNN_DIM: int = 32
    NODE_DIM: int = 64
    MODEL_DIM: int = 96
    FFN_DIM: int = 192
    N_HEADS: int = 4
    N_LAYERS: int = 6
    DROPOUT: float = 0.15
    TOPK_SIM: int = 4
    TOPK_EVENT: int = 1
    TEMPORAL_RADIUS: int = 2
    DROP_SCALE: float = 0.01

    LR: float = 3e-4
    MIN_LR: float = 1e-5
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 180
    BATCH_SIZE: int = 8
    CLIP_GRAD_NORM: float = 1.0
    SEED: int = 42
    EARLY_STOPPING: bool = True
    EARLY_STOP_PATIENCE: int = 30
    EARLY_STOP_MIN_DELTA: float = 1e-4
    EARLY_STOP_WARMUP: int = 30

    UNDER_PENALTY: float = 1.30
    RUL_LOSS_WEIGHT: float = 1.0
    TRAJ_LOSS_WEIGHT: float = 0.6
    SHORT_TRAJ_LOSS_WEIGHT: float = 0.5
    TRAJ_L1_LOSS_WEIGHT: float = 0.2
    SLOPE_LOSS_WEIGHT: float = 0.1

    def for_outer_cell(self, name):
        if self.DATASET == "nasa":
            layers, topk, window = NASA_FOLD_SETTINGS[name]
            return replace(self, N_LAYERS=layers, TOPK_SIM=topk, TRAIN_WINDOW_L=window)
        return self

    def validation_start_groups(self):
        if self.DATASET == "nasa":
            return tuple((start,) for start in self.TEST_STARTS)
        return (self.TEST_STARTS,)


def make_config(dataset):
    if dataset == "nasa":
        return Config("nasa", NASA_CELLS, (30, 50, 70), 4, 2.00, "data/NASA")
    if dataset == "calce":
        return Config("calce", CALCE_CELLS, (300, 400, 500), 3, 1.10, "data/processed_calce")
    raise ValueError(f"Unknown dataset: {dataset}")
