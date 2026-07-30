from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    # -----------------------------
    # Data
    # -----------------------------
    DATA_DIR: str = "./data/NASA"
    BATTERY_NAMES: List[str] = field(default_factory=lambda: ["B0005", "B0006", "B0007", "B0018"])
    INPUT_DIM: int = 4
    CYCLE_LEN: int = 100
    TRAIN_WINDOW_L: int = 20
    WINDOW_H: int = 180
    SHORT_H: int = 30
    TEST_START: int = 70

    # -----------------------------
    # Capacity / SOH / EOL
    # -----------------------------
    INITIAL_CAPACITY: float = 2.0
    EOL_THRESHOLD: float = 0.70
    EOL_SMOOTH_K: int = 3
    EOL_STABLE_K: int = 2
    EOL_CAPACITY_MAP: Dict[str, float] = field(default_factory=lambda: {
        "B0005": 1.40,
        "B0006": 1.40,
        "B0007": 1.44,
        "B0018": 1.40,
    })

    # -----------------------------
    # Graph model
    # -----------------------------
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

    # -----------------------------
    # Auxiliary monotonic trajectory
    # -----------------------------
    DROP_SCALE: float = 0.01

    # -----------------------------
    # Optimization
    # -----------------------------
    LR: float = 3e-4
    MIN_LR: float = 1e-5
    WEIGHT_DECAY: float = 1e-4

    # 最大训练轮数，不代表一定训练满
    EPOCHS: int = 180

    BATCH_SIZE: int = 8
    CLIP_GRAD_NORM: float = 1.0
    SEED: int = 42
    DEVICE: str = "cuda"

    # -----------------------------
    # Early stopping
    # -----------------------------
    EARLY_STOPPING: bool = True

    # 推荐 NASA 小数据集用 30
    EARLY_STOP_PATIENCE: int = 30

    # 验证指标至少改善这么多，才算真正 improvement
    EARLY_STOP_MIN_DELTA: float = 1e-4

    # 前多少轮不允许 early stop，防止太早停止
    EARLY_STOP_WARMUP: int = 30

    # -----------------------------
    # Loss
    # -----------------------------
    UNDER_PENALTY: float = 1.30
    RUL_LOSS_WEIGHT: float = 1.0
    TRAJ_LOSS_WEIGHT: float = 0.6
    SHORT_TRAJ_LOSS_WEIGHT: float = 0.5
    TRAJ_L1_LOSS_WEIGHT: float = 0.2
    SLOPE_LOSS_WEIGHT: float = 0.1