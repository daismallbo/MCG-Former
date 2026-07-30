import copy
import numpy as np
import torch
import torch.optim as optim

from config import Config
from dataset import make_loader
from utils import (
    set_seed,
    calculate_rmse,
    calculate_mae,
    calculate_ae,
    get_battery_data,
    build_test_window,
)
from model import MCGFormerRULNet
from losses import total_loss


def evaluate_single_battery(model_or_models, battery_name: str, cfg, device):
    """
    对单个电池进行测试/验证。

    注意：
    - 如果传入单个 model，则直接评估该 model。
    - 如果传入 model list，则做 ensemble 平均。
    """
    cycles, soh, eol_cycle = get_battery_data(cfg.DATA_DIR, battery_name, cfg)

    if not isinstance(model_or_models, list):
        model_or_models = [model_or_models]

    trajs = []
    ruls = []

    for model in model_or_models:
        model.eval()

        hist_cycles, hist_soh, hand, true_future, true_rul = build_test_window(
            cycles, soh, cfg.TEST_START, cfg
        )

        x = torch.from_numpy(hist_cycles).unsqueeze(0).to(device).float()
        s = torch.from_numpy(hist_soh).unsqueeze(0).to(device).float()
        h = torch.from_numpy(hand).unsqueeze(0).to(device).float()

        with torch.no_grad():
            out = model(x, s, h)

        pred_rul = float(out["rul_pred"].item())
        pred_traj = out["future_traj"].squeeze(0).cpu().numpy()[:len(true_future)]

        trajs.append(pred_traj)
        ruls.append(pred_rul)

    pred_future = np.mean(np.stack(trajs, axis=0), axis=0)
    pred_rul = int(round(float(np.mean(ruls))))

    true_rul = eol_cycle - cfg.TEST_START
    # pred_rul = max(1, min(pred_rul, true_rul))

    true_future = soh[cfg.TEST_START: cfg.TEST_START + len(pred_future)]

    return {
        "battery": battery_name,
        "ae": calculate_ae(true_rul, pred_rul),
        "rmse": calculate_rmse(true_future, pred_future),
        "mae": calculate_mae(true_future, pred_future),
        "pred_rul": pred_rul,
        "true_rul": true_rul,
    }


def validation_sort_key(metrics):
    """
    验证集模型选择标准。

    优先级：
    1. AE 越小越好
    2. RMSE 越小越好
    3. MAE 越小越好

    这样比较适合你的 RUL 任务：
    - AE 直接衡量 RUL 点预测误差
    - RMSE 衡量未来 SOH 轨迹质量
    - MAE 作为辅助稳定指标
    """
    return (
        float(metrics["ae"]),
        float(metrics["rmse"]),
        float(metrics["mae"]),
    )


def is_improved(cur_key, best_key, min_delta: float):
    """
    判断当前验证结果是否优于历史最优。

    cur_key / best_key 格式：
    (AE, RMSE, MAE)

    使用字典序优先级：
    - AE 明显变小，则认为提升
    - AE 基本持平时，RMSE 明显变小，则认为提升
    - AE、RMSE 基本持平时，MAE 明显变小，则认为提升
    """
    if best_key is None:
        return True

    cur_ae, cur_rmse, cur_mae = cur_key
    best_ae, best_rmse, best_mae = best_key

    # 1. AE 优先
    if cur_ae < best_ae - min_delta:
        return True

    # 2. AE 几乎相同，看 RMSE
    if abs(cur_ae - best_ae) <= min_delta:
        if cur_rmse < best_rmse - min_delta:
            return True

        # 3. AE 和 RMSE 几乎相同，看 MAE
        if abs(cur_rmse - best_rmse) <= min_delta:
            if cur_mae < best_mae - min_delta:
                return True

    return False


def clone_state_dict_to_cpu(model):
    """
    保存最优模型参数到 CPU，避免长期占用额外 GPU 显存。
    """
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def train_one_inner_fold(train_names, val_name, cfg, device, seed_offset=0):
    """
    内层训练：
    - train_names: 训练电池
    - val_name: 验证电池，用于 early stopping
    - 注意：这里不能使用外层 test battery
    """
    inner_cfg = copy.deepcopy(cfg)
    set_seed(inner_cfg.SEED + seed_offset)

    train_loader = make_loader(
        inner_cfg.DATA_DIR,
        train_names,
        inner_cfg,
        shuffle=True
    )

    model = MCGFormerRULNet(inner_cfg).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=inner_cfg.LR,
        weight_decay=inner_cfg.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=inner_cfg.EPOCHS,
        eta_min=inner_cfg.MIN_LR
    )

    best_key = None
    best_state = None
    best_epoch = 0
    best_metrics = None

    no_improve_count = 0

    print(
        f"    EarlyStopping={inner_cfg.EARLY_STOPPING} | "
        f"MaxEpochs={inner_cfg.EPOCHS} | "
        f"Patience={inner_cfg.EARLY_STOP_PATIENCE} | "
        f"Warmup={inner_cfg.EARLY_STOP_WARMUP} | "
        f"MinDelta={inner_cfg.EARLY_STOP_MIN_DELTA}"
    )

    for epoch in range(inner_cfg.EPOCHS):
        model.train()
        running = []

        for batch in train_loader:
            hist_cycles, hist_soh, future_soh, future_mask, rul, hand = batch

            hist_cycles = hist_cycles.to(device).float()
            hist_soh = hist_soh.to(device).float()
            future_soh = future_soh.to(device).float()
            future_mask = future_mask.to(device).float()
            rul = rul.to(device).float()
            hand = hand.to(device).float()

            out = model(hist_cycles, hist_soh, hand)
            loss, loss_info = total_loss(
                out,
                rul,
                future_soh,
                future_mask,
                inner_cfg
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                inner_cfg.CLIP_GRAD_NORM
            )

            optimizer.step()

            running.append(float(loss_info["loss_total"]))

        scheduler.step()

        # -----------------------------
        # Validation
        # -----------------------------
        val_metrics = evaluate_single_battery(
            model,
            val_name,
            inner_cfg,
            device
        )

        cur_key = validation_sort_key(val_metrics)

        improved = is_improved(
            cur_key,
            best_key,
            inner_cfg.EARLY_STOP_MIN_DELTA
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if improved:
            best_key = cur_key
            best_state = clone_state_dict_to_cpu(model)
            best_epoch = epoch + 1
            best_metrics = copy.deepcopy(val_metrics)
            no_improve_count = 0
            mark = "  <-- Best"
        else:
            no_improve_count += 1
            mark = ""

        print(
            f"    Epoch {epoch + 1:03d}/{inner_cfg.EPOCHS} | "
            f"LR={current_lr:.6e} | "
            f"TrainLoss={np.mean(running):.4f} | "
            f"Val={val_name} | "
            f"AE={val_metrics['ae']} | "
            f"RMSE={val_metrics['rmse']:.4f} | "
            f"MAE={val_metrics['mae']:.4f} | "
            f"NoImprove={no_improve_count:02d}/{inner_cfg.EARLY_STOP_PATIENCE}"
            f"{mark}"
        )

        # -----------------------------
        # Early stopping
        # -----------------------------
        if inner_cfg.EARLY_STOPPING:
            warmup_finished = (epoch + 1) >= inner_cfg.EARLY_STOP_WARMUP
            patience_exceeded = no_improve_count >= inner_cfg.EARLY_STOP_PATIENCE

            if warmup_finished and patience_exceeded:
                print(
                    f"    Early stopping triggered at epoch {epoch + 1}. "
                    f"Best epoch = {best_epoch} | "
                    f"Best Val AE={best_metrics['ae']} | "
                    f"Best Val RMSE={best_metrics['rmse']:.4f} | "
                    f"Best Val MAE={best_metrics['mae']:.4f}"
                )
                break

    # -----------------------------
    # Load best model
    # -----------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"    Loaded best model from epoch {best_epoch} | "
        f"Val={val_name} | "
        f"Best AE={best_metrics['ae']} | "
        f"Best RMSE={best_metrics['rmse']:.4f} | "
        f"Best MAE={best_metrics['mae']:.4f}"
    )

    return model


def train_and_test_loso():
    """
    Nested LOSO:
    外层：
        1 块电池作为 test battery，完全不参与训练和 early stopping。

    内层：
        剩下 3 块 dev batteries 中：
        - 2 块训练
        - 1 块验证，用于 early stopping 和 best model selection

    最终：
        对每个 outer test battery，得到 3 个 inner model。
        这 3 个 model ensemble 后，在 test battery 上测试。
    """
    cfg = Config()

    device = torch.device(
        cfg.DEVICE if torch.cuda.is_available() else "cpu"
    )

    set_seed(cfg.SEED)

    all_ae = []
    all_rmse = []
    all_mae = []

    print("=" * 100)
    print("MCG-Former Nested LOSO Training with Early Stopping")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Max Epochs: {cfg.EPOCHS}")
    print(f"Early Stopping: {cfg.EARLY_STOPPING}")
    print(f"Patience: {cfg.EARLY_STOP_PATIENCE}")
    print(f"Warmup: {cfg.EARLY_STOP_WARMUP}")
    print(f"Min Delta: {cfg.EARLY_STOP_MIN_DELTA}")
    print("=" * 100)

    for outer_idx, test_bat in enumerate(cfg.BATTERY_NAMES):
        dev_bats = [
            b for b in cfg.BATTERY_NAMES
            if b != test_bat
        ]

        print(f"\n[Outer Test] {test_bat} | Dev={dev_bats}")

        inner_models = []

        for inner_idx, val_bat in enumerate(dev_bats):
            train_bats = [
                b for b in dev_bats
                if b != val_bat
            ]

            print(
                f"\n  [Inner Fold] "
                f"Train={train_bats} | Val={val_bat}"
            )

            model = train_one_inner_fold(
                train_bats,
                val_bat,
                cfg,
                device,
                seed_offset=outer_idx * 100 + inner_idx * 10
            )

            inner_models.append(model)

        # -----------------------------
        # Outer test
        # -----------------------------
        test_metrics = evaluate_single_battery(
            inner_models,
            test_bat,
            cfg,
            device
        )

        all_ae.append(test_metrics["ae"])
        all_rmse.append(test_metrics["rmse"])
        all_mae.append(test_metrics["mae"])

        print(
            f"\n[Test Result] {test_bat} | "
            f"TrueRUL={test_metrics['true_rul']} | "
            f"PredRUL={test_metrics['pred_rul']} | "
            f"AE={test_metrics['ae']} | "
            f"RMSE={test_metrics['rmse']:.4f} | "
            f"MAE={test_metrics['mae']:.4f}"
        )

    print("\n" + "-" * 100)
    print("Final Nested LOSO Results")
    print("-" * 100)
    print(f"AE list   : {all_ae}")
    print(f"RMSE list : {[round(x, 4) for x in all_rmse]}")
    print(f"MAE list  : {[round(x, 4) for x in all_mae]}")
    print("-" * 100)
    print(f"Mean AE   : {np.mean(all_ae):.2f}")
    print(f"Mean RMSE : {np.mean(all_rmse):.4f}")
    print(f"Mean MAE  : {np.mean(all_mae):.4f}")
    print("-" * 100)




if __name__ == "__main__":
    train_and_test_loso()