"""Train and test MCG-Former with the finalized NASA or CALCE protocol."""

import argparse
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch

from mcg_former.config import make_config
from mcg_former.dataset import build_handcrafted_features, left_pad_window, load_battery, make_loader
from mcg_former.losses import total_loss
from mcg_former.model import MCGFormerRULNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate_battery(models, battery, starts, cfg, device):
    for model in models:
        model.eval()
    predictions, trajectories = [], []
    for start in starts:
        cycles, soh = left_pad_window(battery.cycles, battery.soh, start, cfg.TRAIN_WINDOW_L)
        hand = build_handcrafted_features(soh)
        inputs = [torch.from_numpy(array).unsqueeze(0).to(device) for array in (cycles, soh, hand)]
        rul_outputs, soh_outputs = [], []
        for model in models:
            output = model(*inputs)
            rul_outputs.append(output["rul_pred"].item())
            soh_outputs.append(output["future_traj"].squeeze(0).cpu().numpy())

        trajectory = np.mean(np.stack(soh_outputs), axis=0)
        if not np.isfinite(rul_outputs).all() or not np.isfinite(trajectory).all():
            raise FloatingPointError(f"{battery.name} at cycle {start}: non-finite prediction.")

        # start is the last observed 1-based discharge index. The first
        # forecast is for cycle start + 1, whose array index is start.
        true_rul = battery.eol_cycle - start
        scored_steps = min(cfg.WINDOW_H, true_rul)
        errors = trajectory[:scored_steps] - battery.soh[start:start + scored_steps]
        predicted_rul = int(round(float(np.mean(rul_outputs))))
        predictions.append({
            "battery": battery.name,
            "start": start,
            "true_eol": battery.eol_cycle,
            "true_rul": true_rul,
            "pred_rul": predicted_rul,
            "ae": abs(true_rul - predicted_rul),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
            "scored_trajectory_steps": scored_steps,
            "forecast_horizon": cfg.WINDOW_H,
            "n_layers": cfg.N_LAYERS,
            "topk_sim": cfg.TOPK_SIM,
            "window_len": cfg.TRAIN_WINDOW_L,
            "num_ensemble_models": len(models),
        })
        trajectories.append(trajectory)
    return predictions, trajectories


def validation_key(rows):
    return tuple(float(np.mean([row[k] for row in rows])) for k in ("ae", "rmse", "mae"))


def is_improved(candidate, best, delta):
    if best is None:
        return True
    for current, previous in zip(candidate, best):
        if abs(current - previous) > delta:
            return current < previous
    return False


def train_inner_fold(training, validation, starts, cfg, device, seed):
    seed_everything(seed)
    loader = make_loader(training, cfg)
    model = MCGFormerRULNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR
    )
    best, best_state, best_epoch, stale = None, None, 0, 0
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        losses = []
        for batch in loader:
            cycles, soh, future, mask, rul, hand = [item.to(device) for item in batch]
            outputs = model(cycles, soh, hand)
            loss, info = total_loss(outputs, rul, future, mask, cfg)
            if not np.isfinite(info["loss_total"]):
                raise FloatingPointError(f"{validation.name}, epoch {epoch}: non-finite training loss.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
            optimizer.step()
            losses.append(info["loss_total"])
        scheduler.step()

        rows, _ = evaluate_battery([model], validation, starts, cfg, device)
        scores = validation_key(rows)
        if is_improved(scores, best, cfg.EARLY_STOP_MIN_DELTA):
            best = scores
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1

        ae, rmse, mae = scores
        print(f"    epoch={epoch:03d}/{cfg.EPOCHS} loss={np.mean(losses):.5f} "
              f"val_AE={ae:.2f} val_RMSE={rmse:.5f} val_MAE={mae:.5f} "
              f"best_epoch={best_epoch}", flush=True)
        if (cfg.EARLY_STOPPING and epoch >= cfg.EARLY_STOP_WARMUP
                and stale >= cfg.EARLY_STOP_PATIENCE):
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best


def write_csv(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def make_output_dir(requested):
    requested.parent.mkdir(parents=True, exist_ok=True)
    try:
        requested.mkdir()
    except FileExistsError:
        if not requested.is_dir():
            raise
        # Reserve a unique path for a rerun without overwriting older results.
        requested = Path(mkdtemp(prefix=f"{requested.name}_", dir=requested.parent))
    return requested


def append_trajectories(rows, predictions, trajectories, battery, cfg):
    for result, trajectory in zip(predictions, trajectories):
        start = result["start"]
        for step, value in enumerate(trajectory, start=1):
            cycle = start + step
            included = step <= result["scored_trajectory_steps"]
            rows.append({
                "battery": battery.name,
                "start": start,
                "step": step,
                "cycle": cycle,
                "pred_soh": float(value),
                "pred_capacity_ah": float(value) * cfg.INITIAL_CAPACITY,
                "true_soh": float(battery.soh[cycle - 1]) if included else None,
                "included_in_soh_score": included,
            })


def summarize(group, rows):
    return {
        "group": group,
        "num_cases": len(rows),
        "mean_ae": float(np.mean([row["ae"] for row in rows])),
        "mean_rmse": float(np.mean([row["rmse"] for row in rows])),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
    }


def train_and_test(cfg, test_names, requested_output, device):
    batteries = {name: load_battery(cfg, name) for name in cfg.BATTERY_NAMES}
    for name, battery in batteries.items():
        if not all(0 < start < battery.eol_cycle for start in cfg.TEST_STARTS):
            raise ValueError(f"{name}: all starting cycles must precede the EOL at {battery.eol_cycle}.")

    output = make_output_dir(requested_output)
    start_groups = cfg.validation_start_groups()
    settings = {
        **asdict(cfg),
        "device_used": str(device),
        "pytorch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "outer_test_cells": test_names,
        "eol_cycles": {name: cell.eol_cycle for name, cell in batteries.items()},
        "validation_start_groups": start_groups,
        "outer_fold_settings": {name: asdict(cfg.for_outer_cell(name)) for name in test_names},
        "status": "running",
    }
    write_json(output / "settings.json", settings)
    print(f"Dataset={cfg.DATASET.upper()} | device={device} | output={output.resolve()}", flush=True)
    print(f"Observed EOL cycles: {settings['eol_cycles']}", flush=True)

    predictions, trajectories, folds = [], [], []
    for test_name in test_names:
        outer_idx = cfg.BATTERY_NAMES.index(test_name)
        fold_cfg = cfg.for_outer_cell(test_name)
        development = [name for name in cfg.BATTERY_NAMES if name != test_name]
        print(f"[Outer] {test_name} | layers={fold_cfg.N_LAYERS} | "
              f"similarity Top-k={fold_cfg.TOPK_SIM} | history={fold_cfg.TRAIN_WINDOW_L}", flush=True)
        for starts in start_groups:
            models = []
            for inner_idx, validation_name in enumerate(development):
                training_names = [name for name in development if name != validation_name]
                seed = cfg.SEED + 100 * outer_idx + 10 * inner_idx
                print(f"  [Inner] starts={starts} | train={training_names} | "
                      f"val={validation_name} | seed={seed}", flush=True)
                model, best_epoch, scores = train_inner_fold(
                    [batteries[name] for name in training_names], batteries[validation_name],
                    starts, fold_cfg, device, seed,
                )
                models.append(model)
                folds.append({
                    "outer_holdout": test_name,
                    "training_cells": "+".join(training_names),
                    "validation_cell": validation_name,
                    "validation_starts": "+".join(map(str, starts)),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_ae": scores[0],
                    "validation_rmse": scores[1],
                    "validation_mae": scores[2],
                })

            batch_rows, batch_trajectories = evaluate_battery(
                models, batteries[test_name], starts, fold_cfg, device
            )
            predictions.extend(batch_rows)
            append_trajectories(trajectories, batch_rows, batch_trajectories, batteries[test_name], cfg)
            write_csv(output / "folds.csv", folds)
            write_csv(output / "trajectories.csv", trajectories)
            write_csv(output / "predictions.csv", predictions)
            for row in batch_rows:
                print(f"  [Test] {row['battery']} start={row['start']} "
                      f"AE={row['ae']} RMSE={row['rmse']:.6f} MAE={row['mae']:.6f}", flush=True)
            del models, model

    overall = summarize("overall", predictions)
    summaries = [overall]
    summaries.extend(summarize(f"start_{start}", [r for r in predictions if r["start"] == start])
                     for start in cfg.TEST_STARTS)
    summaries.extend(summarize(name, [r for r in predictions if r["battery"] == name])
                     for name in test_names)
    write_csv(output / "summary.csv", summaries)
    settings["status"] = "completed"
    write_json(output / "settings.json", settings)
    print(f"Completed {overall['num_cases']} cases | mean AE={overall['mean_ae']:.4f} "
          f"RMSE={overall['mean_rmse']:.6f} MAE={overall['mean_mae']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("nasa", "calce"), required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-battery", nargs="+", help="Limit the outer test cells for a smoke run.")
    parser.add_argument("--epochs", type=int, help="Override the maximum epochs for a smoke run.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    cfg = make_config(args.dataset)
    if args.data_dir is not None:
        cfg = replace(cfg, DATA_DIR=str(args.data_dir))
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive.")
        cfg = replace(cfg, EPOCHS=args.epochs)

    test_names = list(dict.fromkeys(args.test_battery or cfg.BATTERY_NAMES))
    if not set(test_names).issubset(cfg.BATTERY_NAMES):
        parser.error(f"--test-battery must select from: {', '.join(cfg.BATTERY_NAMES)}.")
    cuda = torch.cuda.is_available()
    if args.device == "cuda" and not cuda:
        parser.error("CUDA is unavailable in this Python environment.")
    device = torch.device(("cuda" if cuda else "cpu") if args.device == "auto" else args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    output = args.output_dir or Path("results") / cfg.DATASET
    train_and_test(cfg, test_names, output, device)


if __name__ == "__main__":
    main()
