import torch
import torch.nn.functional as F


def rul_loss_fn(pred_rul: torch.Tensor, true_rul: torch.Tensor, under_penalty: float = 1.3):
    err = pred_rul.view(-1) - true_rul.view(-1)
    base = torch.where(err < 0, under_penalty * torch.abs(err), torch.abs(err))
    return base.mean()


def trajectory_losses(pred_traj, true_traj, traj_mask, short_h: int):
    mask = traj_mask > 0.5

    mse = (pred_traj - true_traj) ** 2
    long_loss = mse[mask].mean() if mask.any() else mse.mean()

    short_h = min(short_h, true_traj.shape[1])
    short_mask = mask[:, :short_h]
    short_mse = (pred_traj[:, :short_h] - true_traj[:, :short_h]) ** 2
    short_loss = short_mse[short_mask].mean() if short_mask.any() else short_mse.mean()

    l1 = torch.abs(pred_traj - true_traj)
    l1_loss = l1[mask].mean() if mask.any() else l1.mean()

    pred_step = pred_traj[:, 1:] - pred_traj[:, :-1]
    true_step = true_traj[:, 1:] - true_traj[:, :-1]
    slope_mask = mask[:, 1:] & mask[:, :-1]
    slope_loss = F.l1_loss(pred_step[slope_mask], true_step[slope_mask]) if slope_mask.any() else F.l1_loss(pred_step, true_step)

    return long_loss, short_loss, l1_loss, slope_loss


def total_loss(outputs, true_rul, true_traj, traj_mask, cfg):
    rul = rul_loss_fn(outputs["rul_pred"], true_rul, cfg.UNDER_PENALTY)
    long_l, short_l, l1_l, slope_l = trajectory_losses(outputs["future_traj"], true_traj, traj_mask, cfg.SHORT_H)

    total = (
        cfg.RUL_LOSS_WEIGHT * rul
        + cfg.TRAJ_LOSS_WEIGHT * long_l
        + cfg.SHORT_TRAJ_LOSS_WEIGHT * short_l
        + cfg.TRAJ_L1_LOSS_WEIGHT * l1_l
        + cfg.SLOPE_LOSS_WEIGHT * slope_l
    )

    metrics = {
        "loss_total": float(total.item()),
        "loss_rul": float(rul.item()),
        "loss_traj_long": float(long_l.item()),
        "loss_traj_short": float(short_l.item()),
        "loss_traj_l1": float(l1_l.item()),
        "loss_slope": float(slope_l.item()),
    }
    return total, metrics
