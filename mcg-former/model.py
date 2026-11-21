import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CycleCNNEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, out_dim, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feat = self.net(x).squeeze(-1)
        return feat


class RelationProjector(nn.Module):
    def __init__(self, node_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Linear(node_dim + 1, model_dim)

    def forward(self, node_feat: torch.Tensor, hist_soh: torch.Tensor) -> torch.Tensor:
        x = torch.cat([node_feat, hist_soh.unsqueeze(-1)], dim=-1)
        return self.proj(x)


class GraphBuilder(nn.Module):
    def __init__(self, temporal_radius: int, topk_sim: int, topk_event: int):
        super().__init__()
        self.temporal_radius = temporal_radius
        self.topk_sim = topk_sim
        self.topk_event = topk_event

        self.w_temporal = nn.Parameter(torch.tensor(1.0))
        self.w_sim = nn.Parameter(torch.tensor(1.0))
        self.w_event = nn.Parameter(torch.tensor(1.0))

    def _temporal_adj(self, B: int, L: int, device):
        idx = torch.arange(L, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        adj = (dist <= self.temporal_radius).float()
        return adj.unsqueeze(0).repeat(B, 1, 1)

    def _similarity_adj(self, x: torch.Tensor):
        x_n = F.normalize(x, dim=-1)
        sim = torch.matmul(x_n, x_n.transpose(1, 2))
        B, L, _ = sim.shape
        topk = min(self.topk_sim + 1, L)
        vals, idxs = torch.topk(sim, k=topk, dim=-1)
        adj = torch.zeros_like(sim)
        adj.scatter_(dim=-1, index=idxs, src=torch.ones_like(vals))
        return adj

    def _event_adj(self, hist_soh: torch.Tensor):
        B, L = hist_soh.shape
        diff = torch.zeros_like(hist_soh)
        diff[:, 1:] = torch.abs(hist_soh[:, 1:] - hist_soh[:, :-1])
        topk = min(self.topk_event, L)
        _, event_idx = torch.topk(diff, k=topk, dim=-1)
        adj = torch.zeros(B, L, L, device=hist_soh.device)
        for b in range(B):
            ids = event_idx[b]
            adj[b, ids[:, None], ids[None, :]] = 1.0
        adj[:, torch.arange(L), torch.arange(L)] = 1.0
        return adj

    def forward(self, x: torch.Tensor, hist_soh: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, _ = x.shape
        device = x.device
        a_t = self._temporal_adj(B, L, device)
        a_s = self._similarity_adj(x)
        a_e = self._event_adj(hist_soh)
        bias = self.w_temporal * a_t + self.w_sim * a_s + self.w_event * a_e
        return {
            "temporal_adj": a_t,
            "similarity_adj": a_s,
            "event_adj": a_e,
            "attn_bias": bias,
        }


class GraphMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GraphMultiheadSelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class AttentionReadout(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.score(x), dim=1)
        h = torch.sum(w * x, dim=1)
        return h, w.squeeze(-1)


class MCGFormerRULNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.cycle_encoder = CycleCNNEncoder(cfg.INPUT_DIM, cfg.CNN_DIM, cfg.NODE_DIM)
        self.node_fusion = RelationProjector(cfg.NODE_DIM, cfg.MODEL_DIM)
        self.graph_builder = GraphBuilder(cfg.TEMPORAL_RADIUS, cfg.TOPK_SIM, cfg.TOPK_EVENT)

        self.layers = nn.ModuleList([
            GraphTransformerBlock(cfg.MODEL_DIM, cfg.N_HEADS, cfg.FFN_DIM, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        ])

        self.handcrafted_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )

        self.readout = AttentionReadout(cfg.MODEL_DIM, cfg.DROPOUT)

        fusion_dim = cfg.MODEL_DIM + 32
        self.rul_head = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(96, 1),
        )

        self.step_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(128, cfg.WINDOW_H),
        )

    def forward(self, hist_cycles: torch.Tensor, hist_soh: torch.Tensor, handcrafted: torch.Tensor):
        B, L, S, C = hist_cycles.shape
        flat = hist_cycles.reshape(B * L, S, C)
        node_feat = self.cycle_encoder(flat).reshape(B, L, -1)
        x = self.node_fusion(node_feat, hist_soh)

        graph_info = self.graph_builder(x, hist_soh)
        bias = graph_info["attn_bias"]

        for blk in self.layers:
            x = blk(x, bias)

        pooled, attn_weights = self.readout(x)
        hand = self.handcrafted_encoder(handcrafted)
        fused = torch.cat([pooled, hand], dim=-1)

        rul_pred = self.rul_head(fused)

        drops = torch.nn.functional.softplus(self.step_head(fused)) * self.cfg.DROP_SCALE
        init_soh = hist_soh[:, -1:].clone()
        future_traj = init_soh - torch.cumsum(drops, dim=1)

        return {
            "rul_pred": rul_pred,
            "future_traj": future_traj,
            "attn_weights": attn_weights,
            "graph_info": graph_info,
        }
