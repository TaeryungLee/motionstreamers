from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from datasets.hsi_method import HSI_METHOD_SPECS
from models.stage2_generator import SceneTokenEncoderViT, TimestepEmbedder


def _bool_mask(values: torch.Tensor) -> torch.Tensor:
    return values.to(dtype=torch.bool)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe[:, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.shape[0]].to(dtype=x.dtype, device=x.device))


class GoalEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.net(goal).unsqueeze(1)


class HSITransformerBase(nn.Module):
    def __init__(
        self,
        motion_dim: int = 84,
        window_frames: int = 16,
        dim_model: int = 512,
        num_heads: int = 16,
        num_layers: int = 8,
        dropout: float = 0.1,
        num_timesteps: int = 100,
        use_scene: bool = True,
        use_text: bool = True,
        use_hand_goal: bool = True,
        use_body_goal: bool = True,
        num_text_buckets: int = 4096,
        num_goal_type_buckets: int = 512,
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.window_frames = int(window_frames)
        self.dim_model = int(dim_model)
        self.use_scene = bool(use_scene)
        self.use_text = bool(use_text)
        self.use_hand_goal = bool(use_hand_goal)
        self.use_body_goal = bool(use_body_goal)
        self.input = nn.Linear(self.motion_dim, self.dim_model)
        self.output = nn.Linear(self.dim_model, self.motion_dim)
        self.timestep = nn.Sequential(
            TimestepEmbedder(self.dim_model, max_len=max(int(num_timesteps) + 1, 512)),
            nn.Linear(self.dim_model, self.dim_model),
            nn.SiLU(),
            nn.Linear(self.dim_model, self.dim_model),
        )
        self.pos = SinusoidalPositionalEncoding(self.dim_model, dropout=dropout, max_len=max(self.window_frames + 16, 512))
        self.scene = SceneTokenEncoderViT(64, self.dim_model, heads=num_heads, dropout=dropout) if self.use_scene else None
        self.text = nn.Embedding(int(num_text_buckets), self.dim_model) if self.use_text else None
        self.goal_type = nn.Embedding(int(num_goal_type_buckets), self.dim_model)
        self.body_goal = GoalEncoder(self.dim_model) if self.use_body_goal else None
        self.hand_goal = GoalEncoder(self.dim_model) if self.use_hand_goal else None
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=int(num_heads),
            dim_feedforward=self.dim_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def _time_token(self, diffusion_t: torch.Tensor) -> torch.Tensor:
        return self.timestep(diffusion_t).unsqueeze(1)

    def _condition_tokens(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> torch.Tensor:
        x_t = batch["x_t"]
        tokens = [self._time_token(diffusion_t)]
        if self.scene is not None:
            tokens.append(self.scene(batch["scene_occ"].to(dtype=x_t.dtype))[:, :1])
        if self.text is not None:
            text = self.text(batch["text_id"].long()) + self.goal_type(batch["goal_type_id"].long())
            tokens.append(text.unsqueeze(1))
        if self.body_goal is not None:
            body = self.body_goal(batch["body_goal_cond"].to(dtype=x_t.dtype))
            body = body * _bool_mask(batch["need_pelvis_dir"]).to(dtype=x_t.dtype)[:, None, None]
            tokens.append(body)
        if self.hand_goal is not None:
            hand = self.hand_goal(batch["hand_goal_cond"].to(dtype=x_t.dtype))
            hand = hand * _bool_mask(batch["is_pick"]).to(dtype=x_t.dtype)[:, None, None]
            tokens.append(hand)
        return torch.cat(tokens, dim=1)

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        x_t = batch["x_t"]
        cond = self._condition_tokens(batch, diffusion_t)
        motion = self.input(x_t) * math.sqrt(self.dim_model)
        tokens = torch.cat([cond, motion], dim=1).transpose(0, 1)
        tokens = self.pos(tokens)
        hidden = self.transformer(tokens)
        pred = self.output(hidden[-self.window_frames :].transpose(0, 1))
        return {"eps_hat": pred}


class LingoHSIModel(HSITransformerBase):
    """LINGO-style 16-frame autoregressive HSI Transformer."""

    def __init__(self, motion_dim: int = 84, **kwargs: Any) -> None:
        spec = HSI_METHOD_SPECS["lingo"]
        super().__init__(
            motion_dim=motion_dim,
            window_frames=spec.window_frames,
            dim_model=int(kwargs.get("hidden_dim", 512)),
            num_heads=int(kwargs.get("num_heads", 16)),
            num_layers=int(kwargs.get("num_layers", 8)),
            dropout=float(kwargs.get("dropout", 0.1)),
            num_timesteps=int(kwargs.get("num_timesteps", 100)),
            use_scene=True,
            use_text=True,
            use_hand_goal=True,
            use_body_goal=True,
        )


class TrumansHSIModel(HSITransformerBase):
    """TRUMANS/SynHSI-style 32-frame autoregressive HSI Transformer."""

    def __init__(self, motion_dim: int = 84, **kwargs: Any) -> None:
        spec = HSI_METHOD_SPECS["trumans"]
        super().__init__(
            motion_dim=motion_dim,
            window_frames=spec.window_frames,
            dim_model=int(kwargs.get("hidden_dim", 512)),
            num_heads=int(kwargs.get("num_heads", 16)),
            num_layers=int(kwargs.get("num_layers", 8)),
            dropout=float(kwargs.get("dropout", 0.1)),
            num_timesteps=int(kwargs.get("num_timesteps", 100)),
            use_scene=True,
            use_text=True,
            use_hand_goal=False,
            use_body_goal=True,
        )


class DynHSIModel(HSITransformerBase):
    """Dyn-HSI-style 48-frame Transformer with condition adapter."""

    def __init__(self, motion_dim: int = 84, **kwargs: Any) -> None:
        spec = HSI_METHOD_SPECS["dyn_hsi"]
        hidden = int(kwargs.get("hidden_dim", 512))
        super().__init__(
            motion_dim=motion_dim,
            window_frames=spec.window_frames,
            dim_model=hidden,
            num_heads=int(kwargs.get("num_heads", 16)),
            num_layers=int(kwargs.get("num_layers", 8)),
            dropout=float(kwargs.get("dropout", 0.1)),
            num_timesteps=int(kwargs.get("num_timesteps", 100)),
            use_scene=True,
            use_text=True,
            use_hand_goal=True,
            use_body_goal=True,
        )
        self.adapter = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
            nn.Sigmoid(),
        )
        self.traj = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def _condition_tokens(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> torch.Tensor:
        x_t = batch["x_t"]
        time = self._time_token(diffusion_t)
        scene = self.scene(batch["scene_occ"].to(dtype=x_t.dtype))[:, :1]
        text = (self.text(batch["text_id"].long()) + self.goal_type(batch["goal_type_id"].long())).unsqueeze(1)
        goal = self.body_goal(batch["body_goal_cond"].to(dtype=x_t.dtype))
        start = x_t[:, 0, :3]
        goal_vec = batch["body_goal_cond"].to(dtype=x_t.dtype)
        traj = self.traj(torch.cat([start, goal_vec], dim=-1)).unsqueeze(1)
        weights = self.adapter(text.squeeze(1)).to(dtype=x_t.dtype)
        scene = scene * weights[:, 0:1, None]
        traj = traj * weights[:, 1:2, None]
        text = text * weights[:, 2:3, None]
        goal = goal * weights[:, 3:4, None]
        if self.hand_goal is not None:
            hand = self.hand_goal(batch["hand_goal_cond"].to(dtype=x_t.dtype))
            hand = hand * _bool_mask(batch["is_pick"]).to(dtype=x_t.dtype)[:, None, None]
            goal = goal + hand
        return torch.cat([time, scene, traj, text, goal], dim=1)


def build_hsi_method_model(method: str, motion_dim: int = 84, **kwargs: Any) -> nn.Module:
    key = str(method).lower()
    if key == "lingo":
        return LingoHSIModel(motion_dim=motion_dim, **kwargs)
    if key == "trumans":
        return TrumansHSIModel(motion_dim=motion_dim, **kwargs)
    if key == "dyn_hsi":
        return DynHSIModel(motion_dim=motion_dim, **kwargs)
    raise ValueError(f"unknown HSI method: {method}")
