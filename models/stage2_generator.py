from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half <= 0:
        return timesteps[:, None].float()
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def group_count(channels: int) -> int:
    groups = min(32, int(channels))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return groups


def resize_temporal(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] == int(length):
        return x
    return F.interpolate(x, size=int(length), mode="linear", align_corners=False)


class AdaGN1D(nn.Module):
    def __init__(self, channels: int, global_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.global_proj = nn.Linear(global_dim, channels * 2)

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        temporal_scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.global_proj(global_cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None]
        beta = beta[:, :, None]
        if temporal_scale_shift is not None:
            gamma_t, beta_t = temporal_scale_shift
            gamma = gamma + gamma_t
            beta = beta + beta_t
        return h * (1.0 + gamma) + beta


class TrajectoryFiLM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, plan_full: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        plan = plan_full.transpose(1, 2)
        plan = resize_temporal(plan, int(length))
        gamma, beta = self.net(plan).chunk(2, dim=1)
        return gamma, beta


class CrossAttention1D(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        heads = max(1, min(int(num_heads), int(channels)))
        while int(channels) % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.context_proj = nn.Linear(context_dim, channels) if context_dim != channels else nn.Identity()
        self.attn = nn.MultiheadAttention(channels, heads, dropout=float(dropout), batch_first=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.numel() == 0:
            return x
        q = self.norm(x.transpose(1, 2))
        kv = self.context_proj(context)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return x + out.transpose(1, 2)


class STResBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        global_dim: int,
        context_dim: int,
        num_heads: int,
        dropout: float,
        use_traj_film: bool,
    ) -> None:
        super().__init__()
        self.use_traj_film = bool(use_traj_film)
        self.norm1 = AdaGN1D(channels, global_dim)
        self.norm2 = AdaGN1D(channels, global_dim)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(float(dropout))
        self.cross = CrossAttention1D(channels, context_dim, num_heads=num_heads, dropout=dropout)
        self.traj1 = TrajectoryFiLM(channels) if self.use_traj_film else None
        self.traj2 = TrajectoryFiLM(channels) if self.use_traj_film else None

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        context: torch.Tensor,
        plan_full: torch.Tensor | None,
    ) -> torch.Tensor:
        scale_shift1 = self.traj1(plan_full, x.shape[-1]) if self.traj1 is not None and plan_full is not None else None
        h = self.norm1(x, global_cond, scale_shift1)
        h = F.gelu(h)
        h = self.conv1(h)
        h = self.dropout(h)
        scale_shift2 = self.traj2(plan_full, h.shape[-1]) if self.traj2 is not None and plan_full is not None else None
        h = self.norm2(h, global_cond, scale_shift2)
        h = F.gelu(h)
        h = self.conv2(h)
        x = x + h
        return self.cross(x, context)


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        return self.conv(x)


class SceneTokenEncoder3D(nn.Module):
    def __init__(self, in_channels: int = 1, context_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv3d(64, context_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(group_count(context_dim), context_dim),
            nn.GELU(),
        )
        self.type_embed = nn.Parameter(torch.randn(2, context_dim) * 0.02)

    def forward(self, current: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        cur = self.net(current).flatten(2).transpose(1, 2) + self.type_embed[0][None, None]
        goal_tokens = self.net(goal).flatten(2).transpose(1, 2) + self.type_embed[1][None, None]
        return torch.cat([cur, goal_tokens], dim=1)


class HistoryEncoder(nn.Module):
    def __init__(self, motion_dim: int, context_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(motion_dim, context_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(context_dim, context_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.summary = nn.Parameter(torch.randn(1, 1, context_dim) * 0.02)

    def forward(self, motion: torch.Tensor, history_frames: torch.Tensor) -> torch.Tensor:
        h_len = max(1, int(history_frames.max().item()))
        hist = motion[:, :h_len].transpose(1, 2)
        tokens = self.net(hist).transpose(1, 2)
        summary = self.summary.expand(motion.shape[0], -1, -1)
        return torch.cat([summary, tokens], dim=1)


class GoalTokenEncoder(nn.Module):
    def __init__(self, context_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim))
        self.left = nn.Sequential(nn.Linear(4, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim))
        self.right = nn.Sequential(nn.Linear(4, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim))
        self.type_embed = nn.Parameter(torch.randn(3, context_dim) * 0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch["goal_valid"].to(batch["body_goal"].dtype)
        body = self.body(torch.cat([batch["body_goal"], valid[:, 0:1]], dim=-1)) + self.type_embed[0][None]
        left = self.left(torch.cat([batch["left_hand_goal"], valid[:, 1:2]], dim=-1)) + self.type_embed[1][None]
        right = self.right(torch.cat([batch["right_hand_goal"], valid[:, 2:3]], dim=-1)) + self.type_embed[2][None]
        return torch.stack([body, left, right], dim=1)


class TextTokenEncoder(nn.Module):
    def __init__(self, context_dim: int, vocab_size: int = 4096) -> None:
        super().__init__()
        self.text = nn.Embedding(vocab_size, context_dim)
        self.goal_type = nn.Embedding(vocab_size, context_dim)
        self.type_embed = nn.Parameter(torch.randn(2, context_dim) * 0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        text = self.text(batch["text_id"].long()) + self.type_embed[0][None]
        goal = self.goal_type(batch["goal_type_id"].long()) + self.type_embed[1][None]
        return torch.stack([text, goal], dim=1)


@dataclass
class CondUNetConfig:
    motion_dim: int
    input_extra_dim: int
    hidden_dim: int = 256
    context_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    use_traj_film: bool = False


class CondUNet1D(nn.Module):
    def __init__(self, config: CondUNetConfig) -> None:
        super().__init__()
        self.config = config
        c0 = int(config.hidden_dim)
        c1 = int(config.hidden_dim)
        c2 = int(config.hidden_dim * 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.input = nn.Conv1d(config.motion_dim + config.input_extra_dim, c0, kernel_size=1)
        self.down0 = STResBlock1D(c0, config.hidden_dim, config.context_dim, config.num_heads, config.dropout, config.use_traj_film)
        self.to_down1 = Downsample1D(c0, c1)
        self.down1 = STResBlock1D(c1, config.hidden_dim, config.context_dim, config.num_heads, config.dropout, config.use_traj_film)
        self.to_mid = Downsample1D(c1, c2)
        self.mid = STResBlock1D(c2, config.hidden_dim, config.context_dim, config.num_heads, config.dropout, config.use_traj_film)
        self.up1 = Upsample1D(c2, c1)
        self.fuse1 = nn.Conv1d(c1 + c1, c1, kernel_size=1)
        self.block_up1 = STResBlock1D(c1, config.hidden_dim, config.context_dim, config.num_heads, config.dropout, config.use_traj_film)
        self.up0 = Upsample1D(c1, c0)
        self.fuse0 = nn.Conv1d(c0 + c0, c0, kernel_size=1)
        self.block_up0 = STResBlock1D(c0, config.hidden_dim, config.context_dim, config.num_heads, config.dropout, config.use_traj_film)
        self.out_norm = nn.GroupNorm(group_count(c0), c0)
        self.out = nn.Conv1d(c0, config.motion_dim, kernel_size=1)

    def timestep_cond(self, diffusion_t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(diffusion_t, self.config.hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        extra: torch.Tensor,
        context: torch.Tensor,
        diffusion_t: torch.Tensor,
        plan_full: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global_cond = self.timestep_cond(diffusion_t)
        inp = torch.cat([x, extra], dim=-1).transpose(1, 2)
        h0 = self.input(inp)
        h0 = self.down0(h0, global_cond, context, plan_full)
        skip0 = h0
        h1 = self.to_down1(h0)
        h1 = self.down1(h1, global_cond, context, plan_full)
        skip1 = h1
        h2 = self.to_mid(h1)
        h2 = self.mid(h2, global_cond, context, plan_full)
        h = self.up1(h2, skip1.shape[-1])
        h = self.fuse1(torch.cat([h, skip1], dim=1))
        h = self.block_up1(h, global_cond, context, plan_full)
        h = self.up0(h, skip0.shape[-1])
        h = self.fuse0(torch.cat([h, skip0], dim=1))
        h = self.block_up0(h, global_cond, context, plan_full)
        h = F.gelu(self.out_norm(h))
        return self.out(h).transpose(1, 2)


class Stage2BaseGenerator(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_traj_film: bool = False,
        input_extra_dim: int = 4,
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.history = HistoryEncoder(motion_dim, context_dim)
        self.scene = SceneTokenEncoder3D(1, context_dim)
        self.unet = CondUNet1D(
            CondUNetConfig(
                motion_dim=motion_dim,
                input_extra_dim=input_extra_dim,
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_traj_film=use_traj_film,
            )
        )

    def base_extra(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([batch["history_mask"], batch["target_mask"], batch["valid_mask"]], dim=-1).to(batch["x_t"].dtype)

    def base_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history_tokens = self.history(batch["motion"], batch["history_frames"])
        scene_tokens = self.scene(batch["scene_current"], batch["scene_goal"])
        return torch.cat([history_tokens, scene_tokens], dim=1)

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class MoveWaitGen(Stage2BaseGenerator):
    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        root_plan_frames: int = 15,
        **_: object,
    ) -> None:
        super().__init__(
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_traj_film=True,
            input_extra_dim=3,
        )
        self.root_plan_frames = int(root_plan_frames)

    def plan_full(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_t = batch["x_t"]
        B, T = x_t.shape[:2]
        plan = x_t.new_zeros((B, T, 4))
        h_len = max(1, int(batch["history_frames"].max().item()))
        fill_len = min(int(batch["root_plan"].shape[1]), max(T - h_len, 0))
        if fill_len > 0:
            plan[:, h_len : h_len + fill_len, :3] = batch["root_plan"][:, :fill_len].to(x_t.dtype)
            plan[:, h_len : h_len + fill_len, 3] = batch["root_plan_mask"][:, :fill_len].to(x_t.dtype)
        return plan

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.base_context(batch)
        pred = self.unet(batch["x_t"], self.base_extra(batch), context, diffusion_t, plan_full=self.plan_full(batch))
        return {"x0_hat": pred}


class ActionGen(Stage2BaseGenerator):
    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        text_vocab_size: int = 4096,
        **_: object,
    ) -> None:
        super().__init__(
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_traj_film=False,
            input_extra_dim=4,
        )
        self.text = TextTokenEncoder(context_dim, vocab_size=text_vocab_size)
        self.goals = GoalTokenEncoder(context_dim)

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        time_channel = batch["action_time"].to(batch["x_t"].dtype)[..., None]
        extra = torch.cat([self.base_extra(batch), time_channel], dim=-1)
        context = torch.cat([self.base_context(batch), self.text(batch), self.goals(batch)], dim=1)
        pred = self.unet(batch["x_t"], extra, context, diffusion_t, plan_full=None)
        return {"x0_hat": pred}


class Stage2Generator(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        task: str = "move_wait",
        hidden_dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 4,
        num_timesteps: int = 100,
        dropout: float = 0.1,
        root_plan_frames: int = 15,
        text_vocab_size: int = 4096,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.task = str(task)
        self.num_timesteps = int(num_timesteps)
        cls = MoveWaitGen if self.task == "move_wait" else ActionGen
        self.model = cls(
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            root_plan_frames=root_plan_frames,
            text_vocab_size=text_vocab_size,
            **kwargs,
        )

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(batch, diffusion_t=diffusion_t)
