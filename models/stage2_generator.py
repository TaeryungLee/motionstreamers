from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(int(max_groups), int(channels))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return groups


def framewise_group_norm(x: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
    batch, channels, frames = x.shape
    y = x.permute(0, 2, 1).reshape(batch * frames, channels)
    y = norm(y)
    return y.reshape(batch, frames, channels).permute(0, 2, 1)


def ada_shift_scale(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def resize_temporal(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] == int(length):
        return x
    return F.interpolate(x, size=int(length), mode="linear", align_corners=False)


def pad_temporal_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    frames = x.shape[-1]
    pad = (int(multiple) - (frames % int(multiple))) % int(multiple)
    if pad > 0:
        x = F.pad(x, (0, pad), value=0.0)
    return x, pad


def full_context_mask(tokens: torch.Tensor) -> torch.Tensor:
    return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.pe[timesteps.long()]


class Downsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim_in: int, dim_out: int | None = None) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim_in, dim_out or dim_in, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        zero: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(group_count(out_channels, n_groups), out_channels)
        self.activation = nn.Mish()
        if zero:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = framewise_group_norm(x, self.norm)
        return self.activation(x)


class Conv1DAdaGNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, n_groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(group_count(out_channels, n_groups), out_channels)
        self.activation = nn.Mish()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = framewise_group_norm(x, self.norm)
        x = ada_shift_scale(x, shift, scale)
        return self.activation(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        dropout: float = 0.1,
        adagn: bool = True,
        zero: bool = True,
    ) -> None:
        super().__init__()
        self.adagn = bool(adagn)
        self.block0 = Conv1DAdaGNBlock(in_channels, out_channels, kernel_size, n_groups) if adagn else Conv1DBlock(
            in_channels, out_channels, kernel_size, n_groups
        )
        self.block1 = Conv1DBlock(out_channels, out_channels, kernel_size, n_groups, zero=zero)
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, out_channels * 2 if adagn else out_channels),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        if zero:
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        if self.adagn:
            scale, shift = self.time_mlp(time_emb).chunk(2, dim=-1)
            out = self.block0(x, scale[:, :, None], shift[:, :, None])
        else:
            out = self.block0(x) + self.time_mlp(time_emb)[:, :, None]
        out = self.block1(out)
        out = self.dropout(out)
        return out + self.residual(x)


class LinearCrossAttention(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        heads = max(1, min(int(num_heads), int(channels)))
        while int(channels) % heads != 0 and heads > 1:
            heads -= 1
        self.num_heads = heads
        self.norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(context_dim)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(context_dim, channels)
        self.value = nn.Linear(context_dim, channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        if context.numel() == 0:
            return x
        hidden = x.transpose(1, 2)
        batch, frames, channels = hidden.shape
        tokens = context.shape[1]
        heads = self.num_heads
        query = self.query(self.norm(hidden)).view(batch, frames, heads, -1)
        key = self.key(self.context_norm(context)).view(batch, tokens, heads, -1)
        value = self.value(self.context_norm(context)).view(batch, tokens, heads, -1)
        query = F.softmax(query, dim=-1)
        if context_mask is not None:
            mask = context_mask.to(device=key.device, dtype=torch.bool)
            key = key.masked_fill(~mask[:, :, None, None], -torch.finfo(key.dtype).max)
            value = value * mask[:, :, None, None].to(value.dtype)
        key = F.softmax(key, dim=1)
        attn = self.dropout(torch.einsum("bnhd,bnhe->bhde", key, value))
        out = torch.einsum("bthd,bhde->bthe", query, attn).reshape(batch, frames, channels)
        return x + out.transpose(1, 2)


class FrameWiseAdaBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        frame_cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        zero: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels, n_groups), channels)
        self.cond_proj = nn.Conv1d(frame_cond_dim, channels * 2, kernel_size=1)
        self.activation = nn.Mish()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        if zero:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, frame_cond: torch.Tensor | None) -> torch.Tensor:
        if frame_cond is None:
            return x
        cond = resize_temporal(frame_cond.transpose(1, 2), x.shape[-1])
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        out = framewise_group_norm(x, self.norm)
        out = ada_shift_scale(out, shift, scale)
        out = self.activation(out)
        out = self.conv(out)
        return x + out


class CondConv1DBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_dim: int,
        time_dim: int,
        num_heads: int,
        frame_cond_dim: int | None = None,
        adagn: bool = True,
        zero: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.resblock = ResidualTemporalBlock(
            dim_in,
            dim_out,
            time_dim=time_dim,
            adagn=adagn,
            zero=zero,
            dropout=dropout,
        )
        self.frame_block = FrameWiseAdaBlock(dim_out, frame_cond_dim, zero=zero) if frame_cond_dim is not None else None
        self.cross_attn = LinearCrossAttention(dim_out, context_dim, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        frame_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.resblock(x, time_emb)
        if self.frame_block is not None:
            x = self.frame_block(x, frame_cond)
        return self.cross_attn(x, context, context_mask=context_mask)


class StableMoFusionUNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_dim: int,
        base_dim: int = 512,
        dim_mults: tuple[int, ...] = (2, 2, 2, 2),
        time_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        frame_cond_dim: int | None = None,
        adagn: bool = True,
        zero: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.context_dim = int(context_dim)
        self.base_dim = int(base_dim)
        self.dim_mults = tuple(int(v) for v in dim_mults)
        self.time_dim = int(time_dim)
        self.frame_cond_dim = frame_cond_dim
        dims = [self.input_dim, *[int(self.base_dim * m) for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(
            TimestepEmbedder(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(
                nn.ModuleList(
                    [
                        CondConv1DBlock(
                            dim_in,
                            dim_out,
                            context_dim,
                            self.time_dim,
                            num_heads,
                            frame_cond_dim=frame_cond_dim,
                            adagn=adagn,
                            zero=zero,
                            dropout=dropout,
                        ),
                        CondConv1DBlock(
                            dim_out,
                            dim_out,
                            context_dim,
                            self.time_dim,
                            num_heads,
                            frame_cond_dim=frame_cond_dim,
                            adagn=adagn,
                            zero=zero,
                            dropout=dropout,
                        ),
                        Downsample1D(dim_out),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = CondConv1DBlock(
            mid_dim,
            mid_dim,
            context_dim,
            self.time_dim,
            num_heads,
            frame_cond_dim=frame_cond_dim,
            adagn=adagn,
            zero=zero,
            dropout=dropout,
        )
        self.mid_block2 = CondConv1DBlock(
            mid_dim,
            mid_dim,
            context_dim,
            self.time_dim,
            num_heads,
            frame_cond_dim=frame_cond_dim,
            adagn=adagn,
            zero=zero,
            dropout=dropout,
        )
        last_dim = mid_dim
        for dim_out in reversed(dims[1:]):
            self.ups.append(
                nn.ModuleList(
                    [
                        Upsample1D(last_dim, dim_out),
                        CondConv1DBlock(
                            dim_out * 2,
                            dim_out,
                            context_dim,
                            self.time_dim,
                            num_heads,
                            frame_cond_dim=frame_cond_dim,
                            adagn=adagn,
                            zero=zero,
                            dropout=dropout,
                        ),
                        CondConv1DBlock(
                            dim_out,
                            dim_out,
                            context_dim,
                            self.time_dim,
                            num_heads,
                            frame_cond_dim=frame_cond_dim,
                            adagn=adagn,
                            zero=zero,
                            dropout=dropout,
                        ),
                    ]
                )
            )
            last_dim = dim_out
        self.final_conv = nn.Conv1d(dims[1], self.output_dim, kernel_size=1)
        if zero:
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        extra: torch.Tensor,
        context: torch.Tensor,
        diffusion_t: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        frame_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_frames = x.shape[1]
        x_in = torch.cat([x, extra], dim=-1).transpose(1, 2)
        x_in, pad = pad_temporal_to_multiple(x_in, multiple=16)
        if frame_cond is not None and pad > 0:
            frame_cond = F.pad(frame_cond.transpose(1, 2), (0, pad), value=0.0).transpose(1, 2)
        time_emb = self.time_mlp(diffusion_t)
        skips: list[torch.Tensor] = []
        h = x_in
        for block1, block2, downsample in self.downs:
            h = block1(h, time_emb, context, context_mask, frame_cond)
            h = block2(h, time_emb, context, context_mask, frame_cond)
            skips.append(h)
            h = downsample(h)
        h = self.mid_block1(h, time_emb, context, context_mask, frame_cond)
        h = self.mid_block2(h, time_emb, context, context_mask, frame_cond)
        for upsample, block1, block2 in self.ups:
            h = upsample(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = torch.cat((h, skip), dim=1)
            h = block1(h, time_emb, context, context_mask, frame_cond)
            h = block2(h, time_emb, context, context_mask, frame_cond)
        out = self.final_conv(h)
        return out[:, :, :original_frames].transpose(1, 2)


class SceneTokenEncoderViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        context_dim: int = 256,
        image_size: int = 32,
        patch_size: int = 8,
        depth: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = nn.Conv2d(in_channels, context_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.randn(1, 1, context_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, context_dim) * 0.02)
        enc_heads = max(1, min(int(heads), int(context_dim)))
        while int(context_dim) % enc_heads != 0 and enc_heads > 1:
            enc_heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=enc_heads,
            dim_feedforward=context_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, scene_occ: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(scene_occ).flatten(2).transpose(1, 2)
        cls = self.cls.expand(scene_occ.shape[0], -1, -1).to(tokens.dtype)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]].to(tokens.dtype)
        return self.norm(self.encoder(tokens))


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
        self.body = nn.Sequential(nn.Linear(3, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim))
        self.hand = nn.Sequential(nn.Linear(3, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim))
        self.type_embed = nn.Parameter(torch.randn(2, context_dim) * 0.02)

    def forward(self, batch: dict[str, torch.Tensor], include_hand: bool) -> tuple[torch.Tensor, torch.Tensor]:
        valid = batch["goal_valid"].to(batch["body_goal_cond"].dtype)
        body = self.body(batch["body_goal_cond"]) + self.type_embed[0][None]
        tokens = [body]
        masks = [valid[:, 0] > 0.5]
        if include_hand:
            hand = self.hand(batch["hand_goal_cond"]) + self.type_embed[1][None]
            tokens.append(hand)
            masks.append(valid[:, 1] > 0.5)
        return torch.stack(tokens, dim=1), torch.stack(masks, dim=1)


class T5TextTokenEncoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        model_name: str = "google/mt5-small",
        max_length: int = 64,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only in missing dependency envs
            raise ImportError("T5 text conditioning requires transformers to be installed.") from exc

        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )
        t5_model = AutoModel.from_pretrained(self.model_name, local_files_only=bool(local_files_only))
        self.encoder = t5_model.encoder if hasattr(t5_model, "encoder") else t5_model
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        hidden_dim = int(getattr(t5_model.config, "d_model", getattr(t5_model.config, "hidden_size", 512)))
        self.proj = nn.Linear(hidden_dim, context_dim)
        self.type_embed = nn.Parameter(torch.randn(1, 1, context_dim) * 0.02)

    def build_prompts(self, batch: dict[str, Any]) -> list[str]:
        texts = list(batch.get("text", []))
        goal_types = list(batch.get("goal_type", []))
        prompts: list[str] = []
        for idx, text in enumerate(texts):
            raw = str(text or "").strip()
            goal = str(goal_types[idx] if idx < len(goal_types) else "").strip()
            if raw and goal:
                prompts.append(f"{goal}: {raw}")
            elif raw:
                prompts.append(raw)
            elif goal:
                prompts.append(goal)
            else:
                prompts.append("action")
        return prompts

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch["x_t"].device
        prompts = self.build_prompts(batch)
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            ).last_hidden_state
        tokens = self.proj(hidden.to(self.proj.weight.dtype)).to(batch["x_t"].dtype) + self.type_embed.to(batch["x_t"].dtype)
        mask = encoded["attention_mask"].to(dtype=torch.bool)
        return tokens, mask


class RootPlanFrameEncoder(nn.Module):
    def __init__(self, frame_cond_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, frame_cond_dim),
            nn.Mish(),
            nn.Linear(frame_cond_dim, frame_cond_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_t = batch["x_t"]
        batch_size, total_frames = x_t.shape[:2]
        h_len = max(1, int(batch["history_frames"].max().item()))
        plan = batch["root_plan_cond"].to(dtype=x_t.dtype, device=x_t.device)
        mask = batch["root_plan_mask"].to(dtype=x_t.dtype, device=x_t.device)
        vel = torch.zeros_like(plan)
        if plan.shape[1] > 1:
            vel[:, 1:] = plan[:, 1:] - plan[:, :-1]
        root_cond = x_t.new_zeros((batch_size, total_frames, 7))
        length = min(plan.shape[1], max(total_frames - h_len, 0))
        if length > 0:
            root_cond[:, h_len : h_len + length, :3] = plan[:, :length]
            root_cond[:, h_len : h_len + length, 3:6] = vel[:, :length]
            root_cond[:, h_len : h_len + length, 6] = mask[:, :length]
        return self.net(root_cond)


class Stage2BaseGenerator(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        base_dim: int = 512,
        context_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        input_extra_dim: int = 3,
        frame_cond_dim: int | None = None,
        dim_mults: tuple[int, ...] = (2, 2, 2, 2),
        time_dim: int = 512,
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.base_dim = int(base_dim)
        self.context_dim = int(context_dim)
        self.history = HistoryEncoder(motion_dim, context_dim)
        self.scene = SceneTokenEncoderViT(64, context_dim, heads=num_heads, dropout=dropout)
        self.unet = StableMoFusionUNet1D(
            input_dim=motion_dim + int(input_extra_dim),
            output_dim=motion_dim,
            context_dim=context_dim,
            base_dim=base_dim,
            dim_mults=dim_mults,
            time_dim=time_dim,
            num_heads=num_heads,
            dropout=dropout,
            frame_cond_dim=frame_cond_dim,
        )

    def base_extra(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([batch["history_mask"], batch["target_mask"], batch["valid_mask"]], dim=-1).to(batch["x_t"].dtype)

    def base_context(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        history_tokens = self.history(batch["motion"], batch["history_frames"])
        scene_tokens = self.scene(batch["scene_occ"])
        context = torch.cat([history_tokens, scene_tokens], dim=1)
        return context, full_context_mask(context)

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class MoveWaitGen(Stage2BaseGenerator):
    def __init__(
        self,
        motion_dim: int,
        base_dim: int = 512,
        context_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        frame_cond_dim: int = 256,
        **_: object,
    ) -> None:
        super().__init__(
            motion_dim=motion_dim,
            base_dim=base_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            input_extra_dim=3,
            frame_cond_dim=frame_cond_dim,
        )
        self.root_frame = RootPlanFrameEncoder(frame_cond_dim)
        self.goals = GoalTokenEncoder(context_dim)

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        base_context, base_mask = self.base_context(batch)
        goal_tokens, goal_mask = self.goals(batch, include_hand=False)
        context = torch.cat([base_context, goal_tokens], dim=1)
        context_mask = torch.cat([base_mask, goal_mask.to(base_mask.device)], dim=1)
        frame_cond = self.root_frame(batch)
        pred = self.unet(batch["x_t"], self.base_extra(batch), context, diffusion_t, context_mask=context_mask, frame_cond=frame_cond)
        return {"x0_hat": pred}


class ActionGen(Stage2BaseGenerator):
    def __init__(
        self,
        motion_dim: int,
        base_dim: int = 512,
        context_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        text_model_name: str = "google/mt5-small",
        text_max_length: int = 64,
        text_local_files_only: bool = True,
        **_: object,
    ) -> None:
        super().__init__(
            motion_dim=motion_dim,
            base_dim=base_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            input_extra_dim=3,
            frame_cond_dim=None,
        )
        self.text = T5TextTokenEncoder(
            context_dim,
            model_name=text_model_name,
            max_length=text_max_length,
            local_files_only=text_local_files_only,
        )
        self.goals = GoalTokenEncoder(context_dim)

    def forward(self, batch: dict[str, Any], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        base_context, base_mask = self.base_context(batch)
        text_tokens, text_mask = self.text(batch)
        goal_tokens, goal_mask = self.goals(batch, include_hand=True)
        context = torch.cat([base_context, text_tokens, goal_tokens], dim=1)
        context_mask = torch.cat([base_mask, text_mask.to(base_mask.device), goal_mask], dim=1)
        pred = self.unet(batch["x_t"], self.base_extra(batch), context, diffusion_t, context_mask=context_mask, frame_cond=None)
        return {"x0_hat": pred}


class Stage2Generator(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        task: str = "move_wait",
        hidden_dim: int = 512,
        context_dim: int = 256,
        num_heads: int = 8,
        num_timesteps: int = 100,
        dropout: float = 0.1,
        root_plan_frames: int = 15,
        text_model_name: str = "google/mt5-small",
        text_max_length: int = 64,
        text_local_files_only: bool = True,
        num_layers: int = 8,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.task = str(task)
        self.num_timesteps = int(num_timesteps)
        self.num_layers = int(num_layers)
        self.root_plan_frames = int(root_plan_frames)
        cls = MoveWaitGen if self.task == "move_wait" else ActionGen
        self.model = cls(
            motion_dim=motion_dim,
            base_dim=int(hidden_dim),
            context_dim=int(context_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            text_model_name=text_model_name,
            text_max_length=int(text_max_length),
            text_local_files_only=bool(text_local_files_only),
            **kwargs,
        )

    def forward(self, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(batch, diffusion_t=diffusion_t)
