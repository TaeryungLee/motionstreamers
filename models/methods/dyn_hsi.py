from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch import ViT
from .original_hsi import OriginalHSIBase


@dataclass
class DynHSIOutputs:
    eps_hat: torch.Tensor
    traj_hat: torch.Tensor
    conf_hat: torch.Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.pe[:length].to(device=device, dtype=dtype)


class DynamicSceneAwareNavigation(nn.Module):
    """Dyn-HSI vision module: 4-layer waypoint/confidence decoder."""

    def __init__(self, dim: int = 32, text_dim: int = 768, nb_voxels: int = 32, depth: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.position = MLP(3, dim, dim)
        self.goal = MLP(3, dim, dim)
        self.text = MLP(text_dim, dim, dim)
        self.scene = ViT(
            image_size=nb_voxels,
            patch_size=8,
            channels=nb_voxels,
            num_classes=dim,
            dim=dim,
            depth=4,
            heads=heads,
            mlp_dim=dim * 2,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.scene_delta = ViT(
            image_size=nb_voxels,
            patch_size=8,
            channels=nb_voxels,
            num_classes=dim,
            dim=dim,
            depth=4,
            heads=heads,
            mlp_dim=dim * 2,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.query_pos = SinusoidalPosition(dim, max_len=512)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.traj_head = nn.Linear(dim, 3)
        self.conf_head = nn.Linear(dim, 1)

    def forward(
        self,
        current_pos: torch.Tensor,
        goal: torch.Tensor,
        text_emb: torch.Tensor,
        local_occ: torch.Tensor,
        prev_local_occ: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = current_pos.shape[0]
        scene_delta = (local_occ - prev_local_occ).to(dtype=local_occ.dtype)
        memory = torch.stack(
            [
                self.position(current_pos),
                self.goal(goal),
                self.text(text_emb.squeeze(1) if text_emb.ndim == 3 else text_emb),
                self.scene(local_occ),
                self.scene_delta(scene_delta),
            ],
            dim=1,
        )
        queries = self.query_pos(int(horizon), current_pos.device, current_pos.dtype).unsqueeze(0).repeat(B, 1, 1)
        hidden = self.decoder(tgt=queries, memory=memory)
        traj = self.traj_head(hidden)
        conf = torch.sigmoid(self.conf_head(hidden)).squeeze(-1)
        return traj, conf


class HierarchicalExperienceMemory:
    """Dyn-HSI Memory module for context-aware noisy motion priming.

    This class is intentionally non-parametric.  Training code can call
    maybe_store(), and inference can call retrieve().  The stored value is noisy
    motion grouped by coarse verb, then ranked by multimodal similarity.
    """

    def __init__(
        self,
        topk_per_key: int = 200,
        loss_threshold: float = 0.001,
        motion_weight: float = 0.1,
        scene_weight: float = 0.4,
        text_weight: float = 0.5,
    ) -> None:
        self.topk_per_key = int(topk_per_key)
        self.loss_threshold = float(loss_threshold)
        self.motion_weight = float(motion_weight)
        self.scene_weight = float(scene_weight)
        self.text_weight = float(text_weight)
        self.bank: dict[str, list[dict[str, torch.Tensor | float | str]]] = {}

    @staticmethod
    def verb(text: str) -> str:
        words = str(text).lower().strip().split()
        return words[0] if words else "unknown"

    def maybe_store(
        self,
        text: str,
        noisy_motion: torch.Tensor,
        scene_feat: torch.Tensor,
        text_feat: torch.Tensor,
        loss: float,
    ) -> None:
        if float(loss) > self.loss_threshold:
            return
        key = self.verb(text)
        item = {
            "motion": noisy_motion.detach().cpu(),
            "scene": scene_feat.detach().cpu(),
            "text": text_feat.detach().cpu(),
            "loss": float(loss),
            "text_raw": str(text),
        }
        bucket = self.bank.setdefault(key, [])
        bucket.append(item)
        bucket.sort(key=lambda value: float(value["loss"]))
        del bucket[self.topk_per_key :]

    def retrieve(self, text: str, scene_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor | None:
        bucket = self.bank.get(self.verb(text), [])
        if not bucket:
            return None
        scene_cpu = scene_feat.detach().cpu().flatten()
        text_cpu = text_feat.detach().cpu().flatten()
        best = None
        best_score = -float("inf")
        for item in bucket:
            s_scene = F.cosine_similarity(scene_cpu, item["scene"].flatten(), dim=0).item()
            s_text = F.cosine_similarity(text_cpu, item["text"].flatten(), dim=0).item()
            score = self.scene_weight * s_scene + self.text_weight * s_text - 0.01 * float(item["loss"])
            if score > best_score:
                best_score = score
                best = item
        return None if best is None else best["motion"]

    def state_dict(self) -> dict[str, Any]:
        return {"bank": self.bank}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.bank = dict(state.get("bank", {})) if isinstance(state, dict) else {}


class DynHSIDiffusionController(nn.Module):
    """Dyn-HSI Control module with condition adapter."""

    def __init__(self, motion_dim: int = 84, dim: int = 512, text_dim: int = 768, nb_voxels: int = 32, heads: int = 16, layers: int = 8) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.dim = int(dim)
        self.input = nn.Linear(motion_dim, dim)
        self.output = nn.Linear(dim, motion_dim)
        self.timestep = MLP(dim, dim, dim)
        self.time_pos = SinusoidalPosition(dim, max_len=512)
        self.scene = ViT(
            image_size=nb_voxels,
            patch_size=8,
            channels=nb_voxels,
            num_classes=dim,
            dim=dim,
            depth=6,
            heads=heads,
            mlp_dim=dim * 2,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.traj = MLP(4, dim, dim)
        self.text = MLP(text_dim, dim, dim)
        self.goal = MLP(3, dim, dim)
        self.adapter = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(inplace=False), nn.Linear(dim, 4), nn.Sigmoid())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.frame_pos = SinusoidalPosition(dim, max_len=512)

    def _time_emb(self, timesteps: torch.Tensor) -> torch.Tensor:
        base = self.time_pos.pe[timesteps].to(device=timesteps.device)
        return self.timestep(base)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        scene_occ: torch.Tensor,
        traj: torch.Tensor,
        traj_conf: torch.Tensor,
        text_emb: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x_t.shape
        t_emb = self._time_emb(timesteps)
        scene_token = self.scene(scene_occ)
        text_token = self.text(text_emb.squeeze(1) if text_emb.ndim == 3 else text_emb)
        goal_token = self.goal(goal)
        traj_frame = self.traj(torch.cat([traj, traj_conf[..., None]], dim=-1))
        traj_frame = traj_frame * traj_conf[..., None].clamp(0.0, 1.0)
        traj_token = traj_frame.mean(dim=1)
        weights = self.adapter(text_token)
        cond = torch.stack(
            [
                scene_token * weights[:, 0:1],
                traj_token * weights[:, 1:2],
                text_token * weights[:, 2:3],
                goal_token * weights[:, 3:4],
            ],
            dim=1,
        )
        cond = cond + t_emb[:, None, :]
        motion = self.input(x_t) * math.sqrt(self.dim)
        motion = motion + self.frame_pos(T, x_t.device, x_t.dtype).unsqueeze(0)
        motion = motion + t_emb[:, None, :] + traj_frame * weights[:, None, 1:2]
        hidden = self.transformer(torch.cat([cond, motion], dim=1))
        return self.output(hidden[:, cond.shape[1] :])


class DynHSIModel(OriginalHSIBase):
    """Paper-structured Dyn-HSI: Vision + Memory + Control."""

    def __init__(self, motion_dim: int = 84, num_timesteps: int = 100, **kwargs: Any) -> None:
        super().__init__(num_timesteps=num_timesteps)
        dim = int(kwargs.get("hidden_dim", 512))
        heads = int(kwargs.get("num_heads", 16))
        layers = int(kwargs.get("num_layers", 8))
        self.window_frames = 48
        self.history_frames = 2
        self.motion_dim = int(motion_dim)
        self.navigation = DynamicSceneAwareNavigation(dim=32, text_dim=768, nb_voxels=32, depth=4, heads=4)
        self.memory = HierarchicalExperienceMemory(topk_per_key=200, loss_threshold=0.001)
        self.controller = DynHSIDiffusionController(motion_dim=motion_dim, dim=dim, text_dim=768, nb_voxels=32, heads=heads, layers=layers)

    def get_extra_state(self) -> dict[str, Any]:
        return {"memory": self.memory.state_dict()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if isinstance(state, dict) and isinstance(state.get("memory"), dict):
            self.memory.load_state_dict(state["memory"])

    @staticmethod
    def _denormalize_motion(x: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        meta = batch["coord_norm_meta"].to(device=x.device, dtype=x.dtype)
        joints = x.reshape(x.shape[0], x.shape[1], 28, 3).clone()
        joints[..., 0] = joints[..., 0] * meta[:, None, None, 0]
        joints[..., 1] = joints[..., 1] * meta[:, None, None, 3] + meta[:, None, None, 2]
        joints[..., 2] = joints[..., 2] * meta[:, None, None, 1]
        return joints.reshape(x.shape[0], x.shape[1], -1)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        x0 = batch["motion"].to(device=device, dtype=torch.float32)
        B, T, _ = x0.shape
        history = torch.zeros((B, T, 1), device=device, dtype=x0.dtype)
        history[:, : self.history_frames] = 1.0
        x = self._randn_like(x0, generator) * (1.0 - history) + x0 * history

        raw = batch["motion_raw"].to(device=device, dtype=torch.float32).reshape(B, T, 28, 3)
        pelvis = raw[:, :, 0]
        goal = batch["body_goal"].to(device=device, dtype=torch.float32)
        text = batch["text_emb"].to(device=device, dtype=torch.float32)
        scene = batch["scene_occ"].to(device=device, dtype=torch.float32)[:, :32]
        prev_scene = torch.zeros_like(scene)
        nav_traj, nav_conf = self.navigation(pelvis[:, self.history_frames - 1], goal, text, scene, prev_scene, horizon=T - self.history_frames)
        pad_traj = torch.cat([pelvis[:, : self.history_frames], nav_traj], dim=1)
        pad_conf = torch.cat([torch.ones(B, self.history_frames, device=device), nav_conf], dim=1)

        terms = self._ddpm_terms(device)
        for step in self._sampling_steps(num_steps, device):
            t = torch.full((B,), int(step.item()), device=device, dtype=torch.long)
            eps = self.controller(x, t, scene, pad_traj, pad_conf, text, goal)
            model_mean = terms["sqrt_recip_alphas"][t].view(B, 1, 1) * (
                x
                - terms["betas"][t].view(B, 1, 1)
                * eps
                / terms["sqrt_one_minus_alphas_cumprod"][t].view(B, 1, 1).clamp_min(1e-6)
            )
            if int(step.item()) == 0:
                x = model_mean
            else:
                noise = self._randn_like(x, generator)
                x = model_mean + torch.sqrt(terms["posterior_variance"][t].view(B, 1, 1).clamp_min(0.0)) * noise
            x = x * (1.0 - history) + x0 * history
        return {"pred_raw": self._denormalize_motion(x, batch), "gt_raw": batch["motion_raw"].to(device=device, dtype=torch.float32)}

    def loss_step(self, batch: dict[str, Any], args: Any) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.parameters()).device
        x0 = batch["motion"].to(device=device, dtype=torch.float32)
        B, T, _ = x0.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        mask = torch.zeros((B, T, 1), device=device, dtype=x0.dtype)
        mask[:, : self.history_frames] = 1.0
        noise = noise * (1.0 - mask)
        x_t = self.q_sample(x0, t, noise)
        x_t = x_t * (1.0 - mask) + x0 * mask

        raw = batch["motion_raw"].to(device=device, dtype=torch.float32).reshape(B, T, 28, 3)
        pelvis = raw[:, :, 0]
        traj_gt = pelvis[:, self.history_frames :]
        goal = batch["body_goal"].to(device=device, dtype=torch.float32)
        text = batch["text_emb"].to(device=device, dtype=torch.float32)
        scene = batch["scene_occ"].to(device=device, dtype=torch.float32)[:, :32]
        prev_scene = torch.zeros_like(scene)
        nav_traj, nav_conf = self.navigation(pelvis[:, self.history_frames - 1], goal, text, scene, prev_scene, horizon=T - self.history_frames)
        traj_err = (nav_traj - traj_gt).square().mean(dim=-1)
        conf_target = torch.exp(-traj_err.detach())
        loss_traj = traj_err.mean()
        loss_conf = F.binary_cross_entropy(nav_conf.clamp(1e-5, 1.0 - 1e-5), conf_target.clamp(0.0, 1.0))

        pad_traj = torch.cat([pelvis[:, : self.history_frames], nav_traj], dim=1)
        pad_conf = torch.cat([torch.ones(B, self.history_frames, device=device), nav_conf], dim=1)
        eps_hat = self.controller(x_t, t, scene, pad_traj, pad_conf, text, goal)
        target_mask = (1.0 - mask).squeeze(-1)
        diff = (eps_hat - noise).square().mean(dim=-1)
        loss_diff = (diff * target_mask).sum() / target_mask.sum().clamp_min(1.0)
        loss = loss_diff + loss_traj + loss_conf
        if self.training:
            with torch.no_grad():
                texts = batch.get("text", [""] * B)
                for i in range(B):
                    self.memory.maybe_store(
                        str(texts[i] if i < len(texts) else ""),
                        x_t[i],
                        scene[i],
                        text[i],
                        float(loss.detach().cpu()),
                    )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "diffusion_loss": float(loss_diff.detach().cpu()),
            "traj_loss": float(loss_traj.detach().cpu()),
            "confidence_loss": float(loss_conf.detach().cpu()),
            "eps_rmse_x1000": float(diff.mean().sqrt().detach().cpu() * 1000.0),
        }
        return loss, metrics
