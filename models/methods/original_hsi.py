from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUMANS_JOINT28_TO_ORIGINAL24 = list(range(22)) + [24, 26]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    previous_utils = sys.modules.get("utils")
    shim = types.ModuleType("utils")
    import numpy as np

    shim.np = np
    shim.torch = torch
    shim.nn = nn
    shim.F = F
    shim.math = math
    shim.transform_points = _transform_points
    shim.extract = _extract
    shim.linear_beta_schedule = lambda timesteps: torch.linspace(1e-4, 2e-2, int(timesteps))
    sys.modules["utils"] = shim
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    return module


_lingo_synhsi = None
_trumans_synhsi = None


def lingo_synhsi():
    global _lingo_synhsi
    if _lingo_synhsi is None:
        _lingo_synhsi = _load_module(
            "worldstreamers_lingo_synhsi",
            PROJECT_ROOT / "lingo-release" / "code" / "models" / "synhsi.py",
        )
    return _lingo_synhsi


def trumans_synhsi():
    global _trumans_synhsi
    if _trumans_synhsi is None:
        _trumans_synhsi = _load_module(
            "worldstreamers_trumans_synhsi",
            PROJECT_ROOT / "trumans" / "models" / "synhsi.py",
        )
    return _trumans_synhsi


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = values.to(device=timesteps.device).gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (target.ndim - 1))).to(device=target.device, dtype=target.dtype)


def _linear_beta_schedule(num_steps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(1e-4, 2e-2, int(num_steps), device=device, dtype=torch.float32)


def _build_schedule(num_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    betas = _linear_beta_schedule(num_steps, device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)


def _transform_points(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    values = x.reshape(x.shape[0], -1, 3)
    values = torch.bmm(values, mat[:, :3, :3].transpose(1, 2)) + mat[:, None, :3, 3]
    return values.reshape(shape)


def _smooth_l1_masked(noise: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inv = torch.logical_not(mask)
    return F.smooth_l1_loss(noise[inv], pred[inv])


class OriginalSceneBatchAdapter:
    """Runtime scene-query object matching the original datasets' sampler API."""

    def __init__(
        self,
        batch: dict[str, Any],
        device: torch.device,
        nb_voxels: int | tuple[int, int, int],
        mesh_grid: tuple[float, float, float, float, float, float],
        min_xyz: torch.Tensor,
        max_xyz: torch.Tensor,
    ) -> None:
        self.device = device
        self.load_scene = True
        self.nb_voxels = nb_voxels
        self.mesh_grid = tuple(float(v) for v in mesh_grid)
        self.min_torch = min_xyz.to(device=device)
        self.max_torch = max_xyz.to(device=device)
        self.scene_paths = [str(path) for path in batch.get("scene_occ_path", [])]
        self.grid_meta = list(batch.get("grid_meta", []))
        self._cache: dict[str, torch.Tensor] = {}

    def normalize_torch(self, data: torch.Tensor) -> torch.Tensor:
        shape = data.shape
        values = data.reshape(-1, 3)
        out = -1.0 + 2.0 * (values - self.min_torch) / (self.max_torch - self.min_torch).clamp_min(1e-6)
        return out.reshape(shape)

    def denormalize_torch(self, data: torch.Tensor) -> torch.Tensor:
        shape = data.shape
        values = data.reshape(-1, 3)
        out = (values + 1.0) * (self.max_torch - self.min_torch) * 0.5 + self.min_torch
        return out.reshape(shape)

    def create_meshgrid(self, batch_size: int = 1) -> torch.Tensor:
        if isinstance(self.nb_voxels, tuple):
            size = self.nb_voxels
        else:
            size = (int(self.nb_voxels), int(self.nb_voxels), int(self.nb_voxels))
        bbox = self.mesh_grid
        x = torch.linspace(bbox[0], bbox[1], size[0], device=self.device)
        y = torch.linspace(bbox[2], bbox[3], size[1], device=self.device)
        z = torch.linspace(bbox[4], bbox[5], size[2], device=self.device)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        return grid.repeat(int(batch_size), 1, 1)

    def _load_scene(self, path: str) -> torch.Tensor | None:
        if not path:
            return None
        if path not in self._cache:
            import numpy as np

            arr = np.asarray(np.load(PROJECT_ROOT / path if not Path(path).is_absolute() else path, mmap_mode="r"), dtype=np.bool_).copy()
            self._cache[path] = torch.as_tensor(arr, device=self.device, dtype=torch.bool)
        return self._cache[path]

    def get_occ_for_points(self, points: torch.Tensor, *args: Any) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        out = torch.ones((batch_size, num_points), device=points.device, dtype=torch.bool)
        flat = points.reshape(batch_size, num_points, 3)
        for b in range(batch_size):
            scene = self._load_scene(self.scene_paths[b] if b < len(self.scene_paths) else "")
            meta = self.grid_meta[b] if b < len(self.grid_meta) else None
            if scene is None or not isinstance(meta, dict):
                continue
            mins = torch.tensor([meta["x_min"], meta["y_min"], meta["z_min"]], device=points.device, dtype=points.dtype)
            maxs = torch.tensor([meta["x_max"], meta["y_max"], meta["z_max"]], device=points.device, dtype=points.dtype)
            res = torch.tensor([meta["x_res"], meta["y_res"], meta["z_res"]], device=points.device, dtype=torch.long)
            vox = torch.floor((flat[b] - mins) / (maxs - mins).clamp_min(1e-6) * res.to(points.dtype)).to(torch.long)
            valid = torch.all((vox >= 0) & (vox < res), dim=-1)
            vox = torch.minimum(torch.maximum(vox, torch.zeros_like(vox)), (res - 1)[None])
            values = scene[vox[:, 0], vox[:, 1], vox[:, 2]]
            values = torch.where(valid, values, torch.ones_like(values, dtype=torch.bool))
            out[b] = values
        return out


class OriginalHSIBase(nn.Module):
    def __init__(self, num_timesteps: int = 100) -> None:
        super().__init__()
        self.num_timesteps = int(num_timesteps)

    def _schedule(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _build_schedule(self.num_timesteps, x.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha, sqrt_one_minus = self._schedule(x_start)
        return _extract(sqrt_alpha, t, x_start) * x_start + _extract(sqrt_one_minus, t, x_start) * noise

    def _ddpm_terms(self, device: torch.device) -> dict[str, torch.Tensor]:
        betas = _linear_beta_schedule(self.num_timesteps, device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        return {
            "betas": betas,
            "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        }

    def _sampling_steps(self, num_steps: int | None, device: torch.device) -> torch.Tensor:
        steps = self.num_timesteps if num_steps is None or int(num_steps) <= 0 else int(num_steps)
        if steps >= self.num_timesteps:
            return torch.arange(self.num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
        values = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).round().long()
        values = torch.unique_consecutive(values)
        if int(values[-1].item()) != 0:
            values = torch.cat([values, torch.zeros(1, device=device, dtype=torch.long)], dim=0)
        return values

    @staticmethod
    def _randn_like(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

    @staticmethod
    def _apply_masked_values(x: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, values, x)

    @staticmethod
    def _min_max_from_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        meta = batch["coord_norm_meta"]
        x_scale = meta[:, 0].max().to(device)
        z_scale = meta[:, 1].max().to(device)
        y_center = meta[:, 2].mean().to(device)
        y_scale = meta[:, 3].max().to(device)
        min_xyz = torch.stack([-x_scale, y_center - y_scale, -z_scale]).to(torch.float32)
        max_xyz = torch.stack([x_scale, y_center + y_scale, z_scale]).to(torch.float32)
        return min_xyz, max_xyz

    @staticmethod
    def _mat(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        world_to_local = batch["world_to_local"].to(device=device, dtype=torch.float32)
        anchor = batch["anchor_root"].to(device=device, dtype=torch.float32)
        mat = torch.eye(4, device=device, dtype=torch.float32).repeat(world_to_local.shape[0], 1, 1)
        mat[:, :3, :3] = world_to_local.transpose(1, 2)
        mat[:, :3, 3] = torch.stack([anchor[:, 0], torch.zeros_like(anchor[:, 1]), anchor[:, 2]], dim=-1)
        return mat


class FaithfulLingoModel(OriginalHSIBase):
    """Original LINGO SynHSI Unet with only input-format adaptation."""

    def __init__(self, motion_dim: int = 84, num_timesteps: int = 100, **_: Any) -> None:
        super().__init__(num_timesteps=num_timesteps)
        synhsi = lingo_synhsi()
        self.net = synhsi.Unet(
            dim_model=512,
            num_heads=16,
            num_layers=8,
            dropout_p=0.1,
            dim_input=int(motion_dim),
            dim_output=int(motion_dim),
            nb_voxels=[32, 32, 32],
            free_p=0.1,
            load_scene=True,
            load_language=True,
            load_hand_goal=True,
            load_pelvis_goal=True,
            language_feature_dim=768,
            scene_type="occ_two",
        )
        self.motion_dim = int(motion_dim)
        self.window_frames = 16
        self.history_frames = 2

    def load_original_checkpoint(self, path: str | Path, map_location: str | torch.device = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.net.load_state_dict(state)

    def _adapter(self, batch: dict[str, Any], device: torch.device) -> OriginalSceneBatchAdapter:
        min_xyz, max_xyz = self._min_max_from_batch(batch, device)
        return OriginalSceneBatchAdapter(
            batch,
            device=device,
            nb_voxels=(32, 32, 32),
            mesh_grid=(-0.6, 0.6, 0.1, 1.2, -0.6, 0.6),
            min_xyz=min_xyz,
            max_xyz=max_xyz,
        )

    def _conditioning_occ(self, batch: dict[str, Any], x_noisy: torch.Tensor, adapter: OriginalSceneBatchAdapter, mat: torch.Tensor) -> torch.Tensor:
        B = x_noisy.shape[0]
        grid = adapter.create_meshgrid(batch_size=B)
        x_orig = _transform_points(adapter.denormalize_torch(x_noisy), mat)
        mat_for_query = mat.clone()
        mat_for_query[:, :3, 3] = x_orig[:, 0, 0:3]
        mat_for_query[:, 1, 3] = 0.0
        query_points = _transform_points(grid, mat_for_query)
        occ = adapter.get_occ_for_points(query_points, None)
        occ = occ.reshape(B, 32, 32, 32).float()

        pelvis_goal = batch["body_goal"].to(device=x_noisy.device, dtype=torch.float32)
        is_loco = batch["is_loco"].to(device=x_noisy.device, dtype=torch.bool)
        pelvis_goal_copy = pelvis_goal.clone()
        norm = torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True).clamp_min(1e-6)
        pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / norm * 0.8
        mat_for_query_goal = mat.clone()
        need_pelvis = batch["need_pelvis_dir"].to(device=x_noisy.device, dtype=torch.bool)
        pelvis_goal_world = _transform_points(pelvis_goal_copy[:, None], mat).squeeze(1)
        mat_for_query_goal[need_pelvis, :3, 3] = pelvis_goal_world[need_pelvis]
        mat_for_query_goal[~need_pelvis, :3, 3] = mat_for_query[~need_pelvis, :3, 3]
        mat_for_query_goal[:, 1, 3] = 0.0
        query_points_goal = _transform_points(grid, mat_for_query_goal)
        occ_goal = adapter.get_occ_for_points(query_points_goal, None)
        occ_goal = occ_goal.reshape(B, 32, 32, 32).float()
        return torch.cat([occ.permute(0, 2, 1, 3), occ_goal.permute(0, 2, 1, 3)], dim=1)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        raw = batch["motion_raw"].to(device=device, dtype=torch.float32)
        adapter = self._adapter(batch, device)
        x_start = adapter.normalize_torch(raw)
        mask = torch.zeros_like(x_start, dtype=torch.bool)
        mask[:, : self.history_frames, :] = True
        x = self._apply_masked_values(self._randn_like(x_start, generator), x_start, mask)
        mat = self._mat(batch, device)
        terms = self._ddpm_terms(device)
        for step in self._sampling_steps(num_steps, device):
            t = torch.full((x.shape[0],), int(step.item()), device=device, dtype=torch.long)
            occ = self._conditioning_occ(batch, x, adapter, mat)
            eps = self.net(
                x,
                occ,
                t,
                batch["text_emb"].to(device=device, dtype=torch.float32),
                batch["body_goal"].to(device=device, dtype=torch.float32),
                batch["hand_goal"].to(device=device, dtype=torch.float32),
                batch["is_pick"].to(device=device, dtype=torch.bool),
                batch["need_scene"].to(device=device, dtype=torch.bool),
                batch["need_pelvis_dir"].to(device=device, dtype=torch.bool),
                batch["pi"].to(device=device, dtype=torch.long),
                batch["need_pi"].to(device=device, dtype=torch.bool),
            )
            model_mean = _extract(terms["sqrt_recip_alphas"], t, x) * (
                x - _extract(terms["betas"], t, x) * eps / _extract(terms["sqrt_one_minus_alphas_cumprod"], t, x).clamp_min(1e-6)
            )
            if int(step.item()) == 0:
                x = model_mean
            else:
                x = model_mean + torch.sqrt(_extract(terms["posterior_variance"], t, x).clamp_min(0.0)) * self._randn_like(x, generator)
            x = self._apply_masked_values(x, x_start, mask)
        return {"pred_raw": adapter.denormalize_torch(x), "gt_raw": raw}

    def loss_step(self, batch: dict[str, Any], args: Any) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.parameters()).device
        raw = batch["motion_raw"].to(device=device, dtype=torch.float32)
        adapter = self._adapter(batch, device)
        x_start = adapter.normalize_torch(raw)
        B = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        mask = torch.zeros_like(x_start, dtype=torch.bool)
        mask[:, : self.history_frames, :] = True
        noise[mask] = 0.0
        x_noisy = self.q_sample(x_start, t, noise)

        mat = self._mat(batch, device)
        with torch.no_grad():
            occ = self._conditioning_occ(batch, x_noisy, adapter, mat)

        pred = self.net(
            x_noisy,
            occ,
            t,
            batch["text_emb"].to(device=device, dtype=torch.float32),
            batch["body_goal"].to(device=device, dtype=torch.float32),
            batch["hand_goal"].to(device=device, dtype=torch.float32),
            batch["is_pick"].to(device=device, dtype=torch.bool),
            batch["need_scene"].to(device=device, dtype=torch.bool),
            batch["need_pelvis_dir"].to(device=device, dtype=torch.bool),
            batch["pi"].to(device=device, dtype=torch.long),
            batch["need_pi"].to(device=device, dtype=torch.bool),
        )
        loss = _smooth_l1_masked(noise, pred, mask)
        metrics = {"loss": float(loss.detach().cpu()), "eps_rmse_x1000": float((pred - noise).square().mean().sqrt().detach().cpu() * 1000.0)}
        return loss, metrics


class FaithfulTrumansModel(OriginalHSIBase):
    """Original TRUMANS/SynHSI body Unet with only input-format adaptation."""

    def __init__(self, motion_dim: int = 72, num_timesteps: int = 100, **_: Any) -> None:
        super().__init__(num_timesteps=num_timesteps)
        synhsi = trumans_synhsi()
        self.net = synhsi.Unet(
            dim_model=512,
            num_heads=16,
            num_layers=8,
            dropout_p=0.1,
            dim_input=int(motion_dim),
            dim_output=int(motion_dim),
            nb_voxels=32,
            free_p=0.0,
            nb_actions=10,
            ac_type="last_add_first_token",
            no_scene=False,
            no_action=False,
        )
        self.motion_dim = int(motion_dim)
        self.window_frames = 16
        self.history_frames = 2

    def load_original_checkpoint(self, path: str | Path, map_location: str | torch.device = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.net.load_state_dict(state)

    def _adapter(self, batch: dict[str, Any], device: torch.device) -> OriginalSceneBatchAdapter:
        min_xyz, max_xyz = self._min_max_from_batch(batch, device)
        return OriginalSceneBatchAdapter(
            batch,
            device=device,
            nb_voxels=32,
            mesh_grid=(-0.6, 0.6, 0.0, 1.2, -0.6, 0.6),
            min_xyz=min_xyz,
            max_xyz=max_xyz,
        )

    def _conditioning_occ(self, x_noisy: torch.Tensor, adapter: OriginalSceneBatchAdapter, mat: torch.Tensor) -> torch.Tensor:
        B = x_noisy.shape[0]
        grid = adapter.create_meshgrid(batch_size=B)
        x_orig = _transform_points(adapter.denormalize_torch(x_noisy), mat)
        mat_for_query = mat.clone()
        mat_for_query[:, :3, 3] = x_orig[:, -1, 0:3]
        mat_for_query[:, 1, 3] = 0.0
        query_points = _transform_points(grid, mat_for_query)
        occ = adapter.get_occ_for_points(query_points, None)
        return occ.reshape(B, 32, 32, 32).float().permute(0, 2, 1, 3)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        raw28 = batch["motion_raw"].to(device=device, dtype=torch.float32).reshape(batch["motion_raw"].shape[0], -1, 28, 3)
        raw24 = raw28[:, :, TRUMANS_JOINT28_TO_ORIGINAL24].reshape(raw28.shape[0], raw28.shape[1], -1)
        adapter = self._adapter(batch, device)
        x_start = adapter.normalize_torch(raw24)
        mask = torch.zeros_like(x_start, dtype=torch.bool)
        mask[:, : self.history_frames, :] = True
        mask[:, -1, 0] = True
        mask[:, -1, 2] = True
        x = self._apply_masked_values(self._randn_like(x_start, generator), x_start, mask)
        mat = self._mat(batch, device)
        terms = self._ddpm_terms(device)
        action_label = batch["action_label"].to(device=device, dtype=torch.float32)
        for step in self._sampling_steps(num_steps, device):
            t = torch.full((x.shape[0],), int(step.item()), device=device, dtype=torch.long)
            occ = self._conditioning_occ(x, adapter, mat)
            eps = self.net(x, occ, t, action_label, mask)
            model_mean = _extract(terms["sqrt_recip_alphas"], t, x) * (
                x - _extract(terms["betas"], t, x) * eps / _extract(terms["sqrt_one_minus_alphas_cumprod"], t, x).clamp_min(1e-6)
            )
            if int(step.item()) == 0:
                x = model_mean
            else:
                x = model_mean + torch.sqrt(_extract(terms["posterior_variance"], t, x).clamp_min(0.0)) * self._randn_like(x, generator)
            x = self._apply_masked_values(x, x_start, mask)

        pred24 = adapter.denormalize_torch(x).reshape(raw28.shape[0], raw28.shape[1], len(TRUMANS_JOINT28_TO_ORIGINAL24), 3)
        pred28 = raw28.clone()
        pred28[:, :, TRUMANS_JOINT28_TO_ORIGINAL24] = pred24
        return {"pred_raw": pred28.reshape(raw28.shape[0], raw28.shape[1], -1), "gt_raw": raw28.reshape(raw28.shape[0], raw28.shape[1], -1)}

    def loss_step(self, batch: dict[str, Any], args: Any) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.parameters()).device
        raw28 = batch["motion_raw"].to(device=device, dtype=torch.float32).reshape(batch["motion_raw"].shape[0], -1, 28, 3)
        raw = raw28[:, :, TRUMANS_JOINT28_TO_ORIGINAL24].reshape(raw28.shape[0], raw28.shape[1], -1)
        adapter = self._adapter(batch, device)
        x_start = adapter.normalize_torch(raw)
        B = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        mask = torch.zeros_like(x_start, dtype=torch.bool)
        mask[:, : self.history_frames, :] = True
        mask[:, -1, 0] = True
        mask[:, -1, 2] = True
        noise[mask] = 0.0
        x_noisy = self.q_sample(x_start, t, noise)

        mat = self._mat(batch, device)
        with torch.no_grad():
            occ = self._conditioning_occ(x_noisy, adapter, mat)

        pred = self.net(
            x_noisy,
            occ,
            t,
            batch["action_label"].to(device=device, dtype=torch.float32),
            mask,
        )
        loss = _smooth_l1_masked(noise, pred, mask)
        metrics = {"loss": float(loss.detach().cpu()), "eps_rmse_x1000": float((pred - noise).square().mean().sqrt().detach().cpu() * 1000.0)}
        return loss, metrics
