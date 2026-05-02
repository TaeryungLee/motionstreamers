from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch import ViT
from .original_hsi import OriginalHSIBase


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


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


def _world_from_local(local: torch.Tensor, anchor_root: torch.Tensor, world_to_local: torch.Tensor) -> torch.Tensor:
    offset = torch.stack([anchor_root[:, 0], torch.zeros_like(anchor_root[:, 1]), anchor_root[:, 2]], dim=-1)
    return torch.bmm(local[:, None], world_to_local).squeeze(1) + offset


def _local_from_world(world: np.ndarray, anchor_root: np.ndarray, world_to_local: np.ndarray) -> np.ndarray:
    offset = np.asarray([anchor_root[0], 0.0, anchor_root[2]], dtype=np.float32)
    return ((np.asarray(world, dtype=np.float32) - offset) @ np.asarray(world_to_local, dtype=np.float32).T).astype(np.float32)


def _grid_index(xz: np.ndarray, meta: dict[str, Any]) -> tuple[int, int]:
    x_min = float(meta["x_min"])
    x_max = float(meta["x_max"])
    z_min = float(meta["z_min"])
    z_max = float(meta["z_max"])
    x_res = int(meta["x_res"])
    z_res = int(meta["z_res"])
    ix = int(np.floor((float(xz[0]) - x_min) / max(x_max - x_min, 1e-6) * x_res))
    iz = int(np.floor((float(xz[1]) - z_min) / max(z_max - z_min, 1e-6) * z_res))
    return int(np.clip(ix, 0, x_res - 1)), int(np.clip(iz, 0, z_res - 1))


def _world_xz(index: tuple[int, int], meta: dict[str, Any]) -> np.ndarray:
    ix, iz = index
    x_min = float(meta["x_min"])
    x_max = float(meta["x_max"])
    z_min = float(meta["z_min"])
    z_max = float(meta["z_max"])
    x_res = int(meta["x_res"])
    z_res = int(meta["z_res"])
    x = x_min + (float(ix) + 0.5) / max(float(x_res), 1.0) * (x_max - x_min)
    z = z_min + (float(iz) + 0.5) / max(float(z_res), 1.0) * (z_max - z_min)
    return np.asarray([x, z], dtype=np.float32)


def _nearest_free(obstacle: np.ndarray, node: tuple[int, int], max_radius: int = 64) -> tuple[int, int] | None:
    h, w = obstacle.shape
    sx, sz = int(np.clip(node[0], 0, h - 1)), int(np.clip(node[1], 0, w - 1))
    if not bool(obstacle[sx, sz]):
        return sx, sz
    best: tuple[int, int] | None = None
    best_dist = float("inf")
    for radius in range(1, int(max_radius) + 1):
        x0, x1 = max(0, sx - radius), min(h - 1, sx + radius)
        z0, z1 = max(0, sz - radius), min(w - 1, sz + radius)
        candidates: list[tuple[int, int]] = []
        for x in range(x0, x1 + 1):
            candidates.append((x, z0))
            candidates.append((x, z1))
        for z in range(z0 + 1, z1):
            candidates.append((x0, z))
            candidates.append((x1, z))
        for x, z in candidates:
            if bool(obstacle[x, z]):
                continue
            dist = float((x - sx) ** 2 + (z - sz) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = (x, z)
        if best is not None:
            return best
    return None


def _astar(obstacle: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_runs: int = 250000) -> list[tuple[int, int]] | None:
    h, w = obstacle.shape
    start_free = _nearest_free(obstacle, start, max_radius=max(h, w))
    goal_free = _nearest_free(obstacle, goal, max_radius=max(h, w))
    if start_free is None or goal_free is None:
        return None
    start = start_free
    goal = goal_free
    if start == goal:
        return [start]

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    g_score = np.full((h, w), np.inf, dtype=np.float32)
    came_x = np.full((h, w), -1, dtype=np.int32)
    came_z = np.full((h, w), -1, dtype=np.int32)
    closed = np.zeros((h, w), dtype=np.bool_)
    g_score[start] = 0.0
    heap: list[tuple[float, float, int, int]] = [(heuristic(start, goal), 0.0, start[0], start[1])]
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]
    runs = 0
    while heap and runs < int(max_runs):
        _, current_g, x, z = heapq.heappop(heap)
        if closed[x, z]:
            continue
        closed[x, z] = True
        if (x, z) == goal:
            path = [(x, z)]
            while (x, z) != start:
                px, pz = int(came_x[x, z]), int(came_z[x, z])
                if px < 0 or pz < 0:
                    return None
                x, z = px, pz
                path.append((x, z))
            path.reverse()
            return path
        runs += 1
        for dx, dz, cost in neighbors:
            nx, nz = x + dx, z + dz
            if nx < 0 or nx >= h or nz < 0 or nz >= w or bool(obstacle[nx, nz]) or closed[nx, nz]:
                continue
            candidate_g = current_g + float(cost)
            if candidate_g >= float(g_score[nx, nz]):
                continue
            g_score[nx, nz] = candidate_g
            came_x[nx, nz] = x
            came_z[nx, nz] = z
            heapq.heappush(heap, (candidate_g + heuristic((nx, nz), goal), candidate_g, nx, nz))
    return None


def _resample_path(points: np.ndarray, horizon: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError("empty path")
    if len(points) == 1:
        return np.repeat(points[:1], int(horizon), axis=0)
    seg = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if float(dist[-1]) <= 1e-6:
        return np.repeat(points[:1], int(horizon), axis=0)
    query = np.linspace(0.0, float(dist[-1]), int(horizon) + 1, dtype=np.float32)[1:]
    x = np.interp(query, dist, points[:, 0])
    z = np.interp(query, dist, points[:, 1])
    return np.stack([x, z], axis=-1).astype(np.float32)


def _build_obstacle_xz(occ: np.ndarray, meta: dict[str, Any], margin_m: float = 0.25) -> np.ndarray:
    arr = np.asarray(occ)
    if arr.ndim != 3:
        raise ValueError(f"scene occupancy must be 3D, got {arr.shape}")
    y_min = float(meta["y_min"])
    y_max = float(meta["y_max"])
    y_res = int(meta["y_res"])
    y0 = int(np.floor((0.1 - y_min) / max(y_max - y_min, 1e-6) * y_res))
    y1 = int(np.ceil((1.2 - y_min) / max(y_max - y_min, 1e-6) * y_res))
    y0 = int(np.clip(y0, 0, y_res - 1))
    y1 = int(np.clip(max(y1, y0 + 1), y0 + 1, y_res))
    obstacle = np.any(arr[:, y0:y1, :].astype(bool), axis=1)
    dx = (float(meta["x_max"]) - float(meta["x_min"])) / max(float(meta["x_res"]), 1.0)
    dz = (float(meta["z_max"]) - float(meta["z_min"])) / max(float(meta["z_res"]), 1.0)
    rx = max(1, int(math.ceil(float(margin_m) / max(dx, 1e-6))))
    rz = max(1, int(math.ceil(float(margin_m) / max(dz, 1e-6))))
    tensor = torch.from_numpy(obstacle.astype(np.float32))[None, None]
    dilated = F.max_pool2d(tensor, kernel_size=(2 * rx + 1, 2 * rz + 1), stride=1, padding=(rx, rz))
    return dilated[0, 0].numpy() > 0.5


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

    def _astar_future_trajectory(
        self,
        batch: dict[str, Any],
        current_local: torch.Tensor,
        goal_local: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = current_local.device
        dtype = current_local.dtype
        B = int(current_local.shape[0])
        anchor_root = batch["anchor_root"].to(device=device, dtype=dtype)
        world_to_local = batch["world_to_local"].to(device=device, dtype=dtype)
        current_world = _world_from_local(current_local, anchor_root, world_to_local).detach().cpu().numpy()
        if "body_goal_world" in batch:
            goal_world = batch["body_goal_world"].to(device=device, dtype=dtype).detach().cpu().numpy()
        else:
            goal_world = _world_from_local(goal_local, anchor_root, world_to_local).detach().cpu().numpy()
        anchor_np = anchor_root.detach().cpu().numpy()
        world_to_local_np = world_to_local.detach().cpu().numpy()
        paths = torch.zeros((B, int(horizon), 3), device=device, dtype=dtype)
        valid = torch.zeros((B,), device=device, dtype=torch.bool)
        scene_paths = list(batch.get("scene_occ_path", [""] * B))
        grid_meta = list(batch.get("grid_meta", [{} for _ in range(B)]))
        for i in range(B):
            path_str = str(scene_paths[i]) if i < len(scene_paths) else ""
            meta = grid_meta[i] if i < len(grid_meta) and isinstance(grid_meta[i], dict) else {}
            if not path_str or not meta:
                continue
            try:
                occ = np.load(repo_path(path_str), mmap_mode="r")
                obstacle = _build_obstacle_xz(occ, meta, margin_m=0.25)
                start_idx = _grid_index(current_world[i, [0, 2]], meta)
                goal_idx = _grid_index(goal_world[i, [0, 2]], meta)
                path_idx = _astar(obstacle, start_idx, goal_idx)
                if path_idx is None or len(path_idx) <= 0:
                    continue
                path_xz = np.stack([_world_xz(node, meta) for node in path_idx], axis=0)
                future_xz = _resample_path(path_xz, int(horizon))
                y = np.linspace(float(current_world[i, 1]), float(goal_world[i, 1]), int(horizon), dtype=np.float32)
                future_world = np.stack([future_xz[:, 0], y, future_xz[:, 1]], axis=-1)
                future_local = _local_from_world(future_world, anchor_np[i], world_to_local_np[i])
                paths[i] = torch.as_tensor(future_local, device=device, dtype=dtype)
                valid[i] = True
            except Exception:
                continue
        return paths, valid

    def _prime_from_memory(
        self,
        x: torch.Tensor,
        history: torch.Tensor,
        batch: dict[str, Any],
        scene: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        texts = batch.get("text", [""] * x.shape[0])
        values = x.clone()
        for i in range(int(x.shape[0])):
            retrieved = self.memory.retrieve(str(texts[i] if i < len(texts) else ""), scene[i], text[i])
            if retrieved is None:
                continue
            motion = retrieved.to(device=x.device, dtype=x.dtype)
            if tuple(motion.shape) != tuple(x[i].shape):
                continue
            values[i] = motion * (1.0 - history[i]) + x[i] * history[i]
        return values

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
        x = self._prime_from_memory(x, history, batch, scene, text)
        prev_scene = torch.zeros_like(scene)
        nav_traj, nav_conf = self.navigation(pelvis[:, self.history_frames - 1], goal, text, scene, prev_scene, horizon=T - self.history_frames)
        astar_traj, astar_valid = self._astar_future_trajectory(batch, pelvis[:, self.history_frames - 1], goal, horizon=T - self.history_frames)
        nav_weight = nav_conf[..., None].clamp(0.0, 1.0)
        astar_mask = astar_valid[:, None, None]
        future_traj = torch.where(astar_mask, nav_weight * nav_traj + (1.0 - nav_weight) * astar_traj, nav_traj)
        future_conf = torch.where(astar_valid[:, None], torch.maximum(nav_conf, torch.full_like(nav_conf, 0.5)), nav_conf)
        pad_traj = torch.cat([pelvis[:, : self.history_frames], future_traj], dim=1)
        pad_conf = torch.cat([torch.ones(B, self.history_frames, device=device), future_conf], dim=1)

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
