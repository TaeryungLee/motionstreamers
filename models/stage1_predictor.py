from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(depth - 1, 1)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SceneEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, token_grid: tuple[int, int] = (4, 4)) -> None:
        super().__init__()
        self.token_grid = token_grid
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(token_grid),
        )
        self.pos = nn.Parameter(torch.zeros(token_grid[0] * token_grid[1], hidden_dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, scene_maps: torch.Tensor, goal_map: torch.Tensor) -> torch.Tensor:
        x = torch.cat([scene_maps, goal_map], dim=1)
        x = self.net(x)
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos[None]


class Stage1TrajectoryDenoiser(nn.Module):
    def __init__(
        self,
        slots: int = 4,
        past_frames: int = 12,
        future_frames: int = 90,
        scene_channels: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.slots = int(slots)
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.hidden_dim = int(hidden_dim)

        self.coord_in = MLP(2, hidden_dim, hidden_dim)
        self.vel_in = MLP(2, hidden_dim, hidden_dim)
        self.goal_in = MLP(4, hidden_dim, hidden_dim)
        self.timestep_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.scene_encoder = SceneEncoder(in_channels=scene_channels + 1, hidden_dim=hidden_dim)

        self.entity_embed = nn.Embedding(slots, hidden_dim)
        self.future_time_embed = nn.Embedding(future_frames, hidden_dim)
        self.past_time_embed = nn.Embedding(past_frames, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = MLP(hidden_dim, hidden_dim, 2, depth=2)

    def _entity_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        entity_ids = torch.arange(self.slots, device=device)
        return self.entity_embed(entity_ids)[None].expand(batch_size, -1, -1)

    def forward(self, x_t: torch.Tensor, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> torch.Tensor:
        B, K, T, _ = x_t.shape
        if K != self.slots or T != self.future_frames:
            raise ValueError(f"expected x_t [B,{self.slots},{self.future_frames},2], got {tuple(x_t.shape)}")
        device = x_t.device
        entity_valid = batch["entity_valid"].bool()
        entity_tok = self._entity_tokens(B, device)
        diff_tok = self.timestep_mlp(sinusoidal_embedding(diffusion_t, self.hidden_dim)).view(B, 1, 1, self.hidden_dim)

        future_t = self.future_time_embed(torch.arange(T, device=device)).view(1, 1, T, self.hidden_dim)
        future_tokens = self.coord_in(x_t) + entity_tok[:, :, None] + future_t + diff_tok
        future_tokens_flat = future_tokens.reshape(B, K * T, self.hidden_dim)
        future_valid = entity_valid[:, :, None].expand(B, K, T).reshape(B, K * T)

        past_rel_pos = batch["past_rel_pos"]
        past_vel = batch["past_vel"]
        P = past_rel_pos.shape[2]
        if P != self.past_frames:
            raise ValueError(f"expected {self.past_frames} past frames, got {P}")
        past_t = self.past_time_embed(torch.arange(P, device=device)).view(1, 1, P, self.hidden_dim)
        past_frame_tokens = self.coord_in(past_rel_pos) + self.vel_in(past_vel) + entity_tok[:, :, None] + past_t
        past_tokens = past_frame_tokens.mean(dim=2)

        goal_feat = torch.cat(
            [
                batch["goal_rel_xy"],
                batch["goal_valid"].to(x_t.dtype)[:, None],
                batch["goal_in_crop"].to(x_t.dtype)[:, None],
            ],
            dim=-1,
        )
        goal_tokens = self.goal_in(goal_feat)[:, None]
        scene_tokens = self.scene_encoder(batch["scene_maps"], batch["goal_map"])

        tokens = torch.cat([future_tokens_flat, past_tokens, scene_tokens, goal_tokens], dim=1)
        scene_valid = torch.ones(B, scene_tokens.shape[1], device=device, dtype=torch.bool)
        goal_valid = torch.ones(B, 1, device=device, dtype=torch.bool)
        token_valid = torch.cat([future_valid, entity_valid, scene_valid, goal_valid], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~token_valid)
        future_encoded = encoded[:, : K * T].reshape(B, K, T, self.hidden_dim)
        pred = self.out(self.out_norm(future_encoded))
        return pred * entity_valid[:, :, None, None].to(pred.dtype)


class Stage1Predictor(nn.Module):
    def __init__(
        self,
        slots: int = 4,
        past_frames: int = 12,
        future_frames: int = 90,
        scene_channels: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        num_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        self.denoiser = Stage1TrajectoryDenoiser(
            slots=slots,
            past_frames=past_frames,
            future_frames=future_frames,
            scene_channels=scene_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        betas = torch.linspace(1e-4, 0.02, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0), persistent=False)
        self.register_buffer("posterior_variance", posterior_variance, persistent=False)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp_min(1e-20)), persistent=False)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1, persistent=False)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2, persistent=False)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return values[timesteps].view((target.shape[0],) + (1,) * (target.ndim - 1)).to(dtype=target.dtype)

    def _sample_shape(self, batch: dict[str, torch.Tensor], shape: tuple[int, ...] | None = None) -> tuple[int, int, int, int]:
        if shape is not None:
            if len(shape) != 4:
                raise ValueError(f"Stage1 sample shape must be [B,K,T,2], got {shape}")
            return tuple(int(v) for v in shape)
        if "x0" in batch:
            return tuple(int(v) for v in batch["x0"].shape)
        return (
            int(batch["past_rel_pos"].shape[0]),
            int(batch["past_rel_pos"].shape[1]),
            int(self.denoiser.future_frames),
            2,
        )

    def q_sample(self, x0: torch.Tensor, diffusion_t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, diffusion_t, x0)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, diffusion_t, x0)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def predict_eps_from_x0(self, x_t: torch.Tensor, diffusion_t: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, diffusion_t, x_t) * x_t - x0_hat
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, diffusion_t, x_t).clamp_min(1e-8)

    def forward_denoise(self, x_t: torch.Tensor, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x_t=x_t, batch=batch, diffusion_t=diffusion_t)

    def p_mean_variance(self, x_t: torch.Tensor, batch: dict[str, torch.Tensor], diffusion_t: torch.Tensor) -> dict[str, torch.Tensor]:
        x0_hat = self.forward_denoise(x_t, batch, diffusion_t)
        mean = self._extract(self.posterior_mean_coef1, diffusion_t, x_t) * x0_hat
        mean = mean + self._extract(self.posterior_mean_coef2, diffusion_t, x_t) * x_t
        return {
            "mean": mean,
            "variance": self._extract(self.posterior_variance, diffusion_t, x_t),
            "log_variance": self._extract(self.posterior_log_variance_clipped, diffusion_t, x_t),
            "x0_hat": x0_hat,
        }

    def p_sample(
        self,
        x_t: torch.Tensor,
        batch: dict[str, torch.Tensor],
        diffusion_t: torch.Tensor,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.p_mean_variance(x_t, batch, diffusion_t)
        noise = torch.zeros_like(x_t) if deterministic else torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        nonzero_mask = (diffusion_t != 0).to(x_t.dtype).view((x_t.shape[0],) + (1,) * (x_t.ndim - 1))
        x_prev = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return x_prev, out["x0_hat"]

    def ddim_sample_step(
        self,
        x_t: torch.Tensor,
        batch: dict[str, torch.Tensor],
        diffusion_t: torch.Tensor,
        next_t: torch.Tensor,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0_hat = self.forward_denoise(x_t, batch, diffusion_t)
        eps_hat = self.predict_eps_from_x0(x_t, diffusion_t, x0_hat)
        alpha_t = self._extract(self.alphas_cumprod, diffusion_t, x_t)
        alpha_next = self._extract(self.alphas_cumprod, next_t, x_t)
        sigma = eta * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next).clamp_min(0.0))
        direction = torch.sqrt((1.0 - alpha_next - sigma.square()).clamp_min(0.0)) * eps_hat
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator) if eta > 0.0 else torch.zeros_like(x_t)
        return torch.sqrt(alpha_next) * x0_hat + direction + sigma * noise, x0_hat

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        shape: tuple[int, int, int, int] | None = None,
        num_steps: int | None = None,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        return_all: bool = False,
    ) -> dict[str, torch.Tensor]:
        sample_shape = self._sample_shape(batch, shape)
        device = batch["past_rel_pos"].device
        dtype = batch["past_rel_pos"].dtype
        x_t = torch.randn(sample_shape, device=device, dtype=dtype, generator=generator)
        steps = max(1, min(int(num_steps or self.num_timesteps), self.num_timesteps))
        intermediates: list[torch.Tensor] = []
        last_x0_hat = x_t
        if steps == self.num_timesteps:
            for step in reversed(range(self.num_timesteps)):
                diffusion_t = torch.full((sample_shape[0],), step, device=device, dtype=torch.long)
                x_t, last_x0_hat = self.p_sample(x_t, batch, diffusion_t, generator=generator, deterministic=deterministic)
                if return_all:
                    intermediates.append(last_x0_hat)
        else:
            schedule = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).round().long()
            schedule = torch.unique_consecutive(schedule)
            if schedule[-1] != 0:
                schedule = torch.cat([schedule, torch.zeros(1, device=device, dtype=torch.long)])
            for idx, step in enumerate(schedule):
                if int(step.item()) == 0:
                    diffusion_t = torch.zeros((sample_shape[0],), device=device, dtype=torch.long)
                    last_x0_hat = self.forward_denoise(x_t, batch, diffusion_t)
                    x_t = last_x0_hat
                else:
                    next_step = schedule[idx + 1] if idx + 1 < len(schedule) else torch.zeros((), device=device, dtype=torch.long)
                    diffusion_t = torch.full((sample_shape[0],), int(step.item()), device=device, dtype=torch.long)
                    next_t = torch.full((sample_shape[0],), int(next_step.item()), device=device, dtype=torch.long)
                    x_t, last_x0_hat = self.ddim_sample_step(
                        x_t,
                        batch,
                        diffusion_t,
                        next_t,
                        eta=0.0 if deterministic else 1.0,
                        generator=generator,
                    )
                if return_all:
                    intermediates.append(last_x0_hat)
        out = {"x0_hat": last_x0_hat}
        if return_all:
            out["trajectory"] = torch.stack(intermediates, dim=0)
        return out

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        diffusion_t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x0 = batch["x0"]
        if diffusion_t is None:
            diffusion_t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, diffusion_t, noise)
        x0_hat = self.forward_denoise(x_t, batch, diffusion_t)
        return {"x0_hat": x0_hat, "diffusion_t": diffusion_t, "noise": noise}
