
#!/usr/bin/env python
"""
Diffusion / Flow-matching actor for Offline IQL with Advantage-Conditioned policy extraction.

This file is designed to *minimize changes* to an existing OfflineIQLPolicy (Gaussian policy)
by only replacing the actor and the actor losses, while keeping:
  - critic / value heads unchanged
  - batch / forward() interface unchanged
  - discrete-action support unchanged (optional argmax via discrete critic)

Core ideas (from RECAP / advantage-conditioned policy extraction):
  * Train a single policy model that can represent:
      - an *unconditional / reference* policy π(a | o)   (adv_token = "null")
      - an *advantage-conditioned* policy π(a | o, I)   (adv_token in {0,1})
    using a surrogate objective for diffusion/flow models (flow-matching MSE).
  * At inference, request "optimality" by setting I=1 and (optionally) using classifier-free guidance (CFG)
    between conditional and unconditional predictions.

Notes:
  - This implementation uses a DiT-style Transformer to predict the *flow velocity* vθ(x_t, t, cond)
    for continuous actions in a continuous-time flow matching setup.
  - Observations are encoded with the existing SACObservationEncoder, then reshaped into tokens:
      token_i ∈ R^{latent_dim} for each image / env / state component.
    Proprioception is naturally one of these tokens (OBS_STATE).
  - The flow is trained in a bounded "model space" (default: actions scaled to [-1, 1] if bounds exist).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

import numpy as np

# ---- LeRobot imports (expected to exist in your codebase) ----
from lerobot.policies.offline.modeling_offline import OfflineIQLPolicy
from lerobot.policies.offline.configuration_offline import OfflineIQLConfig  # base type for hints
from lerobot.policies.sac.modeling_sac import DISCRETE_DIMENSION_INDEX, SACObservationEncoder
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

# You should define and register this config in configuration_offline.py (see response text).
from lerobot.policies.offline.configuration_offline import OfflineIQLDiTFlowAdvConfig  # type: ignore

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for a scalar time t ∈ [0, 1]."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Time embedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: shape [B] or [B, 1], values in [0,1] (or any real; embedding is periodic).

        Returns:
            emb: shape [B, dim]
        """
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.dim() != 1:
            raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")

        half = self.dim // 2
        device = t.device
        # frequencies: exp(-log(max_period) * i/half)
        freq = torch.exp(
            -torch.log(torch.tensor(float(self.max_period), device=device)) * torch.arange(half, device=device) / half
        )
        # outer product [B, half]
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax to turn a conv feature map into K keypoints.

    This is the operation used in "Deep Spatial Autoencoders for Visuomotor Learning" (Finn et al.).

    Given features of shape (B, C, H, W), it computes a softmax over the spatial locations for each
    (learned) channel and returns the expected 2D coordinates (x, y) in normalized image coordinates
    in [-1, 1].

    If `num_kp` is provided, we first learn a 1x1 convolution that maps C channels -> K heatmaps,
    so the output is exactly K keypoints.
    """

    def __init__(self, input_shape: tuple[int, int, int], num_kp: int | None = None):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (C,H,W), got {input_shape}")
        in_c, in_h, in_w = input_shape
        self._in_c, self._in_h, self._in_w = int(in_c), int(in_h), int(in_w)

        if num_kp is not None:
            if num_kp <= 0:
                raise ValueError(f"num_kp must be > 0, got {num_kp}")
            self.nets = nn.Conv2d(self._in_c, int(num_kp), kernel_size=1)
            self._out_c = int(num_kp)
        else:
            self.nets = None
            self._out_c = self._in_c

        # Use numpy to match common implementations exactly.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer('pos_grid', torch.cat([pos_x, pos_y], dim=1), persistent=False)

    def forward(self, features: Tensor) -> Tensor:
        """Compute keypoints.

        Args:
            features: (B, C, H, W)

        Returns:
            keypoints: (B, K, 2)
        """
        if features.dim() != 4:
            raise ValueError(f"features must be 4D (B,C,H,W), got {tuple(features.shape)}")
        if features.shape[-2] != self._in_h or features.shape[-1] != self._in_w:
            raise ValueError(
                f"SpatialSoftmax expected (H,W)=({self._in_h},{self._in_w}), got {tuple(features.shape[-2:])}"
            )

        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B*K, H*W]
        features = features.reshape(-1, self._in_h * self._in_w)
        attn = F.softmax(features, dim=-1)
        # [B*K, H*W] x [H*W, 2] -> [B*K, 2]
        expected_xy = attn @ self.pos_grid
        # -> [B, K, 2]
        return expected_xy.view(-1, self._out_c, 2)


class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm (DiT-style simplification).

    We remove affine parameters from LayerNorm and generate scale/shift from a conditioning vector.
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, D]
            cond: [B, C]

        Returns:
            [B, L, D]
        """
        h = self.norm(x)
        scale_shift = self.to_scale_shift(cond)  # [B, 2D]
        scale, shift = scale_shift.chunk(2, dim=-1)  # each [B, D]
        # Broadcast across sequence length
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return h


class DiTBlock(nn.Module):
    """Transformer block with AdaLN conditioning (time embedding)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.ada_ln1 = AdaLayerNorm(dim, cond_dim=dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ada_ln2 = AdaLayerNorm(dim, cond_dim=dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, D]
            t_emb: [B, D]  (already projected to model dim)

        Returns:
            x: [B, L, D]
        """
        h = self.ada_ln1(x, t_emb)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout1(attn_out)

        h = self.ada_ln2(x, t_emb)
        x = x + self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# DiT + Flow Matching decoder
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """A transformer-style block that updates *query tokens* by cross-attending to conditioning tokens.

    This is closer to a (decoder) transformer than a DiT AdaLN block:
      - Query sequence: action token(s)
      - Key/Value sequence: condition tokens (obs tokens + advantage token)

    Time conditioning is assumed to already be injected into the query tokens (e.g., by addition).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        self.ln_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Forward.

        Args:
            q:  (B, Lq, D) query tokens (action tokens)
            kv: (B, Lc, D) condition tokens (obs tokens + adv token)

        Returns:
            Updated q of shape (B, Lq, D)
        """
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        attn_out, _ = self.cross_attn(qn, kvn, kvn, need_weights=False)
        q = q + self.drop_attn(attn_out)
        q = q + self.mlp(self.ln_mlp(q))
        return q


class DiTFlowDecoder(nn.Module):
    """Cross-attention DiT-like decoder that predicts a flow velocity for a single action.

    Input:
        - action x_t: (B, action_dim)
        - time t: (B,)
        - cond_tokens: (B, N, model_dim)  (obs tokens + advantage token)

    Output:
        - velocity v: (B, action_dim)

    Notes:
        - We do **not** concatenate cond+action into one sequence.
        - The action token(s) are the **query** sequence; cond tokens are **key/value**.
        - Time embedding is injected into the action token only.
    """

    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_cond_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.max_cond_tokens = int(max_cond_tokens)

        self.action_in = nn.Linear(self.action_dim, self.model_dim)

        # Time embedding -> model_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim * 4),
            nn.SiLU(),
            nn.Linear(self.model_dim * 4, self.model_dim),
        )

        # Separate learned positional embeddings for condition and action query.
        self.pos_cond = nn.Parameter(torch.zeros(1, self.max_cond_tokens, self.model_dim))
        self.pos_action = nn.Parameter(torch.zeros(1, 1, self.model_dim))

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(dim=self.model_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.final_ln = nn.LayerNorm(self.model_dim)
        self.action_out = nn.Linear(self.model_dim, self.action_dim)

        # Zero-init for stability (common in diffusion decoders)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, x_t: Tensor, t: Tensor, cond_tokens: Tensor) -> Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.dim() != 1:
            raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")

        # Project action into a single query token
        q = self.action_in(x_t).unsqueeze(1)  # (B, 1, D)

        # Inject time embedding into the query only
        t_emb = self.time_embed(t)  # (B, D)
        q = q + t_emb[:, None, :]

        if cond_tokens.dim() != 3 or cond_tokens.shape[-1] != self.model_dim:
            raise ValueError(
                f"cond_tokens must have shape (B, N, {self.model_dim}), got {tuple(cond_tokens.shape)}"
            )

        N = cond_tokens.shape[1]
        if N > self.max_cond_tokens:
            raise ValueError(
                f"Condition token length {N} exceeds max_cond_tokens={self.max_cond_tokens}. "
                "Increase config.dit_max_tokens / image_tokens_per_camera, or reduce the number of tokens."
            )

        kv = cond_tokens + self.pos_cond[:, :N, :]
        q = q + self.pos_action

        for blk in self.blocks:
            q = blk(q, kv)

        q = self.final_ln(q)
        v = self.action_out(q.squeeze(1))
        return v



# ---------------------------------------------------------------------------
# Flow-matching actor wrapper (keeps interfaces similar to Gaussian Policy)
# ---------------------------------------------------------------------------

class FlowMatchingDiTActor(nn.Module):
    """Flow-matching actor module compatible with OfflineIQLPolicy expectations.

    Key design goals:
      - Minimal interface changes vs. the Gaussian `Policy` class.
      - Still uses **single-step** observation only (current timestep).
      - Supports richer visual conditioning by producing **multiple tokens per image** using SpatialSoftmax.

    Tokenization modes:
      - `image_tokens_per_camera <= 1` (default): keep old behavior by calling SACObservationEncoder and
        chunking its concatenated latent vector into tokens.
      - `image_tokens_per_camera > 1`: for each camera image:
          1) run the shared image encoder to obtain a conv feature map
          2) apply SpatialSoftmax(num_kp=K=image_tokens_per_camera)
          3) treat each keypoint as one token after projection to model_dim
        Then append env token and state token (each 1 token) if present.

    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        action_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        encoder_is_shared: bool = False,
        action_low_bound: Optional[list[float]] = None,
        action_high_bound: Optional[list[float]] = None,
        cfg_scale: float = 1.0,
        num_inference_steps: int = 16,
        sampler: str = 'euler',  # 'euler' or 'heun'
        eps_action: float = 1e-6,
        # --- NEW ---
        image_tokens_per_camera: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_is_shared = bool(encoder_is_shared)

        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)

        self.action_low_bound = action_low_bound
        self.action_high_bound = action_high_bound
        self.eps_action = float(eps_action)

        self.cfg_scale = float(cfg_scale)
        self.num_inference_steps = int(num_inference_steps)
        if sampler not in ('euler', 'heun'):
            raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler}")
        self.sampler = sampler

        # How many tokens to extract per camera image.
        self.image_tokens_per_camera = int(image_tokens_per_camera)
        if self.image_tokens_per_camera < 1:
            raise ValueError(f"image_tokens_per_camera must be >= 1, got {self.image_tokens_per_camera}")

        # Token projection (latent_dim -> model_dim) for env/state tokens and legacy chunking tokens.
        latent_dim = int(encoder.config.latent_dim)
        self.token_in = nn.Linear(latent_dim, self.model_dim)

        # If we are using keypoint tokens, create a SpatialSoftmax per camera and a keypoint projector.
        if self.image_tokens_per_camera > 1:
            if not getattr(self.encoder, 'has_images', False):
                raise ValueError('image_tokens_per_camera>1 requires image observations, but encoder.has_images=False')

            self.image_spatial_softmax = nn.ModuleDict()
            for key in self.encoder.image_keys:
                safe_key = key.replace('.', '_')
                # Infer feature map shape from the encoder's spatial embeddings module.
                # This corresponds to the output of encoder.image_encoder.
                if not hasattr(self.encoder, 'spatial_embeddings'):
                    raise ValueError('Encoder does not have spatial_embeddings; cannot infer feature map shape.')
                if safe_key not in self.encoder.spatial_embeddings:
                    raise KeyError(f"Missing spatial embedding for image key '{key}' (safe='{safe_key}')")
                emb = self.encoder.spatial_embeddings[safe_key]
                fm_shape = (int(getattr(emb, 'channel')), int(getattr(emb, 'height')), int(getattr(emb, 'width')))
                self.image_spatial_softmax[safe_key] = SpatialSoftmax(fm_shape, num_kp=self.image_tokens_per_camera)

            # Project (x,y) keypoint coordinates into model_dim tokens.
            self.kp_token_in = nn.Sequential(
                nn.Linear(2, self.model_dim),
                nn.SiLU(),
                nn.Linear(self.model_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
            )

        # Advantage token embedding:
        #   0: unconditional / null (used for CFG unconditional branch)
        #   1: I = 0 (non-improving)
        #   2: I = 1 (improving)
        self.adv_embed = nn.Embedding(3, self.model_dim)

        # Determine maximum conditioning token count:
        # observation tokens = (#image tokens + env token + state token) and we append advantage token.
        max_obs_tokens = self._count_obs_tokens()
        max_cond_tokens = max_obs_tokens + 1  # + advantage token

        self.decoder = DiTFlowDecoder(
            action_dim=self.action_dim,
            model_dim=self.model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_cond_tokens=max_cond_tokens,
        )

    # --------------------
    # Tokenization
    # --------------------

    def _count_obs_tokens(self) -> int:
        n = 0
        if getattr(self.encoder, 'has_images', False):
            per_cam = self.image_tokens_per_camera if self.image_tokens_per_camera > 1 else 1
            n += len(self.encoder.image_keys) * per_cam
        if getattr(self.encoder, 'has_env', False):
            n += 1
        if getattr(self.encoder, 'has_state', False):
            n += 1
        if n <= 0:
            raise ValueError('FlowMatchingDiTActor requires at least one observation component.')
        return n

    def encode_observation_tokens(
        self,
        observations: dict[str, Tensor],
        observation_features: Optional[dict[str, Tensor]] = None,
        detach_encoder: Optional[bool] = None,
    ) -> Tensor:
        """Encode observations into a token sequence (no advantage token yet).

        Returns:
            obs_tokens: (B, N, model_dim)
        """
        detach = self.encoder_is_shared if detach_encoder is None else bool(detach_encoder)

        # ------------------------------------------------------------------
        # Mode A: legacy chunking (1 token per modality)
        # ------------------------------------------------------------------
        if self.image_tokens_per_camera <= 1:
            if detach:
                # Full detach (including env/state) via no_grad for shared encoder usage.
                with torch.no_grad():
                    obs_vec = self.encoder(observations, cache=observation_features, detach=False)
            else:
                obs_vec = self.encoder(observations, cache=observation_features, detach=False)

            latent_dim = int(self.encoder.config.latent_dim)
            if obs_vec.shape[-1] % latent_dim != 0:
                raise ValueError(
                    f"Encoder output_dim={obs_vec.shape[-1]} is not divisible by latent_dim={latent_dim}. "
                    'Tokenization by chunking cannot proceed.'
                )
            n_tokens = obs_vec.shape[-1] // latent_dim
            tokens = obs_vec.reshape(obs_vec.shape[0], n_tokens, latent_dim)  # (B, N, latent_dim)
            tokens = self.token_in(tokens)  # (B, N, model_dim)
            return tokens

        # ------------------------------------------------------------------
        # Mode B: keypoint tokens per camera image (K tokens / camera)
        # ------------------------------------------------------------------
        if not getattr(self.encoder, 'has_images', False):
            raise ValueError('image_tokens_per_camera>1 but encoder.has_images=False')

        # Get cached conv feature maps (B, C, H, W) for each image key.
        cache = observation_features if isinstance(observation_features, dict) else None
        if cache is None:
            if detach:
                with torch.no_grad():
                    cache = self.encoder.get_cached_image_features(observations)
            else:
                cache = self.encoder.get_cached_image_features(observations)

        tokens_list: list[Tensor] = []

        # Image tokens: concatenate across cameras
        img_tokens_per_cam: list[Tensor] = []
        for key in self.encoder.image_keys:
            safe_key = key.replace('.', '_')
            if key not in cache:
                raise KeyError(f"Missing cached feature map for image key '{key}'.")
            feat_map = cache[key]
            if detach:
                feat_map = feat_map.detach()

            # (B, K, 2) in [-1,1]
            kp_xy = self.image_spatial_softmax[safe_key](feat_map)
            # (B, K, D)
            kp_tok = self.kp_token_in(kp_xy)
            img_tokens_per_cam.append(kp_tok)

        if img_tokens_per_cam:
            tokens_list.append(torch.cat(img_tokens_per_cam, dim=1))  # (B, n_cam*K, D)

        # Env/state tokens (1 each)
        if getattr(self.encoder, 'has_env', False):
            if detach:
                with torch.no_grad():
                    env_lat = self.encoder.env_encoder(observations[OBS_ENV_STATE])
            else:
                env_lat = self.encoder.env_encoder(observations[OBS_ENV_STATE])
            env_tok = self.token_in(env_lat).unsqueeze(1)  # (B, 1, D)
            tokens_list.append(env_tok)

        if getattr(self.encoder, 'has_state', False):
            if detach:
                with torch.no_grad():
                    st_lat = self.encoder.state_encoder(observations[OBS_STATE])
            else:
                st_lat = self.encoder.state_encoder(observations[OBS_STATE])
            st_tok = self.token_in(st_lat).unsqueeze(1)  # (B, 1, D)
            tokens_list.append(st_tok)

        if not tokens_list:
            raise ValueError('No observation tokens were produced. Check your config / inputs.')

        return torch.cat(tokens_list, dim=1)

    def _adv_token(self, adv_index: Tensor) -> Tensor:
        """Build advantage token.

        Args:
            adv_index: (B,) int64 values in {0,1,2}

        Returns:
            token: (B, 1, model_dim)
        """
        if adv_index.dtype != torch.long:
            adv_index = adv_index.long()
        tok = self.adv_embed(adv_index)  # (B, D)
        return tok.unsqueeze(1)

    # --------------------
    # Action scaling
    # --------------------

    def action_to_model_space(self, actions: Tensor) -> Tensor:
        """Map environment-space actions to model space for flow matching."""
        if self.action_low_bound is None or self.action_high_bound is None:
            return actions

        low = torch.as_tensor(self.action_low_bound, device=actions.device, dtype=actions.dtype)
        high = torch.as_tensor(self.action_high_bound, device=actions.device, dtype=actions.dtype)
        scaled = 2.0 * (actions - low) / (high - low) - 1.0
        eps = self.eps_action
        return torch.clamp(scaled, min=-1.0 + eps, max=1.0 - eps)

    def action_from_model_space(self, x: Tensor) -> Tensor:
        """Inverse of action_to_model_space."""
        if self.action_low_bound is None or self.action_high_bound is None:
            return x
        low = torch.as_tensor(self.action_low_bound, device=x.device, dtype=x.dtype)
        high = torch.as_tensor(self.action_high_bound, device=x.device, dtype=x.dtype)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return 0.5 * (x + 1.0) * (high - low) + low

    # --------------------
    # Flow field evaluation
    # --------------------

    def velocity(self, x_t: Tensor, t: Tensor, obs_tokens: Tensor, adv_index: Tensor) -> Tensor:
        """Predict velocity vθ(x_t, t | obs, adv_index)."""
        cond = torch.cat([obs_tokens, self._adv_token(adv_index)], dim=1)
        return self.decoder(x_t, t, cond)

    # --------------------
    # Inference / Sampling
    # --------------------

    @torch.no_grad()
    def sample_action(
        self,
        observations: dict[str, Tensor],
        observation_features: Optional[dict[str, Tensor]] = None,
        *,
        advantage_on: bool = True,
        cfg_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        sampler: Optional[str] = None,
        detach_encoder: Optional[bool] = True,
    ) -> Tensor:
        """Sample one continuous action via ODE integration (flow matching)."""
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        num_steps = self.num_inference_steps if num_steps is None else int(num_steps)
        sampler = self.sampler if sampler is None else sampler
        if sampler not in ('euler', 'heun'):
            raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler}")

        obs_tokens = self.encode_observation_tokens(observations, observation_features, detach_encoder=detach_encoder)

        B = obs_tokens.shape[0]
        device = obs_tokens.device
        x = torch.randn(B, self.action_dim, device=device, dtype=obs_tokens.dtype)

        dt = 1.0 / float(num_steps)
        adv_idx_cond = torch.full((B,), 2 if advantage_on else 1, device=device, dtype=torch.long)
        adv_idx_uncond = torch.zeros((B,), device=device, dtype=torch.long)

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=device, dtype=obs_tokens.dtype)

            if cfg_scale == 1.0:
                v = self.velocity(x, t, obs_tokens, adv_idx_cond)
            else:
                v_u = self.velocity(x, t, obs_tokens, adv_idx_uncond)
                v_c = self.velocity(x, t, obs_tokens, adv_idx_cond)
                v = v_u + cfg_scale * (v_c - v_u)

            if sampler == 'euler':
                x = x + v * dt
            else:
                x_euler = x + v * dt
                t_next = torch.full((B,), min((i + 1.5) * dt, 1.0), device=device, dtype=obs_tokens.dtype)
                if cfg_scale == 1.0:
                    v_next = self.velocity(x_euler, t_next, obs_tokens, adv_idx_cond)
                else:
                    v_u2 = self.velocity(x_euler, t_next, obs_tokens, adv_idx_uncond)
                    v_c2 = self.velocity(x_euler, t_next, obs_tokens, adv_idx_cond)
                    v_next = v_u2 + cfg_scale * (v_c2 - v_u2)
                x = x + 0.5 * (v + v_next) * dt

        return self.action_from_model_space(x)



# ---------------------------------------------------------------------------
# Offline policy wrapper (minimal overrides)
# ---------------------------------------------------------------------------

class OfflineIQLDiTFlowAdvPolicy(OfflineIQLPolicy):
    """
    Offline IQL policy that replaces the Gaussian actor with a DiT + flow-matching actor,
    and replaces the actor loss with an advantage-conditioned flow-matching surrogate.

    Critic/value training remains unchanged from OfflineIQLPolicy.
    """

    config_class = OfflineIQLDiTFlowAdvConfig
    name = "offline_dit_flow_adv"

    def __init__(self, config: OfflineIQLDiTFlowAdvConfig):
        super().__init__(config)
        # Advantage threshold for binarization I = 1{A > threshold}
        self.adv_threshold = float(getattr(config, "adv_threshold", 0.0))

        # Weight on conditional loss term (paper uses alpha for the I-conditioned likelihood term)
        self.adv_cond_alpha = float(getattr(config, "adv_cond_alpha", 1.0))

        # How to compute advantage for indicator: "td" or "qv"
        self.advantage_type: str = getattr(config, "advantage_type", "td")

    # ------------------------------------------------------------------ #
    # Actor initialization
    # ------------------------------------------------------------------ #
    def _init_actor(self, continuous_action_dim: int) -> None:
        cfg = self.config

        # DiT / flow params (add these to config)
        model_dim = int(getattr(cfg, "dit_model_dim", 256))
        num_layers = int(getattr(cfg, "dit_num_layers", 8))
        num_heads = int(getattr(cfg, "dit_num_heads", 8))
        mlp_ratio = float(getattr(cfg, "dit_mlp_ratio", 4.0))
        dropout = float(getattr(cfg, "dit_dropout", 0.0))
        cfg_scale = float(getattr(cfg, "cfg_scale", 1.0))
        num_steps = int(getattr(cfg, "flow_num_inference_steps", 16))
        sampler = str(getattr(cfg, "flow_sampler", "euler"))
        eps_action = float(getattr(cfg, "flow_eps_action", 1e-6))

        image_tokens_per_camera = int(getattr(cfg, "image_tokens_per_camera", 1))

        # Keep bounds interface identical to Gaussian Policy
        policy_kwargs = asdict(cfg.policy_kwargs) if hasattr(cfg, "policy_kwargs") else {}
        action_low = policy_kwargs.get("action_low_bound", None)
        action_high = policy_kwargs.get("action_high_bound", None)

        self.actor = FlowMatchingDiTActor(
            encoder=self.encoder_actor,
            action_dim=continuous_action_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            encoder_is_shared=self.shared_encoder,
            action_low_bound=action_low,
            action_high_bound=action_high,
            cfg_scale=cfg_scale,
            num_inference_steps=num_steps,
            sampler=sampler,
            eps_action=eps_action,
            image_tokens_per_camera=image_tokens_per_camera,
        )

    # ------------------------------------------------------------------ #
    # Advantage indicator
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_advantage_and_indicator(
        self,
        *,
        observations: dict[str, Tensor],
        actions_for_q: Tensor,
        observation_features: Tensor | None,
        next_observations: Optional[dict[str, Tensor]] = None,
        next_observation_features: Tensor | None = None,
        rewards: Optional[Tensor] = None,
        done: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            advantage_raw: [B]
            adv_mask: [B] float in {0,1} (I indicator)
        """
        # V(s)
        v = self.value_forward(
            observations=observations,
            observation_features=observation_features,
            use_target=False,
            detach_encoder=True,
        )

        if self.advantage_type == "qv":
            q = self.critic_ensemble(observations, actions_for_q, observation_features)
            q_min = q.min(dim=0)[0]
            adv = q_min - v
        elif self.advantage_type == "td":
            if next_observations is None or rewards is None or done is None:
                raise ValueError("TD advantage requires next_observations, rewards, and done.")
            v_next = self.value_forward(
                observations=next_observations,
                observation_features=next_observation_features,
                use_target=False,
                detach_encoder=True,
            )
            target = rewards + self.config.discount * (1.0 - done.float()) * v_next
            adv = target - v
        else:
            raise ValueError(f"Unknown advantage_type={self.advantage_type}. Use 'td' or 'qv'.")

        adv_mask = (adv > self.adv_threshold).float()
        return adv, adv_mask

    # ------------------------------------------------------------------ #
    # Actor loss (surrogate for diffusion/flow policy)
    # ------------------------------------------------------------------ #
    def compute_loss_actor(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        observation_features: Tensor | None = None,
        
        next_observations: dict[str, Tensor] | None = None,
        next_observation_features: Tensor | None = None,
        rewards: Tensor | None = None,
        done: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """
        Advantage-conditioned flow-matching surrogate loss.

        Surrogate:
            L = E[ ||vθ(x_t,t | o, null) - (a - ε)||^2  +  α * ||vθ(x_t,t | o, I) - (a - ε)||^2 ]

        where:
            ε ~ N(0,I),  t ~ U(0,1),
            x_t = t * a + (1-t) * ε,
            I = 1{Adv > adv_threshold}
        """
        # Discrete-action handling: only train continuous part
        if self.config.num_discrete_actions is not None:
            actions_cont = actions[:, :DISCRETE_DIMENSION_INDEX]
            actions_for_q = actions_cont
        else:
            actions_cont = actions
            actions_for_q = actions

        # Clamp to support (env-space) before scaling into model space
        actions_cont = self._clamp_actions_to_support(actions_cont)

        # Compute advantage indicator I
        adv_raw, adv_mask = self._compute_advantage_and_indicator(
            observations=observations,
            actions_for_q=actions_for_q,
            observation_features=observation_features,
            next_observations=next_observations,
            next_observation_features=next_observation_features,
            rewards=rewards,
            done=done,
        )

        # Flow matching batch construction
        a = self.actor.action_to_model_space(actions_cont)  # [B, action_dim]
        B = a.shape[0]
        device = a.device
        dtype = a.dtype

        t = torch.rand(B, device=device, dtype=dtype)  # [B]
        eps = torch.randn_like(a)  # [B, action_dim]
        x_t = t[:, None] * a + (1.0 - t)[:, None] * eps
        target = a - eps  # velocity along straight-line coupling

        # Encode observation tokens once
        obs_tokens = self.actor.encode_observation_tokens(
            observations,
            observation_features=observation_features,
            detach_encoder=self.actor.encoder_is_shared,
        )

        # Build adv indices
        adv_idx_uncond = torch.zeros((B,), device=device, dtype=torch.long)  # null
        adv_idx_cond = (1.0 + adv_mask).long()  # 1 (I=0) or 2 (I=1)

        v_uncond = self.actor.velocity(x_t, t, obs_tokens, adv_idx_uncond)
        v_cond = self.actor.velocity(x_t, t, obs_tokens, adv_idx_cond)

        # Per-sample MSE, then mean
        loss_uncond = F.mse_loss(v_uncond, target, reduction="none").mean(dim=-1)
        loss_cond = F.mse_loss(v_cond, target, reduction="none").mean(dim=-1)

        loss = loss_uncond.mean() + self.adv_cond_alpha * loss_cond.mean()

        info = {
            "advantage_mean": adv_raw.mean().item(),
            "adv_mask_fraction": adv_mask.mean().item(),
            "loss_uncond": loss_uncond.mean().item(),
            "loss_cond": loss_cond.mean().item(),
        }
        return loss, info

    def compute_loss_actor_bc(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
    ) -> tuple[Tensor, dict]:
        """
        Behavior-cloning stage for flow actor.

        We still compute I from Q-V (no next_state required) and apply the same
        surrogate as in compute_loss_actor. This trains both unconditional and
        advantage-conditioned branches early.
        """
        if self.config.num_discrete_actions is not None:
            actions_cont = actions[:, :DISCRETE_DIMENSION_INDEX]
            actions_for_q = actions_cont
        else:
            actions_cont = actions
            actions_for_q = actions

        actions_cont = self._clamp_actions_to_support(actions_cont)

        # Use Q-V by default for BC stage (stable; no next_state needed)
        old_adv_type = self.advantage_type
        self.advantage_type = "qv"
        with torch.no_grad():
            adv_raw, adv_mask = self._compute_advantage_and_indicator(
                observations=observations,
                actions_for_q=actions_for_q,
                observation_features=None,
            )
        self.advantage_type = old_adv_type

        a = self.actor.action_to_model_space(actions_cont)
        B = a.shape[0]
        device = a.device
        dtype = a.dtype

        t = torch.rand(B, device=device, dtype=dtype)
        eps = torch.randn_like(a)
        x_t = t[:, None] * a + (1.0 - t)[:, None] * eps
        target = a - eps

        obs_tokens = self.actor.encode_observation_tokens(
            observations,
            observation_features=None,
            detach_encoder=False,  # allow BC to tune encoder if not shared
        )

        adv_idx_uncond = torch.zeros((B,), device=device, dtype=torch.long)
        adv_idx_cond = (1.0 + adv_mask).long()

        v_uncond = self.actor.velocity(x_t, t, obs_tokens, adv_idx_uncond)
        v_cond = self.actor.velocity(x_t, t, obs_tokens, adv_idx_cond)

        loss_uncond = F.mse_loss(v_uncond, target, reduction="none").mean(dim=-1)
        loss_cond = F.mse_loss(v_cond, target, reduction="none").mean(dim=-1)
        loss = loss_uncond.mean() + self.adv_cond_alpha * loss_cond.mean()

        info = {
            "bc_advantage_mean": adv_raw.mean().item(),
            "bc_adv_mask_fraction": adv_mask.mean().item(),
            "bc_loss_uncond": loss_uncond.mean().item(),
            "bc_loss_cond": loss_cond.mean().item(),
        }
        return loss, info

    # ------------------------------------------------------------------ #
    # Inference action selection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Sample action using the advantage-conditioned branch (I=1) and optional CFG.

        This mirrors the original policy's public API.
        """
        # Accept both formats: {state: obs_dict} or direct obs_dict
        observations = batch["state"] if "state" in batch else batch

        observation_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observation_features = self.actor.encoder.get_cached_image_features(observations)

        actions_cont = self.actor.sample_action(
            observations,
            observation_features=observation_features,
            advantage_on=True,
            cfg_scale=getattr(self.config, "cfg_scale", 1.0),
            num_steps=getattr(self.config, "flow_num_inference_steps", 16),
            sampler=getattr(self.config, "flow_sampler", "euler"),
            detach_encoder=True,
        )

        # Discrete action head remains unchanged (argmax from discrete critic)
        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(observations, observation_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions_cont, discrete_action], dim=-1)
        else:
            actions = actions_cont

        return actions
    

    # Mock action distribution for actor server
    def _actor_distribution(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
        **kwargs,
    ):
        """
        Used by the actor server to get (action, log_prob) on-policy.
        """
        actions = self.select_action(observations)

        class MockActionDistribution:
            def __init__(self, actions: Tensor):
                self.actions = actions

            def mode(self) -> Tensor:
                return self.actions

            def log_prob(self, _: Tensor) -> Tensor:
                # scalar log_prob per sample
                B = self.actions.shape[0]
                return torch.zeros((B,), device=self.actions.device, dtype=self.actions.dtype)
        
        return MockActionDistribution(actions), actions



