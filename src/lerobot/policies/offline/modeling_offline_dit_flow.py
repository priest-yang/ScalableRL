
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

# ---- LeRobot imports (expected to exist in your codebase) ----
from lerobot.policies.offline.modeling_offline import OfflineIQLPolicy
from lerobot.policies.offline.configuration_offline import OfflineIQLConfig  # base type for hints
from lerobot.policies.sac.modeling_sac import DISCRETE_DIMENSION_INDEX, SACObservationEncoder
from lerobot.utils.constants import ACTION

# You should define and register this config in configuration_offline.py (see response text).
try:
    from lerobot.policies.offline.configuration_offline import OfflineIQLDiTFlowAdvConfig  # type: ignore
except Exception:  # pragma: no cover
    OfflineIQLDiTFlowAdvConfig = OfflineIQLConfig  # fallback for type checking / docs


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

class DiTFlowDecoder(nn.Module):
    """
    A DiT-like Transformer that predicts the flow velocity for a *single* action token,
    conditioned on observation tokens and an advantage token.

    Input:
        - action x_t: [B, action_dim]
        - time t: [B]
        - cond_tokens: [B, N, cond_dim] (already projected to model dim)
    Output:
        - velocity v: [B, action_dim]
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
        self.action_dim = action_dim
        self.model_dim = model_dim

        self.action_in = nn.Linear(action_dim, model_dim)

        # Time embedding -> model_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )

        # Positional embeddings (learned) for cond + action token
        # Sequence length = max_cond_tokens + 1 action token
        self.max_cond_tokens = max_cond_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, max_cond_tokens + 1, model_dim))

        self.blocks = nn.ModuleList(
            [DiTBlock(dim=model_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(model_dim)
        self.action_out = nn.Linear(model_dim, action_dim)

        # Init to small outputs for stability (optional)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, x_t: Tensor, t: Tensor, cond_tokens: Tensor) -> Tensor:
        """
        Args:
            x_t: [B, action_dim]
            t: [B]
            cond_tokens: [B, N, model_dim]

        Returns:
            v: [B, action_dim]
        """
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.dim() != 1:
            raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")

        B = x_t.shape[0]
        # Project action into a single token
        a_tok = self.action_in(x_t).unsqueeze(1)  # [B, 1, D]

        # Time embedding
        t_emb = self.time_embed(t)  # [B, D]
        # Add time embedding to the action token (common diffusion practice)
        a_tok = a_tok + t_emb[:, None, :]

        # Concatenate conditioning tokens and action token
        seq = torch.cat([cond_tokens, a_tok], dim=1)  # [B, N+1, D]

        # Add positional embeddings (truncate / pad if needed)
        L = seq.shape[1]
        if L > self.max_cond_tokens + 1:
            raise ValueError(
                f"Sequence length {L} exceeds max {self.max_cond_tokens + 1}. "
                "Increase config.dit_max_tokens or reduce input tokens."
            )
        seq = seq + self.pos_emb[:, :L, :]

        for blk in self.blocks:
            seq = blk(seq, t_emb)

        seq = self.final_ln(seq)
        a_final = seq[:, -1, :]  # last token is action
        v = self.action_out(a_final)
        return v


# ---------------------------------------------------------------------------
# Flow-matching actor wrapper (keeps interfaces similar to Gaussian Policy)
# ---------------------------------------------------------------------------

class FlowMatchingDiTActor(nn.Module):
    """
    Actor module compatible with OfflineIQLPolicy expectations:
      - has `encoder` attribute (SACObservationEncoder)
      - has `encoder_is_shared`
      - has action bounds attributes for clamping / scaling
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
        sampler: str = "euler",  # "euler" or "heun"
        eps_action: float = 1e-6,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_is_shared = encoder_is_shared

        self.action_dim = action_dim
        self.model_dim = model_dim

        self.action_low_bound = action_low_bound
        self.action_high_bound = action_high_bound
        self.eps_action = eps_action

        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        if sampler not in ("euler", "heun"):
            raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler}")
        self.sampler = sampler

        # Token projection (latent_dim -> model_dim)
        latent_dim = encoder.config.latent_dim
        self.token_in = nn.Linear(latent_dim, model_dim)

        # Advantage token embedding:
        #   0: unconditional / null (used for CFG unconditional branch)
        #   1: I = 0 (non-improving)
        #   2: I = 1 (improving)
        self.adv_embed = nn.Embedding(3, model_dim)

        # Determine maximum conditioning token count:
        # observation tokens = (#images + env + state) and we append advantage token.
        max_obs_tokens = self._count_obs_tokens()
        max_cond_tokens = max_obs_tokens + 1  # + advantage token
        self.decoder = DiTFlowDecoder(
            action_dim=action_dim,
            model_dim=model_dim,
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
        if getattr(self.encoder, "has_images", False):
            n += len(self.encoder.image_keys)
        if getattr(self.encoder, "has_env", False):
            n += 1
        if getattr(self.encoder, "has_state", False):
            n += 1
        if n <= 0:
            raise ValueError("FlowMatchingDiTActor requires at least one observation component.")
        return n

    def encode_observation_tokens(
        self,
        observations: dict[str, Tensor],
        observation_features: Optional[dict[str, Tensor]] = None,
        detach_encoder: Optional[bool] = None,
    ) -> Tensor:
        """
        Encode observations into a token sequence (no advantage token yet).

        Returns:
            obs_tokens: [B, N, model_dim]
        """
        # Encode using existing encoder (vector) then reshape into tokens
        detach = self.encoder_is_shared if detach_encoder is None else detach_encoder
        obs_vec = self.encoder(observations, cache=observation_features, detach=detach)  # [B, out_dim]

        latent_dim = self.encoder.config.latent_dim
        if obs_vec.shape[-1] % latent_dim != 0:
            raise ValueError(
                f"Encoder output_dim={obs_vec.shape[-1]} is not divisible by latent_dim={latent_dim}. "
                "Tokenization by chunking cannot proceed."
            )
        n_tokens = obs_vec.shape[-1] // latent_dim
        tokens = obs_vec.reshape(obs_vec.shape[0], n_tokens, latent_dim)  # [B, N, latent_dim]
        tokens = self.token_in(tokens)  # [B, N, model_dim]
        return tokens

    def _adv_token(self, adv_index: Tensor) -> Tensor:
        """
        Args:
            adv_index: [B] int64 values in {0,1,2}

        Returns:
            token: [B, 1, model_dim]
        """
        if adv_index.dtype != torch.long:
            adv_index = adv_index.long()
        tok = self.adv_embed(adv_index)  # [B, model_dim]
        return tok.unsqueeze(1)

    # --------------------
    # Action scaling
    # --------------------

    def action_to_model_space(self, actions: Tensor) -> Tensor:
        """
        Map environment-space actions to model space for flow matching.

        If bounds exist, map to (-1,1) via affine scaling (same domain as tanh-squashed Gaussian policies).
        Otherwise, leave as-is (assumes actions already roughly standardized).
        """
        if self.action_low_bound is None or self.action_high_bound is None:
            return actions

        low = torch.as_tensor(self.action_low_bound, device=actions.device, dtype=actions.dtype)
        high = torch.as_tensor(self.action_high_bound, device=actions.device, dtype=actions.dtype)
        # [-1,1] scaling
        scaled = 2.0 * (actions - low) / (high - low) - 1.0
        # keep inside (-1,1) to avoid boundary issues when data came from tanh policies
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

    def velocity(
        self,
        x_t: Tensor,
        t: Tensor,
        obs_tokens: Tensor,
        adv_index: Tensor,
    ) -> Tensor:
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
        """
        Sample one continuous action via ODE integration (flow matching).

        Args:
            advantage_on: If True, use I=1 branch (adv_index=2). If False, use I=0 branch (adv_index=1).
            cfg_scale: guidance scale s. If s==1, only conditional branch is used.
                       If s!=1, also evaluates unconditional branch adv_index=0.
            num_steps: number of Euler/Heun steps.
            sampler: "euler" or "heun".
            detach_encoder: typically True for inference.

        Returns:
            action_env: [B, action_dim]
        """
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        num_steps = self.num_inference_steps if num_steps is None else int(num_steps)
        sampler = self.sampler if sampler is None else sampler
        if sampler not in ("euler", "heun"):
            raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler}")

        # Encode observation tokens once
        obs_tokens = self.encode_observation_tokens(observations, observation_features, detach_encoder=detach_encoder)

        B = obs_tokens.shape[0]
        device = obs_tokens.device
        # initial noise sample x_0 ~ N(0, I)
        x = torch.randn(B, self.action_dim, device=device, dtype=obs_tokens.dtype)

        dt = 1.0 / float(num_steps)
        adv_idx_cond = torch.full((B,), 2 if advantage_on else 1, device=device, dtype=torch.long)
        adv_idx_uncond = torch.zeros((B,), device=device, dtype=torch.long)

        for i in range(num_steps):
            # midpoint time for better stability
            t = torch.full((B,), (i + 0.5) * dt, device=device, dtype=obs_tokens.dtype)

            if cfg_scale == 1.0:
                v = self.velocity(x, t, obs_tokens, adv_idx_cond)
            else:
                v_u = self.velocity(x, t, obs_tokens, adv_idx_uncond)
                v_c = self.velocity(x, t, obs_tokens, adv_idx_cond)
                v = v_u + cfg_scale * (v_c - v_u)

            if sampler == "euler":
                x = x + v * dt
            else:
                # Heun / improved Euler
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
        actions = self.select_action(observations)
        class MockActionDistribution:
            def __init__(self, actions: Tensor):
                self.actions = actions
            def mode(self) -> Tensor:
                return self.actions
            def log_prob(self, _: Tensor) -> Tensor:
                return torch.zeros_like(self.actions)
        return MockActionDistribution(actions), actions
