import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .attention import AttentionResidualBlock
from .revin import RevIN

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            grid_points = self.grid.T[self.spline_order : -self.spline_order]
            num_points = grid_points.shape[0]
            
            noise = (
                (
                    torch.rand(num_points, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    grid_points,
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for x given the grid
        x: (batch, in_features)
        grid: (in_features, grid_size + 2 * spline_order + 1)
        return: (batch, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid
        x = x.unsqueeze(-1)
        
        # Extend the grid to cover inputs that are slightly out of range
        # Or clamp inputs? Let's assume input_norm handles most, but safe clamping is good.
        # But B-splines handle range naturally if grid is wide enough.
        # Our grid is [-1, 1]. Input norm (BatchNorm) makes data ~ N(0, 1).
        # So most data is in [-3, 3]. Grid range [-1, 1] is too small!
        # KAN implementation usually requires grid update or inputs in [-1, 1].
        # We should use Tanh on inputs AFTER batchnorm to force them into [-1, 1].
        
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        x: (batch, in_features)
        y: (batch, in_features, out_features)
        """
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        bases = self.b_splines(x)  # (batch, in, coeff_dim)
        
        if self.enable_standalone_scale_spline:
            scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
            y = torch.einsum("bid,oid->bo", bases, scaled_weight)
        else:
            y = torch.einsum("bid,oid->bo", bases, self.spline_weight)
            
        return base_output + y

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute L1 and Entropy regularization loss for the KAN layer to encourage sparsity.
        """
        l1_fake = self.spline_weight.abs().mean()
        # entropy
        p = self.spline_weight.abs() / (self.spline_weight.abs().sum() + 1e-4)
        entropy = -torch.sum(p * torch.log(p + 1e-4))
        return regularize_activation * l1_fake + regularize_entropy * entropy

class KANHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, grid_size, output_dim=1):
        super().__init__()
        self.layers = nn.ModuleList()
        if depth <= 1:
            self.layers.append(KANLinear(input_dim, output_dim, grid_size=grid_size))
        else:
            self.layers.append(KANLinear(input_dim, hidden_dim, grid_size=grid_size))
            for _ in range(max(0, depth - 2)):
                self.layers.append(KANLinear(hidden_dim, hidden_dim, grid_size=grid_size))
            self.layers.append(KANLinear(hidden_dim, output_dim, grid_size=grid_size))

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        reg_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, 'regularization_loss'):
                reg_loss += layer.regularization_loss(regularize_activation, regularize_entropy)
        return reg_loss

class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class StateMixer(nn.Module):
    def __init__(self, d_model, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.net = nn.Sequential(
            nn.Linear(d_model * num_inputs, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_inputs),
        )

    def forward(self, *states):
        if len(states) != self.num_inputs:
            raise ValueError(f'Expected {self.num_inputs} states, got {len(states)}')
        stacked = torch.stack(states, dim=1)
        mix_logits = self.net(torch.cat(states, dim=1))
        mix_weights = torch.softmax(mix_logits, dim=1)
        fused = torch.sum(stacked * mix_weights.unsqueeze(-1), dim=1)
        return torch.tanh(fused), mix_weights


class KHAOS_KAN(nn.Module):
    def __init__(
        self,
        input_dim=5,
        hidden_dim=16,
        output_dim=1,
        layers=2,
        grid_size=5,
        num_heads=4,
        arch_version='iterA2_base',
        horizon_count=1,
        horizon_family_mode='legacy',
        gate_mode='soft_annealed',
        gate_floor_breakout=0.25,
        gate_floor_reversion=0.35,
        gate_anneal_fraction=0.40,
    ):
        super().__init__()
        self.arch_version = arch_version
        self.horizon_count = max(int(horizon_count), 1)
        self.horizon_family_mode = str(horizon_family_mode or 'legacy')
        self.gate_mode = str(gate_mode or 'soft_annealed')
        self.gate_floor_breakout = float(gate_floor_breakout)
        self.gate_floor_reversion = float(gate_floor_reversion)
        self.gate_anneal_fraction = max(float(gate_anneal_fraction), 1e-6)
        self.gate_progress = 0.0
        self.compression_gate_center = 0.0001
        self.directional_gate_center = 0.05
        self.compression_gate_slope_start = 4.0
        self.compression_gate_slope_end = 16.0
        self.directional_gate_slope_start = 2.0
        self.directional_gate_slope_end = 8.0
        self.multiscale_versions = {'iterA3_multiscale', 'iterA4_multiscale', 'iterA5_multiscale'}
        self.input_norm = RevIN(num_features=input_dim, affine=True)
        self.d_model = input_dim
        if input_dim % num_heads != 0:
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.d_model = hidden_dim
        else:
            self.proj = nn.Identity()
        self.attention_block = AttentionResidualBlock(
            d_model=self.d_model, 
            num_heads=num_heads if self.d_model % num_heads == 0 else 1
        )
        head_depth = max(2, layers)

        if self.arch_version in self.multiscale_versions:
            self.global_pool = AttentionPool(self.d_model)
            self.mid_pool = AttentionPool(self.d_model)
            self.short_pool = AttentionPool(self.d_model)
            self.shared_mixer = StateMixer(self.d_model, 2)
            self.transition_mixer = StateMixer(self.d_model, 3)
            self.reversion_mixer = StateMixer(self.d_model, 3)
            self.bear_mixer = StateMixer(self.d_model, 3)
            self.bull_mixer = StateMixer(self.d_model, 3)
            if self.arch_version == 'iterA5_multiscale':
                self.public_reversion_mixer = StateMixer(self.d_model, 3)
            self.direction_gate = nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, 1),
            )
            self.transition_probe = nn.Sequential(
                nn.Linear(self.d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.reversion_probe = nn.Sequential(
                nn.Linear(self.d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.arch_version in {'iterA4_multiscale', 'iterA5_multiscale'}:
                self.public_reversion_gate = nn.Sequential(
                    nn.Linear(self.d_model * 3, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
                self.breakout_residual_gate = nn.Sequential(
                    nn.Linear(self.d_model * 3, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
            self.breakout_head = KANHead(
                self.d_model, hidden_dim, head_depth, grid_size, output_dim=self.horizon_count * 2
            )
            self.bear_reversion_head = KANHead(
                self.d_model, hidden_dim, head_depth, grid_size, output_dim=self.horizon_count * 2
            )
            self.bull_reversion_head = KANHead(
                self.d_model, hidden_dim, head_depth, grid_size, output_dim=self.horizon_count * 2
            )
            if self.arch_version == 'iterA5_multiscale':
                self.public_reversion_head = KANHead(
                    self.d_model, hidden_dim, head_depth, grid_size, output_dim=self.horizon_count * 2
                )
            self.aux_head = nn.Sequential(
                nn.Linear(self.d_model * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * self.horizon_count)
            )
            if self.horizon_count > 1:
                self.breakout_horizon_head = nn.Sequential(
                    nn.Linear(self.d_model * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.horizon_count),
                )
                self.reversion_horizon_head = nn.Sequential(
                    nn.Linear(self.d_model * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.horizon_count),
                )
        else:
            self.temporal_pool = nn.Linear(self.d_model, 1)
            self.breakout_local_pool = nn.Linear(self.d_model, 1)
            self.reversion_local_pool = nn.Linear(self.d_model, 1)
            self.state_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model)
            )
            self.breakout_local_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model)
            )
            self.reversion_local_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model)
            )
            self.breakout_head = KANHead(
                self.d_model, hidden_dim, head_depth, grid_size, output_dim=self.horizon_count * 2
            )
            self.reversion_head = KANHead(
                self.d_model, hidden_dim, head_depth, grid_size, output_dim=self.horizon_count * 2
            )
            self.aux_head = nn.Sequential(
                nn.Linear(self.d_model * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * self.horizon_count)
            )
            if self.horizon_count > 1:
                self.breakout_horizon_head = nn.Sequential(
                    nn.Linear(self.d_model * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.horizon_count),
                )
                self.reversion_horizon_head = nn.Sequential(
                    nn.Linear(self.d_model * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.horizon_count),
                )

    def _resolve_horizon_prior(self, horizon_prior, batch_size, device, dtype):
        if self.horizon_count <= 1:
            return torch.ones(batch_size, 2, 1, device=device, dtype=dtype)
        if horizon_prior is None:
            return None
        if horizon_prior.dim() == 2:
            horizon_prior = horizon_prior.unsqueeze(1).expand(-1, 2, -1)
        prior = horizon_prior.to(device=device, dtype=dtype)
        denom = prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return prior / denom

    def set_epoch_progress(self, epoch, total_epochs):
        total_epochs = max(int(total_epochs) - 1, 1)
        self.gate_progress = float(np.clip(float(epoch) / float(total_epochs), 0.0, 1.0))

    def _resolve_gate_slope(self, slope_start, slope_end):
        if self.gate_mode != 'soft_annealed':
            return float(slope_end)
        anneal_progress = min(self.gate_progress / self.gate_anneal_fraction, 1.0)
        return float(slope_start + (slope_end - slope_start) * anneal_progress)

    def _build_soft_gate(self, signal, center, slope, gate_floor):
        gate_floor = float(np.clip(gate_floor, 0.0, 1.0))
        if self.gate_mode == 'disabled':
            return torch.ones_like(signal)
        if self.gate_mode == 'legacy_hard':
            return torch.maximum(
                torch.sigmoid(slope * (signal - center)),
                torch.full_like(signal, gate_floor),
            )
        sigmoid_gate = torch.sigmoid(slope * (signal - center))
        return gate_floor + (1.0 - gate_floor) * sigmoid_gate

    def _reshape_aux_logits(self, aux_logits):
        if self.horizon_count <= 1:
            return aux_logits.unsqueeze(-1)
        return aux_logits.view(aux_logits.size(0), 2, self.horizon_count)

    def _aggregate_horizon_outputs(
        self,
        breakout_event_logits,
        reversion_event_logits,
        aux_logits_by_horizon,
        breakout_horizon_logits=None,
        reversion_horizon_logits=None,
        horizon_prior=None,
        family_mode=None,
        valid_horizon_mask=None,
        reversion_extras=None,
    ):
        batch_size = breakout_event_logits.size(0)
        breakout_event_logits = breakout_event_logits.view(batch_size, 2, self.horizon_count)
        reversion_event_logits = reversion_event_logits.view(batch_size, 2, self.horizon_count)
        device = breakout_event_logits.device
        dtype = breakout_event_logits.dtype

        if valid_horizon_mask is not None:
            valid_mask = valid_horizon_mask.to(device=device, dtype=dtype)
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0).expand(batch_size, -1)
            valid_mask = (valid_mask > 0.5).to(dtype=dtype)

        if self.horizon_count <= 1:
            breakout_weights = torch.ones(batch_size, 1, self.horizon_count, device=device, dtype=dtype)
            reversion_weights = torch.ones(batch_size, 1, self.horizon_count, device=device, dtype=dtype)
            breakout_horizon_logits = torch.zeros(batch_size, self.horizon_count, device=device, dtype=dtype)
            reversion_horizon_logits = torch.zeros(batch_size, self.horizon_count, device=device, dtype=dtype)
        else:
            if family_mode == 'single_cycle' and horizon_prior is not None:
                breakout_weights = horizon_prior[:, 0, :].unsqueeze(1)
                reversion_weights = horizon_prior[:, 1, :].unsqueeze(1)
            else:
                if breakout_horizon_logits is None:
                    breakout_horizon_logits = torch.zeros(batch_size, self.horizon_count, device=device, dtype=dtype)
                    reversion_horizon_logits = torch.zeros(batch_size, self.horizon_count, device=device, dtype=dtype)
                if valid_horizon_mask is not None:
                    invalid_fill = torch.finfo(breakout_event_logits.dtype).min
                    breakout_horizon_logits = breakout_horizon_logits.masked_fill(valid_mask <= 0.5, invalid_fill)
                    reversion_horizon_logits = reversion_horizon_logits.masked_fill(valid_mask <= 0.5, invalid_fill)
                breakout_weights = torch.softmax(breakout_horizon_logits, dim=-1).unsqueeze(1)
                reversion_weights = torch.softmax(reversion_horizon_logits, dim=-1).unsqueeze(1)

        breakout_pred = torch.sum(breakout_weights * breakout_event_logits, dim=-1)
        reversion_pred = torch.sum(reversion_weights * reversion_event_logits, dim=-1)
        aux_pred = torch.stack(
            [
                torch.sum(breakout_weights.squeeze(1) * aux_logits_by_horizon[:, 0, :], dim=-1),
                torch.sum(reversion_weights.squeeze(1) * aux_logits_by_horizon[:, 1, :], dim=-1),
            ],
            dim=1,
        )
        info = {
            'breakout_horizon_weights': breakout_weights.squeeze(1),
            'reversion_horizon_weights': reversion_weights.squeeze(1),
            'breakout_horizon_logits': breakout_horizon_logits,
            'reversion_horizon_logits': reversion_horizon_logits,
        }
        if reversion_extras:
            for key, value in reversion_extras.items():
                if value is None:
                    continue
                info[key] = value

        return breakout_pred, reversion_pred, aux_pred, info

    def _forward_itera2(self, x, attn_weights, horizon_prior=None, family_mode=None, valid_horizon_mask=None):
        pool_scores = self.temporal_pool(x).squeeze(-1)
        pool_weights = torch.softmax(pool_scores, dim=1)
        pooled = torch.sum(x * pool_weights.unsqueeze(-1), dim=1)
        last_state = x[:, -1, :]
        gate = torch.sigmoid(self.state_gate(torch.cat([last_state, pooled], dim=1)))
        shared_state = torch.tanh(gate * last_state + (1.0 - gate) * pooled)
        local_len = min(5, x.size(1))
        local_x = x[:, -local_len:, :]
        local_scores = self.breakout_local_pool(local_x).squeeze(-1)
        local_weights = torch.softmax(local_scores, dim=1)
        local_pooled = torch.sum(local_x * local_weights.unsqueeze(-1), dim=1)
        breakout_gate = torch.sigmoid(self.breakout_local_gate(torch.cat([shared_state, local_pooled], dim=1)))
        breakout_state = torch.tanh(breakout_gate * local_pooled + (1.0 - breakout_gate) * shared_state)
        reversion_scores = self.reversion_local_pool(local_x).squeeze(-1)
        reversion_weights = torch.softmax(reversion_scores, dim=1)
        reversion_local_pooled = torch.sum(local_x * reversion_weights.unsqueeze(-1), dim=1)
        reversion_gate = torch.sigmoid(self.reversion_local_gate(torch.cat([shared_state, reversion_local_pooled], dim=1)))
        reversion_state = torch.tanh(reversion_gate * reversion_local_pooled + (1.0 - reversion_gate) * shared_state)
        breakout_event_logits = F.softplus(self.breakout_head(breakout_state))
        reversion_event_logits = F.softplus(self.reversion_head(reversion_state))
        aux_logits_by_horizon = self._reshape_aux_logits(
            self.aux_head(torch.cat([shared_state, pooled - last_state], dim=1))
        )
        breakout_horizon_logits = None
        reversion_horizon_logits = None
        if self.horizon_count > 1:
            breakout_horizon_logits = self.breakout_horizon_head(torch.cat([shared_state, local_pooled], dim=1))
            reversion_horizon_logits = self.reversion_horizon_head(
                torch.cat([shared_state, reversion_local_pooled], dim=1)
            )
        breakout_pred, reversion_pred, aux_pred, horizon_info = self._aggregate_horizon_outputs(
            breakout_event_logits=breakout_event_logits,
            reversion_event_logits=reversion_event_logits,
            aux_logits_by_horizon=aux_logits_by_horizon,
            breakout_horizon_logits=breakout_horizon_logits,
            reversion_horizon_logits=reversion_horizon_logits,
            horizon_prior=horizon_prior,
            family_mode=family_mode,
            valid_horizon_mask=valid_horizon_mask,
            reversion_extras={
                'bear_score': torch.relu(reversion_event_logits),
                'bull_score': torch.relu(reversion_event_logits),
                'public_reversion_score': torch.relu(reversion_event_logits),
                'directional_reversion': torch.relu(reversion_event_logits),
                'directional_floor': torch.relu(reversion_event_logits),
            },
        )
        direction_mix_gate = torch.full_like(breakout_pred, 0.5)
        public_reversion_gate = torch.zeros_like(direction_mix_gate)
        breakout_residual_gate = torch.zeros_like(direction_mix_gate)
        compression_gate = torch.ones(breakout_pred.size(0), 1, device=breakout_pred.device, dtype=breakout_pred.dtype)
        directional_gate = torch.ones_like(horizon_info['directional_floor'])
        gate_floor_hit = torch.zeros_like(directional_gate)
        main_pred = torch.stack([breakout_pred, reversion_pred], dim=1)
        info = {
            'attn': attn_weights,
            'pool': pool_weights,
            'breakout_local_pool': local_weights,
            'reversion_local_pool': reversion_weights,
            'direction_gate': direction_mix_gate,
            'public_reversion_gate': public_reversion_gate,
            'breakout_residual_gate': breakout_residual_gate,
            'compression_gate': compression_gate,
            'directional_gate': directional_gate,
            'gate_floor_hit': gate_floor_hit,
            'bear_score': horizon_info['bear_score'],
            'bull_score': horizon_info['bull_score'],
            'public_reversion_score': horizon_info['public_reversion_score'],
            'directional_reversion': horizon_info['directional_reversion'],
            'directional_floor': horizon_info['directional_floor'],
            'direction_gate_mean': direction_mix_gate.mean(),
            'direction_gate_std': direction_mix_gate.std(unbiased=False),
            'public_reversion_gate_mean': public_reversion_gate.mean(),
            'public_reversion_gate_std': public_reversion_gate.std(unbiased=False),
            'breakout_residual_gate_mean': breakout_residual_gate.mean(),
            'breakout_residual_gate_std': breakout_residual_gate.std(unbiased=False),
            'compression_gate_mean': compression_gate.mean(),
            'compression_gate_std': compression_gate.std(unbiased=False),
            'directional_gate_mean': directional_gate.mean(),
            'directional_gate_std': directional_gate.std(unbiased=False),
            'gate_floor_hit_rate': gate_floor_hit.mean(),
            'directional_floor_mean': horizon_info['directional_floor'].mean(),
            'transition_context_score': torch.sigmoid(breakout_pred),
            'reversion_context_score': torch.sigmoid(reversion_pred),
        }
        info.update(horizon_info)
        return main_pred, aux_pred, info

    def _forward_itera3(self, x, attn_weights, horizon_prior=None, family_mode=None, valid_horizon_mask=None):
        batch_size = x.size(0)
        last_state = x[:, -1, :]
        global_state, global_weights = self.global_pool(x)

        mid_len = min(10, x.size(1))
        short_len = min(5, x.size(1))
        mid_x = x[:, -mid_len:, :]
        short_x = x[:, -short_len:, :]
        mid_state, mid_weights = self.mid_pool(mid_x)
        short_state, short_weights = self.short_pool(short_x)

        shared_state, shared_mix = self.shared_mixer(last_state, global_state)
        transition_context, transition_mix = self.transition_mixer(shared_state, mid_state, short_state)
        reversion_context, reversion_mix = self.reversion_mixer(shared_state, mid_state, short_state)
        bear_state, bear_mix = self.bear_mixer(reversion_context, short_state, last_state)
        bull_state, bull_mix = self.bull_mixer(reversion_context, mid_state, last_state)
        public_reversion_state = None
        public_reversion_mix = None
        if self.arch_version == 'iterA5_multiscale':
            public_reversion_state, public_reversion_mix = self.public_reversion_mixer(
                reversion_context,
                mid_state,
                global_state,
            )

        direction_mix_gate = torch.sigmoid(
            self.direction_gate(
                torch.cat(
                    [reversion_context, short_state - mid_state, last_state - global_state],
                    dim=1,
                )
            )
        )
        breakout_state = transition_context
        breakout_residual_gate = torch.zeros_like(direction_mix_gate)
        if self.arch_version in {'iterA4_multiscale', 'iterA5_multiscale'}:
            breakout_residual_gate = torch.sigmoid(
                self.breakout_residual_gate(
                    torch.cat(
                        [transition_context, short_state - mid_state, last_state - shared_state],
                        dim=1,
                    )
                )
            )
            breakout_state = torch.tanh(transition_context + breakout_residual_gate * (short_state - mid_state))
        breakout_event_logits = F.softplus(self.breakout_head(breakout_state))
        bear_score_h = F.softplus(self.bear_reversion_head(bear_state)).view(batch_size, 2, self.horizon_count)
        bull_score_h = F.softplus(self.bull_reversion_head(bull_state)).view(batch_size, 2, self.horizon_count)
        directional_reversion_h = (
            direction_mix_gate.unsqueeze(-1) * bear_score_h +
            (1.0 - direction_mix_gate.unsqueeze(-1)) * bull_score_h
        )
        directional_floor_h = torch.maximum(
            directional_reversion_h,
            torch.maximum(bear_score_h, bull_score_h) - 0.08,
        )
        public_reversion_score_h = torch.maximum(bear_score_h, bull_score_h)
        if self.arch_version == 'iterA5_multiscale':
            public_reversion_score_h = F.softplus(self.public_reversion_head(public_reversion_state)).view(batch_size, 2, self.horizon_count)
        public_reversion_gate = torch.zeros_like(direction_mix_gate)
        compression_gate = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        directional_gate = torch.ones(batch_size, 2, self.horizon_count, device=x.device, dtype=x.dtype)
        gate_floor_hit = torch.zeros_like(directional_gate)
        
        if self.arch_version in {'iterA4_multiscale', 'iterA5_multiscale'}:
            public_reversion_gate = torch.sigmoid(
                self.public_reversion_gate(
                    torch.cat([
                        reversion_context,
                        (short_state - mid_state).abs(),
                        (last_state - global_state).abs()
                    ], dim=1)
                )
            )
            
            # 【优化修改】：打破严格的凸组合限制，解决 public_below_directional_violation_rate 接近 100% 的问题
            # 将加权平均改为“以 directional_reversion 为基座，向上叠加残差”的方式，
            # 确保 reversion_pred 能够突破 max(bear, bull) 的上限限制，满足 loss.py 中的可行域约束。
            # Ansatz Hard-Wiring: 提取物理特征中的 Compression (index 13)
            # 门控1: 压缩率门控 (Compression Gate) - 突破信号必须在压缩期之后
            compression = x[:, -1, 13]
            directional_separation = torch.abs(bull_score_h - bear_score_h)
            compression_gate = self._build_soft_gate(
                compression.unsqueeze(-1),
                center=self.compression_gate_center,
                slope=self._resolve_gate_slope(
                    self.compression_gate_slope_start,
                    self.compression_gate_slope_end,
                ),
                gate_floor=self.gate_floor_breakout,
            )
            
            # 门控2: 方向一致性门控 (Directional Gate) - 反转或趋势必须有明确的方向底座
            directional_gate = self._build_soft_gate(
                directional_separation,
                center=self.directional_gate_center,
                slope=self._resolve_gate_slope(
                    self.directional_gate_slope_start,
                    self.directional_gate_slope_end,
                ),
                gate_floor=self.gate_floor_reversion,
            )

            # 应用硬接线门控到 Breakout
            # breakout_event_logits 形状为 [batch, 2*horizon]，而 gate 形状为 [batch, 2, horizon]
            breakout_gate = torch.maximum(
                compression_gate.unsqueeze(-1) * directional_gate,
                torch.full_like(directional_gate, self.gate_floor_breakout),
            )
            gate_floor_hit = (
                (breakout_gate <= self.gate_floor_breakout + 1e-4) |
                (directional_gate <= self.gate_floor_reversion + 1e-4)
            ).to(dtype=x.dtype)
            breakout_event_logits = (
                breakout_event_logits.view(batch_size, 2, self.horizon_count)
                * breakout_gate
            ).view(batch_size, -1)

            reversion_residual = public_reversion_gate.unsqueeze(-1) * torch.relu(
                public_reversion_score_h - directional_reversion_h + 0.15
            )

            # Apply the detached gate to the public reversion output.
            # If there is no clear direction (directional_gate_hard ~ 0), the reversion event is strictly bounded by directional_reversion_h.
            gated_residual = directional_gate * reversion_residual
            
            reversion_event_logits = directional_reversion_h + gated_residual
        else:
            reversion_event_logits = directional_reversion_h
        aux_logits_by_horizon = self._reshape_aux_logits(
            self.aux_head(torch.cat([shared_state, transition_context - reversion_context], dim=1))
        )
        breakout_horizon_logits = None
        reversion_horizon_logits = None
        if self.horizon_count > 1:
            breakout_horizon_logits = self.breakout_horizon_head(
                torch.cat([breakout_state, short_state - mid_state, last_state - shared_state], dim=1)
            )
            reversion_horizon_logits = self.reversion_horizon_head(
                torch.cat([reversion_context, (short_state - mid_state).abs(), (last_state - global_state).abs()], dim=1)
            )
        breakout_pred, reversion_pred, aux_pred, horizon_info = self._aggregate_horizon_outputs(
            breakout_event_logits=breakout_event_logits,
            reversion_event_logits=reversion_event_logits,
            aux_logits_by_horizon=aux_logits_by_horizon,
            breakout_horizon_logits=breakout_horizon_logits,
            reversion_horizon_logits=reversion_horizon_logits,
            horizon_prior=horizon_prior,
            family_mode=family_mode,
            valid_horizon_mask=valid_horizon_mask,
            reversion_extras={
                'bear_score': bear_score_h,
                'bull_score': bull_score_h,
                'public_reversion_score': public_reversion_score_h,
                'directional_reversion': directional_reversion_h,
                'directional_floor': directional_floor_h,
            },
        )
        main_pred = torch.stack([breakout_pred, reversion_pred], dim=1)
        info = {
            'attn': attn_weights,
            'global_pool': global_weights,
            'mid_pool': mid_weights,
            'short_pool': short_weights,
            'shared_mix': shared_mix,
            'transition_mix': transition_mix,
            'reversion_mix': reversion_mix,
            'bear_mix': bear_mix,
            'bull_mix': bull_mix,
            'public_reversion_mix': public_reversion_mix,
            'direction_gate': direction_mix_gate,
            'public_reversion_gate': public_reversion_gate,
            'breakout_residual_gate': breakout_residual_gate,
            'compression_gate': compression_gate,
            'directional_gate': directional_gate,
            'gate_floor_hit': gate_floor_hit,
            'bear_score': horizon_info['bear_score'],
            'bull_score': horizon_info['bull_score'],
            'public_reversion_score': horizon_info['public_reversion_score'],
            'directional_reversion': horizon_info['directional_reversion'],
            'directional_floor': horizon_info['directional_floor'],
            'direction_gate_mean': direction_mix_gate.mean(),
            'direction_gate_std': direction_mix_gate.std(unbiased=False),
            'public_reversion_gate_mean': public_reversion_gate.mean(),
            'public_reversion_gate_std': public_reversion_gate.std(unbiased=False),
            'breakout_residual_gate_mean': breakout_residual_gate.mean(),
            'breakout_residual_gate_std': breakout_residual_gate.std(unbiased=False),
            'compression_gate_mean': compression_gate.mean(),
            'compression_gate_std': compression_gate.std(unbiased=False),
            'directional_gate_mean': directional_gate.mean(),
            'directional_gate_std': directional_gate.std(unbiased=False),
            'gate_floor_hit_rate': gate_floor_hit.mean(),
            'directional_floor_mean': horizon_info['directional_floor'].mean(),
            'transition_context_score': torch.sigmoid(self.transition_probe(transition_context)),
            'reversion_context_score': torch.sigmoid(self.reversion_probe(reversion_context)),
        }
        info.update(horizon_info)
        return main_pred, aux_pred, info

    def forward(
        self,
        x,
        return_attn_weights=False,
        return_aux=False,
        return_debug=False,
        horizon_prior=None,
        family_mode=None,
        valid_horizon_mask=None,
    ):
        x = self.input_norm(x, mode='norm')
        x = self.proj(x)
        if return_attn_weights:
            x, attn_weights = self.attention_block(x, return_attn_weights=True)
        else:
            x = self.attention_block(x)
            attn_weights = None
        if self.arch_version in self.multiscale_versions:
            main_pred, aux_pred, info = self._forward_itera3(
                x,
                attn_weights,
                horizon_prior=horizon_prior,
                family_mode=family_mode,
                valid_horizon_mask=valid_horizon_mask,
            )
        else:
            main_pred, aux_pred, info = self._forward_itera2(
                x,
                attn_weights,
                horizon_prior=horizon_prior,
                family_mode=family_mode,
                valid_horizon_mask=valid_horizon_mask,
            )
        if return_attn_weights or return_debug:
            if return_aux:
                return main_pred, aux_pred, info
            return main_pred, info
        if return_aux:
            return main_pred, aux_pred
        return main_pred

    def get_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        reg_loss = self.breakout_head.regularization_loss(regularize_activation, regularize_entropy)
        if self.arch_version in self.multiscale_versions:
            reg_loss += self.bear_reversion_head.regularization_loss(regularize_activation, regularize_entropy)
            reg_loss += self.bull_reversion_head.regularization_loss(regularize_activation, regularize_entropy)
            if self.arch_version == 'iterA5_multiscale':
                reg_loss += self.public_reversion_head.regularization_loss(regularize_activation, regularize_entropy)
            return reg_loss
        return reg_loss + self.reversion_head.regularization_loss(regularize_activation, regularize_entropy)
