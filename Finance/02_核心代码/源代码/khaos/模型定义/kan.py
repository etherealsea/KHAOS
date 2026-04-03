import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    def __init__(self, input_dim, hidden_dim, depth, grid_size):
        super().__init__()
        self.layers = nn.ModuleList()
        if depth <= 1:
            self.layers.append(KANLinear(input_dim, 1, grid_size=grid_size))
        else:
            self.layers.append(KANLinear(input_dim, hidden_dim, grid_size=grid_size))
            for _ in range(max(0, depth - 2)):
                self.layers.append(KANLinear(hidden_dim, hidden_dim, grid_size=grid_size))
            self.layers.append(KANLinear(hidden_dim, 1, grid_size=grid_size))

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

class KHAOS_KAN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1, layers=2, grid_size=5, num_heads=4):
        super().__init__()
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
        head_depth = max(2, layers)
        self.breakout_head = KANHead(self.d_model, hidden_dim, head_depth, grid_size)
        self.reversion_head = KANHead(self.d_model, hidden_dim, head_depth, grid_size)
        self.aux_head = nn.Sequential(
            nn.Linear(self.d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, return_attn_weights=False, return_aux=False):
        x = self.input_norm(x, mode='norm')
        x = self.proj(x)
        if return_attn_weights:
            x, attn_weights = self.attention_block(x, return_attn_weights=True)
        else:
            x = self.attention_block(x)
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
        breakout_pred = self.breakout_head(breakout_state)
        reversion_pred = self.reversion_head(reversion_state)
        main_pred = torch.cat([breakout_pred, reversion_pred], dim=1)
        aux_pred = self.aux_head(torch.cat([shared_state, pooled - last_state], dim=1))
        if return_attn_weights:
            if return_aux:
                return main_pred, aux_pred, {'attn': attn_weights, 'pool': pool_weights, 'breakout_local_pool': local_weights, 'reversion_local_pool': reversion_weights}
            return main_pred, {'attn': attn_weights, 'pool': pool_weights, 'breakout_local_pool': local_weights, 'reversion_local_pool': reversion_weights}
        if return_aux:
            return main_pred, aux_pred
        return main_pred

    def get_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.breakout_head.regularization_loss(regularize_activation, regularize_entropy) + self.reversion_head.regularization_loss(regularize_activation, regularize_entropy)
