import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        
        # B-spline logic
        # x shape: [batch, in]
        # grid shape: [in, grid_points]
        
        # Expand grid for batch? No, grid is (in, grid_len)
        # We need to map x to grid range.
        # Assuming x is normalized?
        # KAN usually expects inputs in [-1, 1].
        # We should add a Tanh or LayerNorm before KAN if data is unbounded.
        # But here we assume inputs are normalized.
        
        bases = self.b_splines(x)  # (batch, in, coeff_dim)
        
        # spline_weight: (out, in, coeff_dim)
        # bases: (batch, in, coeff_dim)
        # output: (batch, out)
        
        if self.enable_standalone_scale_spline:
            scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
            y = torch.einsum("bid,oid->bo", bases, scaled_weight)
        else:
            y = torch.einsum("bid,oid->bo", bases, self.spline_weight)
            
        return base_output + y

class KHAOS_KAN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1, layers=2, grid_size=5):
        super().__init__()
        
        # Input Normalization
        # Crucial for Physics features which have small/varying scales
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        self.layers = nn.ModuleList()
        
        # Input Layer
        self.layers.append(KANLinear(input_dim, hidden_dim, grid_size=grid_size))
        
        # Hidden Layers
        for _ in range(layers - 2):
            self.layers.append(KANLinear(hidden_dim, hidden_dim, grid_size=grid_size))
            
        # Output Layer
        if layers > 1:
            self.layers.append(KANLinear(hidden_dim, output_dim, grid_size=grid_size))
            
    def forward(self, x):
        # x: [Batch, Dim]
        
        # Check input dimension for BatchNorm safety
        if x.shape[1] != self.input_norm.num_features:
            # This should ideally not happen if we match config
            # But if it does, we can't use this BatchNorm.
            # print(f"Warning: Input dim {x.shape[1]} != BatchNorm dim {self.input_norm.num_features}")
            # For now, just let it crash or skip normalization?
            # We must crash to debug.
            pass
            
        # Normalize inputs first
        x = self.input_norm(x)
        
        # Force into [-1, 1] for B-Spline grid compatibility
        x = torch.tanh(x)
        
        for layer in self.layers:
            x = layer(x)
        return x
