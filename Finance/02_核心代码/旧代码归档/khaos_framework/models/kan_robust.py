import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """
    KAN Linear Layer with B-Spline approximation
    y = sum_i phi_i(x) * w_i
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
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
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
            
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
            noise = (
                (
                    torch.rand(self.grid_size + self.spline_order + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )[:, :, :self.grid_size + self.spline_order] # Fix size mismatch by slicing
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
                
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for x given the grid
        x: (batch_size, in_features)
        grid: (in_features, grid_size + 2 * spline_order + 1)
        Returns: (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
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
        x: (batch_size, in_features)
        y: (batch_size, in_features, out_features)
        """
        # Simplified for initialization, just return permuted y
        return y.permute(2, 1, 0)

    def forward(self, x: torch.Tensor):
        # x: (batch, in)
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1)
        )
        
        if self.enable_standalone_scale_spline:
            # Fix dimensions: 
            # spline_output is [batch, out] (from einsum approximation)
            # scaler is [out, in]
            # We want to scale the contribution of each input spline to each output
            
            # Re-implementing forward simply:
            bases = self.b_splines(x) # [batch, in, grid+order]
            
            # We want out[b,o] = sum_i ( scaler[o,i] * sum_k (bases[b,i,k] * weights[o,i,k]) )
            
            # 1. Compute spline value per input-output pair
            # val[b, o, i] = sum_k (bases[b,i,k] * weights[o,i,k])
            val = torch.einsum("bik,oik->boi", bases, self.spline_weight)
            
            # 2. Apply scaler
            # scaled_val[b, o, i] = val[b, o, i] * scaler[o, i]
            scaled_val = val * self.spline_scaler[None, :, :]
            
            # 3. Sum over inputs
            spline_output = scaled_val.sum(dim=2)

        return base_output + spline_output

class RobustKAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(RobustKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
            
    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
