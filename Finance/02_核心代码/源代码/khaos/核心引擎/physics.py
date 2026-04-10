import torch
import torch.nn as nn
import torch.nn.functional as F

LOCAL_PHYSICS_WINDOW = 20
SHORT_SMOOTH_WINDOW = 2
PHYSICS_FEATURE_NAMES = [
    'Hurst',
    'Volatility',
    'EKF_Vel',
    'EKF_Res',
    'Entropy',
    'MLE',
    'Price_Mom',
    'EMA_Div',
    'Entropy_Delta',
    'Entropy_Curv',
    'MLE_Delta',
    'EKF_Res_Delta',
    'EMA_Div_Delta',
    'Compression'
]

class DifferentiableEKF(nn.Module):
    def __init__(self, dt=1.0):
        super().__init__()
        self.dt = dt
        self.register_buffer('H_obs', torch.tensor([[1.0, 0.0]]))
        self.register_buffer('I', torch.eye(2))

    def forward(self, price, hurst, vol_sigma):
        B, T = price.shape
        device = price.device
        v0 = (price[:, 1] - price[:, 0]) / self.dt
        x = torch.stack([price[:, 0], v0], dim=1).unsqueeze(2)
        P = torch.eye(2, device=device).unsqueeze(0).repeat(B, 1, 1)
        states = []
        for t in range(T):
            z = price[:, t].view(B, 1, 1)
            h_val = hurst[:, t].view(B, 1, 1)
            sigma = vol_sigma[:, t].view(B, 1, 1)
            rho = 0.5 + 0.5 * torch.sigmoid(10 * (h_val - 0.5))
            F_t = torch.eye(2, device=device).unsqueeze(0).repeat(B, 1, 1)
            F_t[:, 0, 1] = self.dt * rho.squeeze()
            scale = torch.clamp(100 * sigma, min=1.0)
            Q = torch.eye(2, device=device).unsqueeze(0).repeat(B, 1, 1) * scale * 0.1
            R = scale.view(B, 1, 1)
            x_pred = torch.bmm(F_t, x)
            P_pred = torch.bmm(torch.bmm(F_t, P), F_t.transpose(1, 2)) + Q
            H_mat = self.H_obs.unsqueeze(0).repeat(B, 1, 1)
            y_res = z - torch.bmm(H_mat, x_pred)
            S = torch.bmm(torch.bmm(H_mat, P_pred), H_mat.transpose(1, 2)) + R
            S_inv = 1.0 / (S + 1e-6)
            K = torch.bmm(torch.bmm(P_pred, H_mat.transpose(1, 2)), S_inv)
            x = x_pred + torch.bmm(K, y_res)
            I_mat = self.I.unsqueeze(0).repeat(B, 1, 1)
            P = torch.bmm(I_mat - torch.bmm(K, H_mat), P_pred)
            states.append(x.squeeze(2))
        return torch.stack(states, dim=1)

def calculate_hurst_proxy(price, window=LOCAL_PHYSICS_WINDOW):
    pad = window - 1
    price_padded = F.pad(price, (pad, 0), mode='replicate')
    windows = price_padded.unfold(dimension=1, size=window, step=1)
    w_max = windows.max(dim=2)[0]
    w_min = windows.min(dim=2)[0]
    R = w_max - w_min + 1e-8
    S = windows.std(dim=2) + 1e-8
    log_rs = torch.log(R / S)
    log_tau = torch.log(torch.tensor(float(window), device=price.device))
    deviation = log_rs - 0.5 * log_tau
    h_est = 0.5 + 0.3 * torch.tanh((deviation * 20.0) / 10.0)
    return h_est

def calculate_entropy_proxy(high, low, close, window=LOCAL_PHYSICS_WINDOW):
    close_prev = torch.roll(close, shifts=1, dims=1)
    close_prev[:, 0] = close[:, 0]
    tr1 = high - low
    tr2 = torch.abs(high - close_prev)
    tr3 = torch.abs(low - close_prev)
    tr = torch.max(torch.max(tr1, tr2), tr3)
    pad = window - 1
    tr_padded = F.pad(tr.unsqueeze(1), (pad, 0), mode='replicate')
    sum_tr = F.avg_pool1d(tr_padded, kernel_size=window, stride=1).squeeze(1) * window
    high_padded = F.pad(high.unsqueeze(1), (pad, 0), mode='replicate')
    low_padded = F.pad(low.unsqueeze(1), (pad, 0), mode='replicate')
    highest_high = F.max_pool1d(high_padded, kernel_size=window, stride=1).squeeze(1)
    lowest_low = -F.max_pool1d(-low_padded, kernel_size=window, stride=1).squeeze(1)
    rng = highest_high - lowest_low + 1e-8
    return torch.log10((sum_tr / rng) + 1e-8)

def ema_smooth(x, window):
    alpha = 2.0 / (window + 1.0)
    ema = torch.zeros_like(x)
    ema[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        ema[:, t] = alpha * x[:, t] + (1 - alpha) * ema[:, t - 1]
    return ema

# 注意：这里的 MLE 代表 Maximum Lyapunov Exponent (最大李雅普诺夫指数)，
# 用于衡量系统对初始条件的敏感性（混沌发散度）。
# 切勿与 Maximum Likelihood Estimation (最大似然估计) 混淆。
def calculate_lyapunov_proxy(log_returns, window=LOCAL_PHYSICS_WINDOW):
    eps = 1e-8
    r_t_minus_1 = torch.roll(log_returns, shifts=1, dims=1)
    r_t_minus_1[:, 0] = log_returns[:, 0]
    local_divergence = torch.log((torch.abs(log_returns) + eps) / (torch.abs(r_t_minus_1) + eps))
    return ema_smooth(local_divergence, window)

def _diff_feature(x):
    return torch.diff(x, dim=1, prepend=x[:, :1])

def _compute_core_features(open_price, high_price, low_price, close, volume, ema20):
    log_close = torch.log(close + 1e-8)
    h_t = calculate_hurst_proxy(log_close)
    ret = torch.diff(log_close, dim=1, prepend=log_close[:, :1])
    vol = ret.abs()
    ekf_p = log_close.clone()
    alpha = 0.1
    for i in range(1, ekf_p.shape[1]):
        ekf_p[:, i] = alpha * log_close[:, i] + (1 - alpha) * ekf_p[:, i - 1]
    ekf_v = torch.diff(ekf_p, dim=1, prepend=ekf_p[:, :1])
    pe = calculate_entropy_proxy(high_price, low_price, close, window=LOCAL_PHYSICS_WINDOW)
    mle_t = calculate_lyapunov_proxy(ret, window=LOCAL_PHYSICS_WINDOW)
    res = log_close - ekf_p
    price_mom = ret
    ema_div = (close - ema20) / (ema20 + 1e-8)
    d_entropy = _diff_feature(pe)
    dd_entropy = _diff_feature(d_entropy)
    d_mle = _diff_feature(mle_t)
    d_res = _diff_feature(res)
    d_ema_div = _diff_feature(ema_div)
    vol_ref = ema_smooth(vol, LOCAL_PHYSICS_WINDOW)
    ent_ref = ema_smooth(pe, LOCAL_PHYSICS_WINDOW)
    compression = torch.relu(vol_ref - vol) * torch.relu(ent_ref - pe)
    return torch.stack([
        h_t,
        vol,
        ekf_v,
        res,
        pe,
        mle_t,
        price_mom,
        ema_div,
        d_entropy,
        dd_entropy,
        d_mle,
        d_res,
        d_ema_div,
        compression
    ], dim=2)

def compute_physics_features_bulk(x, device='cpu'):
    x_batch = x.unsqueeze(0).cpu()
    chunk_size = 50000
    _, T, _ = x_batch.shape
    all_features = []
    with torch.no_grad():
        for start_idx in range(0, T, chunk_size):
            pad = LOCAL_PHYSICS_WINDOW if start_idx > 0 else 0
            actual_start = start_idx - pad
            end_idx = min(start_idx + chunk_size, T)
            print(f"    [Chunk {start_idx}-{end_idx}] Extracting...")
            x_chunk = x_batch[:, actual_start:end_idx, :].to(device)
            features = _compute_core_features(
                x_chunk[:, :, 0],
                x_chunk[:, :, 1],
                x_chunk[:, :, 2],
                x_chunk[:, :, 3],
                x_chunk[:, :, 4],
                x_chunk[:, :, 5]
            ).squeeze(0)
            valid_features = features if start_idx == 0 else features[pad:]
            all_features.append(valid_features)
        final_features = torch.cat(all_features, dim=0)
    if device == 'cuda':
        torch.cuda.empty_cache()
    return final_features.cpu()

class PhysicsLayer(nn.Module):
    def __init__(self):
        super(PhysicsLayer, self).__init__()
        self.ekf = DifferentiableEKF()

    def forward(self, x):
        with torch.no_grad():
            return _compute_core_features(
                x[:, :, 0],
                x[:, :, 1],
                x[:, :, 2],
                x[:, :, 3],
                x[:, :, 4],
                x[:, :, 5]
            )

