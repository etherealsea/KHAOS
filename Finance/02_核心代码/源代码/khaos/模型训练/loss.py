import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'main': 1.0,
            'aux': 0.35,
            'rank': 0.20,
            'breakout_event_gap': 0.18,
            'reversion_event_gap': 0.28,
            'p3': 0.10,
            'p4': 0.12,
            'p6': 0.12,
            'p7': 0.15,
            'breakout_hard_negative': 0.24,
            'reversion_hard_negative': 0.36
        }
        self.main_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.aux_loss_fn = nn.SmoothL1Loss(reduction='none')

    def _pairwise_rank_loss(self, scores, strengths):
        pos_mask = strengths >= torch.quantile(strengths.detach(), 0.75)
        neg_mask = strengths <= torch.quantile(strengths.detach(), 0.25)
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return scores.new_tensor(0.0)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        return torch.nn.functional.softplus(-diff).mean()

    def _event_margin_loss(self, scores, pos_mask, neg_mask, strengths=None, margin=0.25, scale=0.20):
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return scores.new_tensor(0.0)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        if strengths is not None and torch.any(pos_mask):
            pos_strength = torch.clamp(strengths[pos_mask], 0.0, 2.0)
            dynamic_margin = margin + scale * torch.tanh(pos_strength).unsqueeze(1)
        else:
            dynamic_margin = margin
        return torch.relu(dynamic_margin - diff).mean()

    def forward(self, pred, aux_pred, target, aux_target, physics_state, event_flags, sigma=None):
        if pred.shape != target.shape:
            target = target.view_as(pred)
        if aux_pred.shape != aux_target.shape:
            aux_target = aux_target.view_as(aux_pred)
        H = physics_state[..., 0]
        Vol = physics_state[..., 1]
        Res = physics_state[..., 3]
        Ent = physics_state[..., 4]
        MLE = physics_state[..., 5]
        EMA_Div = physics_state[..., 7]
        dEnt = physics_state[..., 8]
        ddEnt = physics_state[..., 9]
        compression = physics_state[..., 13]
        sigma_ref = torch.clamp(sigma if sigma is not None else torch.ones_like(Res), min=1e-6)

        main_loss = self.main_loss_fn(pred, target)
        main_loss = main_loss.mean(dim=1, keepdim=True)
        aux_loss = self.aux_loss_fn(torch.relu(aux_pred), aux_target).mean(dim=1, keepdim=True)

        pred_vol = pred[..., 0]
        pred_rev = torch.relu(pred[..., 1])
        p3 = torch.relu(Ent - 0.7) * torch.relu(0.0 - pred_vol)
        res_score = Res.abs() / sigma_ref
        ema_score = EMA_Div.abs() / sigma_ref
        alignment = (Res + EMA_Div).abs() / (Res.abs() + EMA_Div.abs() + 1e-6)
        reversion_setup = torch.relu(res_score - 1.0) * torch.relu(ema_score - 0.5) * alignment
        p4 = reversion_setup * torch.relu(0.0 - pred[..., 1])
        p6_lyapunov = torch.relu(MLE) * torch.relu(0.0 - pred_vol)
        vol_mean = Vol.mean()
        p7_csd = torch.relu(H - 0.6) * torch.relu(vol_mean - Vol) * torch.relu(MLE - 0.1) * torch.relu(0.0 - pred_vol)
        continuation_bias = torch.relu(H - 0.55) * torch.relu(MLE) * torch.relu(Vol - vol_mean)
        weak_dislocation = torch.relu(0.5 - reversion_setup)
        p7_false_reversion = continuation_bias * weak_dislocation * pred_rev
        transition_breakout = torch.relu(compression + torch.relu(-dEnt) + torch.relu(ddEnt))
        transition_reversion = torch.relu(torch.relu(H - 0.55) + torch.relu(-Ent) + torch.relu(dEnt))

        breakout_hard_negative = event_flags[..., 2]
        reversion_hard_negative = event_flags[..., 3]
        breakout_event = event_flags[..., 0]
        reversion_event = event_flags[..., 1]
        breakout_hard_negative_penalty = breakout_hard_negative * torch.relu(pred_vol)
        reversion_hard_negative_penalty = reversion_hard_negative * pred_rev
        breakout_event_gap_loss = self._event_margin_loss(
            pred_vol, breakout_event > 0.5, breakout_hard_negative > 0.5, aux_target[..., 0], margin=0.24, scale=0.18
        )
        reversion_event_gap_loss = self._event_margin_loss(
            pred_rev, reversion_event > 0.5, reversion_hard_negative > 0.5, aux_target[..., 1], margin=0.32, scale=0.22
        )

        per_sample_loss = (
            self.weights.get('main', 1.0) * main_loss +
            self.weights.get('aux', 0.35) * aux_loss +
            self.weights.get('p3', 0.10) * p3.unsqueeze(1) +
            self.weights.get('p4', 0.12) * p4.unsqueeze(1) +
            self.weights.get('p6', 0.12) * p6_lyapunov.unsqueeze(1) +
            self.weights.get('p7', 0.15) * (p7_csd + p7_false_reversion).unsqueeze(1) +
            0.05 * transition_breakout.unsqueeze(1) * torch.relu(-aux_pred[..., 0]).unsqueeze(1) +
            0.05 * transition_reversion.unsqueeze(1) * torch.relu(-aux_pred[..., 1]).unsqueeze(1) +
            self.weights.get('breakout_hard_negative', 0.24) * breakout_hard_negative_penalty.unsqueeze(1) +
            self.weights.get('reversion_hard_negative', 0.36) * reversion_hard_negative_penalty.unsqueeze(1)
        )
        rank_loss = self.weights.get('rank', 0.20) * (
            self._pairwise_rank_loss(pred[..., 0], aux_target[..., 0]) +
            self._pairwise_rank_loss(pred[..., 1], aux_target[..., 1])
        ) + (
            self.weights.get('breakout_event_gap', 0.18) * breakout_event_gap_loss +
            self.weights.get('reversion_event_gap', 0.28) * reversion_event_gap_loss
        )

        return per_sample_loss, rank_loss, {
            'main': main_loss.mean().item(),
            'aux': aux_loss.mean().item(),
            'p3_ent_vol': p3.mean().item(),
            'p4_reversion_setup': p4.mean().item(),
            'p6_mle_chaos': p6_lyapunov.mean().item(),
            'p7_csd': p7_csd.mean().item(),
            'p7_false_reversion': p7_false_reversion.mean().item(),
            'rank': rank_loss.item(),
            'event_gap_loss': (breakout_event_gap_loss + reversion_event_gap_loss).item(),
            'breakout_hard_negative': breakout_hard_negative_penalty.mean().item(),
            'reversion_hard_negative': reversion_hard_negative_penalty.mean().item()
        }
