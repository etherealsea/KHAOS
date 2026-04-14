import torch
import torch.nn as nn
import torch.nn.functional as F

from khaos.数据处理.ashare_dataset import EVENT_FLAG_INDEX, TRADE_MASK_INDEX


LOSS_WEIGHT_PRESETS = {
    'default': {
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
        'reversion_hard_negative': 0.42,
        'direction_consistency': 0.18,
        'continuation_suppression': 0.12,
        'horizon_event': 0.30,
        'horizon_aux': 0.16,
        'horizon_align': 0.22,
        'horizon_hard_negative': 0.28,
        'horizon_entropy': 0.04,
        'signal_calibration': 0.10,
        'horizon_margin': 0.14,
    },
    'iterA4': {
        'main': 1.0,
        'aux': 0.35,
        'rank': 0.24,
        'breakout_event_gap': 0.26,
        'reversion_event_gap': 0.30,
        'p3': 0.10,
        'p4': 0.12,
        'p6': 0.12,
        'p7': 0.15,
        'breakout_hard_negative': 0.32,
        'reversion_hard_negative': 0.40,
        'direction_consistency': 0.10,
        'continuation_suppression': 0.16,
        'horizon_event': 0.32,
        'horizon_aux': 0.18,
        'horizon_align': 0.24,
        'horizon_hard_negative': 0.30,
        'horizon_entropy': 0.05,
        'signal_calibration': 0.10,
        'horizon_margin': 0.16,
    },
    'iterA5': {
        'main': 1.0,
        'aux': 0.35,
        'rank': 0.24,
        'breakout_event_gap': 0.28,
        'reversion_event_gap': 0.34,
        'p3': 0.10,
        'p4': 0.12,
        'p6': 0.12,
        'p7': 0.15,
        'breakout_hard_negative': 0.32,
        'reversion_hard_negative': 0.42,
        'direction_consistency': 0.08,
        'continuation_suppression': 0.20,
        'horizon_event': 0.34,
        'horizon_aux': 0.18,
        'horizon_align': 0.26,
        'horizon_hard_negative': 0.30,
        'horizon_entropy': 0.05,
        'signal_calibration': 0.12,
        'horizon_margin': 0.16,
    },
    'shortT_breakout_v1': {
        'main': 1.0,
        'aux': 0.38,
        'rank': 0.24,
        'breakout_event_gap': 0.34,
        'reversion_event_gap': 0.24,
        'p3': 0.10,
        'p4': 0.10,
        'p6': 0.12,
        'p7': 0.14,
        'breakout_hard_negative': 0.46,
        'reversion_hard_negative': 0.32,
        'direction_consistency': 0.12,
        'continuation_suppression': 0.12,
        'horizon_event': 0.34,
        'horizon_aux': 0.16,
        'horizon_align': 0.22,
        'horizon_hard_negative': 0.32,
        'horizon_entropy': 0.05,
        'signal_calibration': 0.10,
        'horizon_margin': 0.18,
    },
    'shortT_balanced_v1': {
        'main': 1.0,
        'aux': 0.36,
        'rank': 0.24,
        'breakout_event_gap': 0.30,
        'reversion_event_gap': 0.30,
        'p3': 0.10,
        'p4': 0.10,
        'p6': 0.12,
        'p7': 0.14,
        'breakout_hard_negative': 0.38,
        'reversion_hard_negative': 0.38,
        'direction_consistency': 0.12,
        'continuation_suppression': 0.14,
        'horizon_event': 0.32,
        'horizon_aux': 0.16,
        'horizon_align': 0.22,
        'horizon_hard_negative': 0.30,
        'horizon_entropy': 0.05,
        'signal_calibration': 0.10,
        'horizon_margin': 0.16,
    },
    'shortT_balanced_v2': {
        'main': 1.0,
        'aux': 0.36,
        'rank': 0.24,
        'breakout_event_gap': 0.28,
        'reversion_event_gap': 0.32,
        'p3': 0.10,
        'p4': 0.10,
        'p6': 0.12,
        'p7': 0.14,
        'breakout_hard_negative': 0.36,
        'reversion_hard_negative': 0.40,
        'direction_consistency': 0.16,
        'continuation_suppression': 0.16,
        'horizon_event': 0.34,
        'horizon_aux': 0.18,
        'horizon_align': 0.24,
        'horizon_hard_negative': 0.32,
        'horizon_entropy': 0.05,
        'signal_calibration': 0.12,
        'horizon_margin': 0.18,
    },
    'shortT_precision_v1': {
        'main': 1.0,
        'aux': 0.34,
        'rank': 0.26,
        'breakout_event_gap': 0.34,
        'reversion_event_gap': 0.46,
        'p3': 0.10,
        'p4': 0.10,
        'p6': 0.12,
        'p7': 0.15,
        'breakout_hard_negative': 0.46,
        'reversion_hard_negative': 0.58,
        'direction_consistency': 0.24,
        'continuation_suppression': 0.26,
        'horizon_event': 0.42,
        'horizon_aux': 0.22,
        'horizon_align': 0.34,
        'horizon_hard_negative': 0.42,
        'horizon_entropy': 0.06,
        'signal_calibration': 0.16,
        'horizon_margin': 0.24,
    },
    'shortT_precision_v2': {
        'main': 1.0,
        'aux': 0.34,
        'rank': 0.26,
        'breakout_event_gap': 0.36,
        'reversion_event_gap': 0.40,
        'p3': 0.10,
        'p4': 0.10,
        'p6': 0.12,
        'p7': 0.15,
        'breakout_hard_negative': 0.48,
        'reversion_hard_negative': 0.56,
        'direction_consistency': 0.24,
        'continuation_suppression': 0.28,
        'horizon_event': 0.42,
        'horizon_aux': 0.22,
        'horizon_align': 0.34,
        'horizon_hard_negative': 0.42,
        'horizon_entropy': 0.06,
        'signal_calibration': 0.20,
        'horizon_margin': 0.24,
    },
    'shortT_discovery_v1': {
        'main': 1.0,
        'aux': 0.46,
        'rank': 0.32,
        'breakout_event_gap': 0.18,
        'reversion_event_gap': 0.20,
        'p3': 0.08,
        'p4': 0.08,
        'p6': 0.10,
        'p7': 0.12,
        'breakout_hard_negative': 0.18,
        'reversion_hard_negative': 0.22,
        'direction_consistency': 0.16,
        'continuation_suppression': 0.18,
        'horizon_event': 0.36,
        'horizon_aux': 0.26,
        'horizon_align': 0.26,
        'horizon_hard_negative': 0.24,
        'horizon_entropy': 0.04,
        'signal_calibration': 0.18,
        'horizon_margin': 0.20,
    },
    'shortT_discovery_guarded_v1': {
        'main': 1.0,
        'aux': 0.40,
        'rank': 0.28,
        'breakout_event_gap': 0.28,
        'reversion_event_gap': 0.34,
        'p3': 0.08,
        'p4': 0.08,
        'p6': 0.10,
        'p7': 0.12,
        'breakout_hard_negative': 0.32,
        'reversion_hard_negative': 0.40,
        'direction_consistency': 0.20,
        'continuation_suppression': 0.22,
        'horizon_event': 0.38,
        'horizon_aux': 0.24,
        'horizon_align': 0.30,
        'horizon_hard_negative': 0.32,
        'horizon_entropy': 0.04,
        'signal_calibration': 0.20,
        'horizon_margin': 0.22,
    },
    'shortT_discovery_guarded_v2': {
        'main': 1.0,
        'aux': 0.38,
        'rank': 0.30,
        'breakout_event_gap': 0.32,
        'reversion_event_gap': 0.40,
        'p3': 0.08,
        'p4': 0.08,
        'p6': 0.10,
        'p7': 0.12,
        'breakout_hard_negative': 0.36,
        'reversion_hard_negative': 0.46,
        'direction_consistency': 0.28,
        'continuation_suppression': 0.30,
        'horizon_event': 0.40,
        'horizon_aux': 0.22,
        'horizon_align': 0.30,
        'horizon_hard_negative': 0.34,
        'horizon_entropy': 0.04,
        'signal_calibration': 0.24,
        'horizon_margin': 0.22,
    },
    'horizon_precision_v1': {
        'main': 1.0,
        'aux': 0.32,
        'rank': 0.24,
        'breakout_event_gap': 0.36,
        'reversion_event_gap': 0.48,
        'p3': 0.08,
        'p4': 0.10,
        'p6': 0.10,
        'p7': 0.14,
        'breakout_hard_negative': 0.48,
        'reversion_hard_negative': 0.60,
        'direction_consistency': 0.22,
        'continuation_suppression': 0.24,
        'horizon_event': 0.48,
        'horizon_aux': 0.22,
        'horizon_align': 0.38,
        'horizon_hard_negative': 0.46,
        'horizon_entropy': 0.08,
        'signal_calibration': 0.20,
        'horizon_margin': 0.28,
    },
}


LOSS_CURRICULUM_PRESETS = {
    'shortT_discovery_guarded_v1': {
        'warmup_fraction': 0.45,
        'start_factor': 0.55,
        'keys': (
            'breakout_event_gap',
            'reversion_event_gap',
            'breakout_hard_negative',
            'reversion_hard_negative',
            'direction_consistency',
            'continuation_suppression',
            'horizon_event',
            'horizon_hard_negative',
            'horizon_margin',
            'signal_calibration',
        ),
        'scale_constraint_weight': True,
    },
    'shortT_discovery_guarded_v2': {
        'warmup_fraction': 0.35,
        'start_factor': 0.60,
        'keys': (
            'breakout_event_gap',
            'reversion_event_gap',
            'breakout_hard_negative',
            'reversion_hard_negative',
            'direction_consistency',
            'continuation_suppression',
            'horizon_event',
            'horizon_hard_negative',
            'horizon_margin',
            'signal_calibration',
        ),
        'scale_constraint_weight': True,
    },
}


CONSTRAINT_PROFILE_PRESETS = {
    'default': {
        'enabled': False,
        'weight': 0.0,
        'blue_margin': 0.12,
        'purple_margin': 0.12,
        'reversion_event_margin': 0.10,
        'continuation_margin': 0.05,
    },
    'teacher_feasible_v1': {
        'enabled': True,
        'weight': 0.35,
        'blue_margin': 0.12,
        'purple_margin': 0.12,
        'reversion_event_margin': 0.10,
        'continuation_margin': 0.05,
    },
    'teacher_feasible_precision_v1': {
        'enabled': True,
        'weight': 0.45,
        'blue_margin': 0.14,
        'purple_margin': 0.14,
        'reversion_event_margin': 0.14,
        'continuation_margin': 0.08,
    },
    'teacher_feasible_discovery_v1': {
        'enabled': True,
        'weight': 0.24,
        'blue_margin': 0.10,
        'purple_margin': 0.10,
        'reversion_event_margin': 0.08,
        'continuation_margin': 0.06,
    },
}


class PhysicsLoss(nn.Module):
    def __init__(self, weights=None, profile='default', constraint_profile='default', family_mode='legacy'):
        super().__init__()
        base_weights = dict(LOSS_WEIGHT_PRESETS.get(profile, LOSS_WEIGHT_PRESETS['default']))
        if weights:
            base_weights.update(weights)
        self.profile = str(profile or 'default')
        self.base_weights = dict(base_weights)
        self.weights = dict(base_weights)
        self.family_mode = str(family_mode or 'legacy')
        self.constraint_profile = str(constraint_profile or 'default')
        self.base_constraint_config = dict(
            CONSTRAINT_PROFILE_PRESETS.get(self.constraint_profile, CONSTRAINT_PROFILE_PRESETS['default'])
        )
        self.constraint_config = dict(self.base_constraint_config)
        self.curriculum_config = dict(LOSS_CURRICULUM_PRESETS.get(self.profile, {}))
        self.main_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.aux_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.horizon_event_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.horizon_aux_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.set_epoch(0, 1)

    def set_epoch(self, epoch, total_epochs):
        self.weights = dict(self.base_weights)
        self.constraint_config = dict(self.base_constraint_config)
        if not self.curriculum_config:
            return
        total_epochs = max(int(total_epochs), 1)
        warmup_fraction = float(self.curriculum_config.get('warmup_fraction', 0.0))
        start_factor = float(self.curriculum_config.get('start_factor', 1.0))
        if warmup_fraction <= 0.0 or start_factor >= 1.0:
            return
        warmup_epochs = max(int(round(total_epochs * warmup_fraction)), 1)
        progress = min(max(int(epoch) + 1, 0) / float(warmup_epochs), 1.0)
        factor = start_factor + (1.0 - start_factor) * progress
        for key in self.curriculum_config.get('keys', ()):
            if key in self.weights:
                self.weights[key] = self.base_weights[key] * factor
        if self.curriculum_config.get('scale_constraint_weight', False):
            self.constraint_config['weight'] = float(self.base_constraint_config.get('weight', 0.0)) * factor

    def _get_flag(self, event_flags, flag_name):
        idx = EVENT_FLAG_INDEX[flag_name]
        if event_flags.shape[-1] <= idx:
            return event_flags.new_zeros(event_flags.shape[:-1])
        return event_flags[..., idx]

    def _pairwise_rank_loss(self, scores, strengths):
        pos_mask = strengths >= torch.quantile(strengths.detach(), 0.75)
        neg_mask = strengths <= torch.quantile(strengths.detach(), 0.25)
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return scores.new_tensor(0.0)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        return F.softplus(-diff).mean()

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

    def _direction_margin_loss(self, preferred_scores, alternate_scores, context_mask, margin=0.10):
        selected = context_mask > 0.5
        if not torch.any(selected):
            return preferred_scores.new_tensor(0.0)
        diff = preferred_scores[selected] - alternate_scores[selected]
        return torch.relu(margin - diff).mean()

    def _masked_violation_stats(self, violation, mask):
        selected = mask > 0.5
        if not torch.any(selected):
            return violation.new_tensor(0.0), violation.new_tensor(0.0)
        selected_violation = violation[selected]
        return selected_violation.mean(), (selected_violation > 1e-6).float().mean()

    def _to_task_horizon_mask(self, tensor, reference):
        if tensor is None:
            return None
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1).expand(-1, reference.size(1), -1)
        return tensor.to(device=reference.device, dtype=reference.dtype)

    def _horizon_alignment_loss(self, horizon_logits, q_horizon, valid_mask):
        if horizon_logits is None:
            return q_horizon.new_zeros(q_horizon.size(0), 1), q_horizon.new_tensor(0.0)
        masked_target = q_horizon * valid_mask
        masked_target = masked_target / masked_target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        masked_logits = horizon_logits.masked_fill(valid_mask <= 0.5, torch.finfo(horizon_logits.dtype).min)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        per_task = -(masked_target * log_probs).sum(dim=-1)
        per_sample = per_task.mean(dim=1, keepdim=True)
        return per_sample, per_sample.mean()

    def _horizon_event_margin(self, event_logits, event_flags, hard_negative, aux_targets, valid_mask):
        breakout_margin = self._event_margin_loss(
            event_logits[:, 0, :].reshape(-1),
            ((event_flags[:, 0, :] > 0.5) & (valid_mask[:, 0, :] > 0.5)).reshape(-1),
            ((hard_negative[:, 0, :] > 0.5) & (valid_mask[:, 0, :] > 0.5)).reshape(-1),
            aux_targets[:, 0, :].reshape(-1),
            margin=0.24,
            scale=0.18,
        )
        reversion_margin = self._event_margin_loss(
            torch.relu(event_logits[:, 1, :]).reshape(-1),
            ((event_flags[:, 1, :] > 0.5) & (valid_mask[:, 1, :] > 0.5)).reshape(-1),
            ((hard_negative[:, 1, :] > 0.5) & (valid_mask[:, 1, :] > 0.5)).reshape(-1),
            aux_targets[:, 1, :].reshape(-1),
            margin=0.32,
            scale=0.22,
        )
        return breakout_margin + reversion_margin

    def _compute_horizon_terms(self, pred, aux_pred, debug_info, horizon_payload):
        event_logits = debug_info.get('event_logits_by_horizon')
        aux_logits = debug_info.get('aux_logits_by_horizon')
        horizon_logits = debug_info.get('horizon_logits')
        horizon_weights = debug_info.get('horizon_weights')
        if event_logits is None or aux_logits is None or horizon_weights is None:
            return pred.new_zeros(pred.size(0), 1), pred.new_tensor(0.0), {}

        targets_by_horizon = horizon_payload.get('targets_by_horizon')
        aux_by_horizon = horizon_payload.get('aux_by_horizon')
        event_flags_by_horizon = horizon_payload.get('event_flags_by_horizon')
        hard_negative_by_horizon = horizon_payload.get('hard_negative_by_horizon')
        q_horizon = horizon_payload.get('q_horizon')
        trade_masks = horizon_payload.get('trade_masks')
        valid_horizon_mask = horizon_payload.get('valid_horizon_mask')
        if any(
            item is None
            for item in (
                targets_by_horizon,
                aux_by_horizon,
                event_flags_by_horizon,
                hard_negative_by_horizon,
                q_horizon,
                valid_horizon_mask,
            )
        ):
            return pred.new_zeros(pred.size(0), 1), pred.new_tensor(0.0), {}

        valid_mask = self._to_task_horizon_mask(valid_horizon_mask, event_logits)
        q_horizon = q_horizon.to(device=event_logits.device, dtype=event_logits.dtype) * valid_mask
        q_horizon = q_horizon / q_horizon.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        targets_by_horizon = targets_by_horizon.to(device=event_logits.device, dtype=event_logits.dtype)
        aux_by_horizon = aux_by_horizon.to(device=event_logits.device, dtype=event_logits.dtype)
        event_flags_by_horizon = event_flags_by_horizon.to(device=event_logits.device, dtype=event_logits.dtype)
        hard_negative_by_horizon = hard_negative_by_horizon.to(device=event_logits.device, dtype=event_logits.dtype)
        selector_focus = 0.30 + 0.70 * q_horizon + 0.35 * event_flags_by_horizon + 0.20 * hard_negative_by_horizon
        selector_focus = selector_focus * valid_mask
        selector_denom = selector_focus.sum(dim=(1, 2), keepdim=False).unsqueeze(1).clamp_min(1e-6)

        event_loss = self.horizon_event_loss_fn(event_logits, targets_by_horizon)
        event_loss = (event_loss * selector_focus).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / selector_denom

        aux_loss = self.horizon_aux_loss_fn(torch.relu(aux_logits), aux_by_horizon)
        aux_loss = (aux_loss * selector_focus).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / selector_denom

        hard_negative_penalty = (
            (F.relu(event_logits) * hard_negative_by_horizon * valid_mask).sum(dim=(1, 2), keepdim=False).unsqueeze(1) /
            valid_mask.sum(dim=(1, 2), keepdim=False).unsqueeze(1).clamp_min(1e-6)
        )

        if self.family_mode == 'single_cycle':
            horizon_align_per_sample = pred.new_zeros(pred.size(0), 1)
            horizon_align_scalar = pred.new_tensor(0.0)
        else:
            horizon_align_per_sample, horizon_align_scalar = self._horizon_alignment_loss(
                horizon_logits=horizon_logits,
                q_horizon=q_horizon,
                valid_mask=valid_mask,
            )

        entropy_per_task = (-horizon_weights.clamp_min(1e-8).log() * horizon_weights).sum(dim=-1)
        entropy_loss = entropy_per_task.mean(dim=1, keepdim=True)

        trade_block = pred.new_zeros(pred.size(0), 1)
        if trade_masks is not None and trade_masks.shape[-1] > TRADE_MASK_INDEX['trade_block']:
            trade_block = trade_masks[..., TRADE_MASK_INDEX['trade_block']].to(
                device=pred.device,
                dtype=pred.dtype,
            ).unsqueeze(1)
        signal_strength = torch.relu(pred[..., 0:1]) + torch.relu(pred[..., 1:2])
        calibration_loss = trade_block * signal_strength

        horizon_per_sample = (
            self.weights.get('horizon_event', 0.0) * event_loss +
            self.weights.get('horizon_aux', 0.0) * aux_loss +
            self.weights.get('horizon_align', 0.0) * horizon_align_per_sample +
            self.weights.get('horizon_hard_negative', 0.0) * hard_negative_penalty +
            self.weights.get('horizon_entropy', 0.0) * entropy_loss +
            self.weights.get('signal_calibration', 0.0) * calibration_loss
        )
        horizon_margin = self._horizon_event_margin(
            event_logits=event_logits,
            event_flags=event_flags_by_horizon,
            hard_negative=hard_negative_by_horizon,
            aux_targets=aux_by_horizon,
            valid_mask=valid_mask,
        )
        horizon_rank = self.weights.get('horizon_margin', 0.0) * horizon_margin

        return horizon_per_sample, horizon_rank, {
            'horizon_event': event_loss.mean().item(),
            'horizon_aux': aux_loss.mean().item(),
            'horizon_align': horizon_align_scalar.item(),
            'horizon_hard_negative': hard_negative_penalty.mean().item(),
            'horizon_entropy': entropy_loss.mean().item(),
            'signal_calibration': calibration_loss.mean().item(),
            'horizon_margin': horizon_margin.item(),
        }

    def forward(
        self,
        pred,
        aux_pred,
        target,
        aux_target,
        physics_state,
        event_flags,
        sigma=None,
        debug_info=None,
        horizon_payload=None,
    ):
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
        breakout_event = self._get_flag(event_flags, 'breakout_event')
        reversion_event = self._get_flag(event_flags, 'reversion_event')
        blue_context = self._get_flag(event_flags, 'reversion_down_context')
        purple_context = self._get_flag(event_flags, 'reversion_up_context')
        continuation_pressure = self._get_flag(event_flags, 'continuation_pressure')
        breakout_hard_negative_penalty = breakout_hard_negative * torch.relu(pred_vol)
        reversion_hard_negative_penalty = reversion_hard_negative * pred_rev
        breakout_event_gap_loss = self._event_margin_loss(
            pred_vol,
            breakout_event > 0.5,
            breakout_hard_negative > 0.5,
            aux_target[..., 0],
            margin=0.24,
            scale=0.18,
        )
        reversion_event_gap_loss = self._event_margin_loss(
            pred_rev,
            reversion_event > 0.5,
            reversion_hard_negative > 0.5,
            aux_target[..., 1],
            margin=0.32,
            scale=0.22,
        )

        if debug_info is not None:
            blue_score = torch.relu(debug_info['blue_score'].squeeze(-1))
            purple_score = torch.relu(debug_info['purple_score'].squeeze(-1))
            directional_floor = debug_info.get('directional_floor')
            if directional_floor is not None:
                directional_floor = torch.relu(directional_floor.squeeze(-1))
            else:
                directional_floor = torch.maximum(blue_score, purple_score)
        else:
            blue_score = pred_rev
            purple_score = pred_rev
            directional_floor = pred_rev

        direction_consistency_loss = (
            self._direction_margin_loss(blue_score, purple_score, blue_context, margin=0.12) +
            self._direction_margin_loss(purple_score, blue_score, purple_context, margin=0.12)
        )
        constraint_cfg = self.constraint_config
        blue_over_purple_raw = torch.relu(
            purple_score + constraint_cfg.get('blue_margin', 0.12) - blue_score
        )
        purple_over_blue_raw = torch.relu(
            blue_score + constraint_cfg.get('purple_margin', 0.12) - purple_score
        )
        public_below_directional_raw = torch.relu(
            directional_floor + constraint_cfg.get('reversion_event_margin', 0.10) - pred_rev
        )
        continuation_public_raw = torch.relu(
            pred_rev - (directional_floor + constraint_cfg.get('continuation_margin', 0.05))
        )
        blue_over_purple_violation, blue_over_purple_violation_rate = self._masked_violation_stats(
            blue_over_purple_raw,
            blue_context,
        )
        purple_over_blue_violation, purple_over_blue_violation_rate = self._masked_violation_stats(
            purple_over_blue_raw,
            purple_context,
        )
        public_below_directional_violation, public_below_directional_violation_rate = self._masked_violation_stats(
            public_below_directional_raw,
            reversion_event,
        )
        continuation_public_violation, continuation_public_violation_rate = self._masked_violation_stats(
            continuation_public_raw,
            continuation_pressure,
        )
        constraint_penalty = (
            blue_context * blue_over_purple_raw +
            purple_context * purple_over_blue_raw +
            reversion_event * public_below_directional_raw +
            continuation_pressure * continuation_public_raw
        )
        continuation_suppression = continuation_pressure * (
            pred_rev +
            0.35 * blue_score +
            0.35 * purple_score
        )

        horizon_per_sample = pred.new_zeros(pred.size(0), 1)
        horizon_rank = pred.new_tensor(0.0)
        horizon_logs = {}
        if horizon_payload is not None and debug_info is not None:
            horizon_per_sample, horizon_rank, horizon_logs = self._compute_horizon_terms(
                pred=pred,
                aux_pred=aux_pred,
                debug_info=debug_info,
                horizon_payload=horizon_payload,
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
            self.weights.get('reversion_hard_negative', 0.42) * reversion_hard_negative_penalty.unsqueeze(1) +
            self.weights.get('continuation_suppression', 0.12) * continuation_suppression.unsqueeze(1) +
            constraint_cfg.get('weight', 0.0) * constraint_penalty.unsqueeze(1) +
            horizon_per_sample
        )
        rank_loss = (
            self.weights.get('rank', 0.20) * (
                self._pairwise_rank_loss(pred[..., 0], aux_target[..., 0]) +
                self._pairwise_rank_loss(pred[..., 1], aux_target[..., 1])
            ) +
            self.weights.get('breakout_event_gap', 0.18) * breakout_event_gap_loss +
            self.weights.get('reversion_event_gap', 0.28) * reversion_event_gap_loss +
            self.weights.get('direction_consistency', 0.18) * direction_consistency_loss +
            horizon_rank
        )

        logs = {
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
            'reversion_hard_negative': reversion_hard_negative_penalty.mean().item(),
            'direction_consistency': direction_consistency_loss.item(),
            'continuation_suppression': continuation_suppression.mean().item(),
            'constraint_penalty': constraint_penalty.mean().item(),
            'blue_over_purple_violation': blue_over_purple_violation.item(),
            'blue_over_purple_violation_rate': blue_over_purple_violation_rate.item(),
            'purple_over_blue_violation': purple_over_blue_violation.item(),
            'purple_over_blue_violation_rate': purple_over_blue_violation_rate.item(),
            'public_below_directional_violation': public_below_directional_violation.item(),
            'public_below_directional_violation_rate': public_below_directional_violation_rate.item(),
            'continuation_public_violation': continuation_public_violation.item(),
            'continuation_public_violation_rate': continuation_public_violation_rate.item(),
        }
        logs.update(horizon_logs)
        return per_sample_loss, rank_loss, logs
