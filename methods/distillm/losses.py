import importlib.util
import os
import torch
import torch.nn.functional as F


def _load_soft_dtw():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(base_dir, "DTW-KD", "soft_dtw_cuda.py")
    if not os.path.exists(module_path):
        return None
    spec = importlib.util.spec_from_file_location("soft_dtw_cuda", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "SoftDTW", None)


SoftDTW = _load_soft_dtw()
_HAS_SOFT_DTW = SoftDTW is not None

def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def l2_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes normalized L2 loss only for valid (non-padded) tokens."""
    # Compute mean squared error per token (averaged over vocab dimension)
    mse_per_token = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # [B, L]
    # Apply mask and average over valid tokens
    masked_losses = mse_per_token * mask.float()
    valid_tokens = mask.sum()
    if valid_tokens > 0:
        return masked_losses.sum() / valid_tokens
    else:
        return torch.tensor(0.0, device=pred.device)


def cosine_similarity_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes masked cosine similarity loss."""
    # Compute cosine similarity per token
    cos_sim = F.cosine_similarity(pred.float(), target.float(), dim=-1)  # [B, L]
    # Apply mask and compute loss for valid tokens only
    masked_cos_sim = cos_sim * mask.float()
    valid_tokens = mask.sum()
    if valid_tokens > 0:
        return (1 - masked_cos_sim.sum() / valid_tokens)
    else:
        return torch.tensor(0.0, device=pred.device)


def hybrid_loss_masked(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor, 
    cosine_weight: float = 0.6, 
    l2_weight: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid loss combining cosine similarity and L2 loss.
    
    Args:
        pred: Predicted logits [B, L, V]
        target: Target logits [B, L, V] 
        mask: Attention mask [B, L]
        cosine_weight: Weight for cosine similarity loss (emphasizes direction)
        l2_weight: Weight for L2 loss (emphasizes magnitude)
    
    Returns:
        Combined loss value, cosine loss, l2 loss
    """
    # Cosine similarity loss (for directional alignment)
    cosine_loss = cosine_similarity_loss_masked(pred, target, mask)
    
    # L2 loss (for magnitude preservation)
    l2_loss = l2_loss_masked(pred, target, mask)
    
    # Combine with weights
    hybrid_loss = cosine_weight * cosine_loss + l2_weight * l2_loss
    
    return hybrid_loss, cosine_loss, l2_loss


def cosine_similarity_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes 1 - mean(cosine_similarity(a, b))."""
    return (1 - F.cosine_similarity(a.float(), b.float(), dim=-1)).mean()


def _pairwise_cost_matrix(student_seq: torch.Tensor, teacher_seq: torch.Tensor, distance: str) -> torch.Tensor:
    if distance == "cosine":
        student_norm = F.normalize(student_seq.float(), dim=-1)
        teacher_norm = F.normalize(teacher_seq.float(), dim=-1)
        return 1.0 - student_norm @ teacher_norm.transpose(0, 1)
    if distance == "l2":
        return torch.cdist(student_seq.float(), teacher_seq.float(), p=2).pow(2)
    raise ValueError(f"Unsupported DTW distance: {distance}")


def _pool_by_units(
    seq: torch.Tensor,
    unit_ids: torch.Tensor | None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if unit_ids is None:
        return seq, weights

    valid = unit_ids >= 0
    if not torch.any(valid):
        empty = seq[:0]
        return empty, weights[:0] if weights is not None else None

    unit_ids = unit_ids[valid]
    seq = seq[valid]
    if weights is not None:
        weights = weights[valid]

    uniq, counts = torch.unique_consecutive(unit_ids, return_counts=True)
    if uniq.numel() == 0:
        empty = seq[:0]
        return empty, weights[:0] if weights is not None else None

    ends = torch.cumsum(counts, dim=0)
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    pooled_seq = []
    pooled_weights = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        segment = seq[start:end]
        pooled_seq.append(segment.mean(dim=0))
        if weights is not None:
            pooled_weights.append(weights[start:end].mean())

    pooled_seq = torch.stack(pooled_seq, dim=0)
    pooled_weights = torch.stack(pooled_weights) if weights is not None else None
    return pooled_seq, pooled_weights


def _apply_importance_weights(cost: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return cost
    weights = weights.clamp_min(0.0)
    denom = weights.mean().clamp_min(1e-6)
    weights = weights / denom
    return cost * (weights.view(-1, 1) + weights.view(1, -1)) * 0.5


def _soft_dtw_banded(cost: torch.Tensor, gamma: float, window_size: int | None) -> torch.Tensor:
    # Always compute in float32 to avoid fp16 overflow/NaN
    cost = cost.float()
    n, m = cost.shape
    large = 1e9
    dp = torch.full((n + 1, m + 1), large, device=cost.device, dtype=torch.float32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        if window_size is None:
            j_start, j_end = 1, m
        else:
            j_start = max(1, i - window_size)
            j_end = min(m, i + window_size)
        for j in range(j_start, j_end + 1):
            r0 = dp[i - 1, j - 1]
            r1 = dp[i - 1, j]
            r2 = dp[i, j - 1]
            # Clamp to avoid -inf/nan in logsumexp
            stacked = torch.stack((-r0 / gamma, -r1 / gamma, -r2 / gamma)).clamp(min=-1e6)
            softmin = -gamma * torch.logsumexp(stacked, dim=0)
            dp[i, j] = cost[i - 1, j - 1] + softmin
    return dp[n, m]


def _soft_dtw_banded_alignment(cost: torch.Tensor, gamma: float, window_size: int | None) -> torch.Tensor:
    cost_for_grad = cost.clone().detach().requires_grad_(True)
    dtw = _soft_dtw_banded(cost_for_grad, gamma, window_size)
    alignment = torch.autograd.grad(dtw, cost_for_grad, retain_graph=False, create_graph=False)[0]
    alignment = alignment.clamp_min(0.0)
    alignment = alignment / (alignment.sum(dim=-1, keepdim=True) + 1e-9)
    return alignment


def _apply_sdtw_band_penalty_from_alignment(
    cost: torch.Tensor,
    alignment: torch.Tensor,
    band_width: int,
    band_penalty: float,
    band_center_blend: float,
    band_entropy_coef: float,
    band_warmup_steps: int,
    current_step: int | None,
) -> torch.Tensor:
    row_entropy = -(alignment * torch.log(alignment + 1e-9)).sum(dim=-1)
    s_len, t_len = cost.shape
    lin_center = torch.arange(s_len, device=cost.device, dtype=torch.float32) * (float(t_len) / float(s_len))
    soft_center = (alignment * torch.arange(t_len, device=cost.device).view(1, -1)).sum(dim=-1)
    centers = band_center_blend * soft_center + (1.0 - band_center_blend) * lin_center
    width = band_width + band_entropy_coef * row_entropy
    j = torch.arange(t_len, device=cost.device).view(1, -1).float()
    dist = (j - centers.view(-1, 1)).abs()
    band = dist <= width.view(-1, 1)

    if band_warmup_steps and band_warmup_steps > 0 and current_step is not None:
        pen_scale = min(1.0, float(current_step + 1) / float(band_warmup_steps))
    else:
        pen_scale = 1.0
    penalty = band_penalty * pen_scale
    return cost + (~band).float() * penalty


def _apply_sdtw_band_penalty(
    cost: torch.Tensor,
    gamma: float,
    window_size: int | None,
    band_width: int,
    band_penalty: float,
    band_center_blend: float,
    band_entropy_coef: float,
    band_warmup_steps: int,
    current_step: int | None,
) -> torch.Tensor:
    alignment = _soft_dtw_banded_alignment(cost, gamma, window_size)
    return _apply_sdtw_band_penalty_from_alignment(
        cost,
        alignment,
        band_width,
        band_penalty,
        band_center_blend,
        band_entropy_coef,
        band_warmup_steps,
        current_step,
    )


def dtw_distillation_loss(
    student_hiddens,
    teacher_hiddens,
    student_schedule,
    teacher_schedule,
    attention_mask: torch.Tensor,
    projector=None,
    window_size: int | None = 32,
    gamma: float = 0.1,
    distance: str = "cosine",
    normalize: bool = True,
    use_divergence: bool = False,
    band_source: str = "none",
    band_width: int = 0,
    band_penalty: float = 1.0,
    band_center_blend: float = 0.7,
    band_entropy_coef: float = 2.0,
    band_warmup_steps: int = 0,
    current_step: int | None = None,
    unit_ids: torch.Tensor | None = None,
    importance_weights: torch.Tensor | None = None,
):
    """
    Compute banded soft-DTW loss between student and teacher hidden states.
    Uses the provided layer schedules and ignores masked tokens.
    """
    batch_size = attention_mask.size(0)
    lengths = attention_mask.sum(dim=1).long()
    total_loss = torch.tensor(0.0, device=attention_mask.device, dtype=torch.float32)
    count = 0

    student_layer = student_hiddens[-1]
    teacher_layer = teacher_hiddens[-1]

    if projector is not None:
        student_layer = student_layer.to(projector.ln.weight.dtype)
        student_layer = projector(student_layer)
    elif student_layer.size(-1) != teacher_layer.size(-1):
        raise ValueError("DTW requires a projector when student/teacher dimensions differ.")

    # Always use float32 for numerical stability
    student_layer = student_layer.float()
    teacher_layer = teacher_layer.float()

    use_cuda_sdtw = _HAS_SOFT_DTW and torch.cuda.is_available() and attention_mask.is_cuda

    if use_cuda_sdtw:
        sdtw = SoftDTW(use_cuda=True, gamma=gamma, bandwidth=window_size)
        cost_s2t_batch = []
        cost_s2s_batch = []
        cost_t2t_batch = []
        pooled_lens = []
        too_long = False

        for b in range(batch_size):
            seq_len = int(lengths[b].item())
            if seq_len < 2:
                continue
            student_seq = student_layer[b, :seq_len]
            teacher_seq = teacher_layer[b, :seq_len]

            unit_row = unit_ids[b, :seq_len] if unit_ids is not None else None
            weight_row = importance_weights[b, :seq_len] if importance_weights is not None else None

            student_seq, pooled_weights = _pool_by_units(student_seq, unit_row, weight_row)
            teacher_seq, _ = _pool_by_units(teacher_seq, unit_row, None)

            if student_seq.numel() == 0 or teacher_seq.numel() == 0:
                continue

            pooled_len = student_seq.size(0)
            if pooled_len > 1024:
                too_long = True
                break

            pooled_lens.append(pooled_len)

            cost_s2t = _pairwise_cost_matrix(student_seq, teacher_seq, distance)
            cost_s2t = _apply_importance_weights(cost_s2t, pooled_weights)
            if band_source == "sdtw" and band_width > 0:
                lengths_tensor = torch.tensor([pooled_len], device=cost_s2t.device, dtype=torch.long)
                _, alignment = sdtw.forward_with_cost_matrix(
                    cost_s2t.unsqueeze(0),
                    return_alignment=True,
                    lengths=lengths_tensor,
                )
                alignment = alignment.squeeze(0)
                cost_s2t = _apply_sdtw_band_penalty_from_alignment(
                    cost_s2t,
                    alignment,
                    band_width,
                    band_penalty,
                    band_center_blend,
                    band_entropy_coef,
                    band_warmup_steps,
                    current_step,
                )

            cost_s2t_batch.append(cost_s2t)

            if use_divergence:
                cost_s2s = _pairwise_cost_matrix(student_seq, student_seq, distance)
                cost_t2t = _pairwise_cost_matrix(teacher_seq, teacher_seq, distance)
                cost_s2s = _apply_importance_weights(cost_s2s, pooled_weights)
                cost_t2t = _apply_importance_weights(cost_t2t, pooled_weights)
                cost_s2s_batch.append(cost_s2s)
                cost_t2t_batch.append(cost_t2t)

        if too_long:
            use_cuda_sdtw = False
        elif len(cost_s2t_batch) == 0:
            return torch.tensor(0.0, device=attention_mask.device, dtype=torch.float32)
        else:
            lengths_tensor = torch.tensor(pooled_lens, device=attention_mask.device, dtype=torch.long)
            max_len = int(lengths_tensor.max().item())
            padded_s2t = torch.full((len(cost_s2t_batch), max_len, max_len), float("inf"), device=attention_mask.device, dtype=cost_s2t_batch[0].dtype)
            for i, cost in enumerate(cost_s2t_batch):
                l = cost.size(0)
                padded_s2t[i, :l, :l] = cost
            s2t = sdtw.forward_with_cost_matrix(padded_s2t, lengths=lengths_tensor)

            if use_divergence:
                padded_s2s = torch.full_like(padded_s2t, float("inf"))
                padded_t2t = torch.full_like(padded_s2t, float("inf"))
                for i, (cost_s2s, cost_t2t) in enumerate(zip(cost_s2s_batch, cost_t2t_batch)):
                    l = cost_s2s.size(0)
                    padded_s2s[i, :l, :l] = cost_s2s
                    padded_t2t[i, :l, :l] = cost_t2t
                s2s = sdtw.forward_with_cost_matrix(padded_s2s, lengths=lengths_tensor)
                t2t = sdtw.forward_with_cost_matrix(padded_t2t, lengths=lengths_tensor)

            for i, pooled_len in enumerate(pooled_lens):
                dtw = s2t[i]
                if use_divergence:
                    dtw = dtw - 0.5 * (s2s[i] + t2t[i])
                if normalize:
                    dtw = dtw / (2.0 * pooled_len)
                total_loss = total_loss + dtw
                count += 1

        if not use_cuda_sdtw:
            total_loss = torch.tensor(0.0, device=attention_mask.device, dtype=torch.float32)
            count = 0

    if not use_cuda_sdtw:
        for b in range(batch_size):
            seq_len = int(lengths[b].item())
            if seq_len < 2:
                continue
            student_seq = student_layer[b, :seq_len]
            teacher_seq = teacher_layer[b, :seq_len]

            if unit_ids is not None:
                unit_row = unit_ids[b, :seq_len]
            else:
                unit_row = None

            if importance_weights is not None:
                weight_row = importance_weights[b, :seq_len]
            else:
                weight_row = None

            student_seq, pooled_weights = _pool_by_units(student_seq, unit_row, weight_row)
            teacher_seq, _ = _pool_by_units(teacher_seq, unit_row, None)

            if student_seq.numel() == 0 or teacher_seq.numel() == 0:
                continue

            pooled_len = student_seq.size(0)
            cost_s2t = _pairwise_cost_matrix(student_seq, teacher_seq, distance)
            cost_s2t = _apply_importance_weights(cost_s2t, pooled_weights)
            if band_source == "sdtw" and band_width > 0:
                cost_s2t = _apply_sdtw_band_penalty(
                    cost_s2t,
                    gamma,
                    window_size,
                    band_width,
                    band_penalty,
                    band_center_blend,
                    band_entropy_coef,
                    band_warmup_steps,
                    current_step,
                )
            s2t = _soft_dtw_banded(cost_s2t, gamma, window_size)

            if use_divergence:
                cost_s2s = _pairwise_cost_matrix(student_seq, student_seq, distance)
                cost_t2t = _pairwise_cost_matrix(teacher_seq, teacher_seq, distance)
                cost_s2s = _apply_importance_weights(cost_s2s, pooled_weights)
                cost_t2t = _apply_importance_weights(cost_t2t, pooled_weights)
                s2s = _soft_dtw_banded(cost_s2s, gamma, window_size)
                t2t = _soft_dtw_banded(cost_t2t, gamma, window_size)
                if normalize:
                    s2t = s2t / (2.0 * pooled_len)
                    s2s = s2s / (2.0 * pooled_len)
                    t2t = t2t / (2.0 * pooled_len)
                dtw = s2t - 0.5 * (s2s + t2t)
            else:
                dtw = s2t
                if normalize:
                    dtw = dtw / (2.0 * pooled_len)

            total_loss = total_loss + dtw
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=attention_mask.device, dtype=torch.float32)
    result = total_loss / count
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=attention_mask.device, dtype=torch.float32)
    return result


def velocity_field_loss(
    student_hiddens, 
    teacher_hiddens,
    velocity_field, 
    projector,
    teacher_schedule,
    student_schedule,
    attention_mask,
    device=0
):
    """
    Compute FRFD velocity field loss for rectified flow distillation.
    """
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    # Sample time t once per batch
    batch_size = student_hiddens[0].size(0)
    t = torch.rand(batch_size, 1, 1, device=device, dtype=torch.float32)
    num_distill_layers = len(teacher_schedule)
    
    # Loop over all distillation layers
    for j, (teacher_layer_idx, student_layer_idx) in enumerate(zip(teacher_schedule, student_schedule)):
        # Get hidden states for the current layer pair
        y_S = student_hiddens[student_layer_idx].to(device=f"cuda:{device}", dtype=torch.float32)
        y_T = teacher_hiddens[teacher_layer_idx].to(device=f"cuda:{device}", dtype=torch.float32)
        
        # Project student features to teacher's dimension
        y_S = projector(y_S)

        # Create interpolated features Y_t
        Y_t = (1 - t) * y_S + t * y_T
        
        # Compute target velocity
        target_velocity = y_T - y_S
        
        # Predict velocity using the velocity field model
        layer_indices = torch.tensor([j] * y_S.size(0), device=device, dtype=torch.long)
        predicted_velocity = velocity_field(Y_t, t.squeeze(1).squeeze(1), layer_indices)
        
        # Accumulate the MSE loss for this layer
        loss_per_token = F.mse_loss(predicted_velocity, target_velocity, reduction='none').mean(dim=-1)
        loss_per_token *= attention_mask
        loss = loss_per_token.sum() / attention_mask.sum()
        total_loss += loss / num_distill_layers
    
    return total_loss


def frfd_distillation_loss(
    student_hiddens,
    velocity_field,
    projector,
    student_schedule,
    attention_mask,
    num_distill_layers,
    cur_t=1,
    device=0
):
    """
    Compute FRFD rectified flow distillation loss.
    Exactly matches the original FRFD stage2 implementation.
    """
    # Calculate Rectified Flow Distillation Loss
    loss_rfd = 0
    # delta_t = 1.0 / (num_distill_layers - 1)
    
    if attention_mask.sum() > 0:
        for j in range(num_distill_layers):
            h_S_current = student_hiddens[student_schedule[j]].to(device=f"cuda:{device}", dtype=torch.float32)
            # h_S_next = student_hiddens[student_schedule[j+1]].to(device=f"cuda:{device}", dtype=torch.float32)
            
            actual_y_next = projector(h_S_current)
            with torch.no_grad():
                y_S_j = projector(h_S_current.detach())
                B, L, V = y_S_j.shape
                
                # Get ideal update from velocity field
                t = torch.full((B,), 0, device=device, dtype=torch.float32)
                layer_indices = torch.full((B,), j, device=device, dtype=torch.long)
                ideal_update = velocity_field(y_S_j, t, layer_indices)
                
                # Target for next layer: Euler step from current layer
                target_y_next = y_S_j + cur_t * ideal_update #* delta_t
            
            layer_loss = cosine_similarity_loss_masked(actual_y_next, target_y_next, attention_mask)
            loss_rfd += layer_loss / (num_distill_layers)
            
    else:
        loss_rfd = torch.tensor(0.0, device=device)
    
    return loss_rfd

def soft_label_distill_loss(student_logits, teacher_logits, mask, distill_temperature = 2.0):
    student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

    loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()

    return loss

def get_fdd_loss(t_hiddens, s_hiddens, mask, teacher, student, teacher_schedule, student_schedule):
    i = 0
    traj_loss, der_loss = 0.0, 0.0
    pre_s_hidden_logs, pre_t_hidden_logs = None, None
    # mask = (no_model_batch["label"] != -100).int()

    for s_idx, t_idx in zip(student_schedule, teacher_schedule):
        s_hidden = s_hiddens[s_idx]
        t_hidden = t_hiddens[t_idx]
        # if args.model_type == 'opt':
        #     s_decoder_proj = student.module.model.model.decoder.project_out
        #     if s_decoder_proj is not None:
        #         s_hidden = s_decoder_proj(s_hidden)

        #     t_decoder_proj = teacher.model.decoder.project_out
        #     if t_decoder_proj is not None:
        #         t_hidden = t_decoder_proj(t_hidden)

        s_hidden_logits = student.module.lm_head(s_hidden)
        t_hidden_logits = teacher.lm_head(t_hidden)
        # traj_loss += forward_kl(s_hidden_logits, t_hidden_logits, no_model_batch)
        traj_loss += soft_label_distill_loss(s_hidden_logits, t_hidden_logits, mask)

        # FIX #5: Cast to float32 to prevent fp16 overflow in log_softmax
        s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1, dtype=torch.float32)
        t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1, dtype=torch.float32)

        if i > 0:
            # FIX #2: Detach previous logs to stop gradient flow through all layers
            # This prevents exponential gradient accumulation across layers
            delta_hidden_student = s_hidden_logs - pre_s_hidden_logs.detach()
            delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs.detach()

            cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-6)
            # Clamp cosine similarity to valid range to prevent numerical errors
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            cos_sim_loss = 1 - cos_sim
            cos_sim_loss = (cos_sim_loss * mask).sum() / (mask.sum() + 1e-8)

            # FIX #4: Clamp derivative loss to prevent unbounded growth
            cos_sim_loss = torch.clamp(cos_sim_loss, 0.0, 10.0)

            der_loss += cos_sim_loss

        pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs

        i += 1

    # FIX #3: Safe division to avoid division by zero
    if i == 0:
        return torch.tensor(0.0, device=mask.device, dtype=torch.float32)

    avg_traj_loss = traj_loss / i
    avg_der_loss = der_loss / max(i - 1, 1)  # Avoid division by zero when i=1

    return avg_traj_loss + avg_der_loss

def get_csd_loss(logits, teacher_logits, no_model_batch, mode="SS"):
    student_probs = F.softmax(logits, dim=-1)
    teacher_probs = F.softmax(logits, dim=-1)
    
    assert type(mode) == str and len(mode) == 2, "wrong mode format"
    def get_weight_func(mode):
        if mode == "S":
            return torch.clone(student_probs).detach()
        if mode == "T":
            return torch.clone(teacher_probs).detach()
        if mode == "U":
            return torch.ones(student_probs.shape) / student_probs.shape[-1]
        raise Exception("unsupported mode")
    # if mode == "SS":
    #     w1 = torch.clone(student_probs).detach()
    #     w2 = torch.clone(student_probs).detach()
    # elif mode == "TS":
    #     w1 = torch.clone(teacher_probs).detach()
    #     w2 = torch.clone(student_probs).detach()
    w1 = get_weight_func(mode[0])
    w2 = get_weight_func(mode[1])
    
    # student_logits_mean_w1 = torch.sum(w1 * logits.detach(), dim=-1)
    # student_logits_mean_w2 = torch.sum(w2 * logits.detach(), dim=-1)
    # teacher_logits_mean_w1 = torch.sum(w1 * teacher_logits, dim=-1)
    # teacher_logits_mean_w2 = torch.sum(w2 * teacher_logits, dim=-1)
    
    # student_logits_norm_w1 = logits.detach() - torch.sum(w1 * logits.detach(), dim=-1)
    # student_logits_norm_w2 = logits.detach() - torch.sum(w2 * logits.detach(), dim=-1)
    # teacher_logits_norm_w1 = teacher_logits - torch.sum(w1 * teacher_logits, dim=-1)
    # teacher_logits_norm_w2 = teacher_logits - torch.sum(w2 * teacher_logits, dim=-1)
    
    s_t_diff = logits - teacher_logits
    
    w_grad = w1 * (s_t_diff - torch.sum(w2 * s_t_diff, dim=-1)) + w2 * (s_t_diff - torch.sum(w1 * s_t_diff, dim=-1))
    csd_loss_per_token = (w_grad.detach() * logits / 2).sum(dim=-1)
    
    mask = (no_model_batch["label"] != -100).int()
    csd_loss = torch.sum(csd_loss_per_token.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return csd_loss