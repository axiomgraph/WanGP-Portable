"""Parallel Decoding Distillation heads and schedules for MiniMax H3."""

import torch
import torch.nn as nn
import torch.nn.functional as F


PDD_NUM_STEPS = 32
PDD_BLOCK_SIZE = 4


def shifted_sigma(shift, sigma):
    return float(shift) * sigma / (1.0 + (float(shift) - 1.0) * sigma)


def pdd_time_grid(shift, num_steps=PDD_NUM_STEPS, max_t=1.0):
    """Times of the fine grid's nodes: the shifted noise levels of num_steps equal base intervals, the largest scaled to max_t
    (FastGen's grid on [0, max_t]: MiniMax H3's PDD uses 1.0, FastH3 Omni 0.999)."""
    sigma = torch.linspace(1.0, 0.0, int(num_steps) + 1, dtype=torch.float64, device="cpu")
    return 1.0 - float(max_t) * shifted_sigma(shift, sigma)


def pdd_sampling_plan(step_sizes, start, block_size=PDD_BLOCK_SIZE):
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype, device=step_sizes.device)
    span = step_sizes[start:start + block_size].sum()
    plan[0, start:start + block_size] = step_sizes[start:start + block_size] / span
    return plan


def pdd_sampling_plans(shift, num_steps=PDD_NUM_STEPS, block_size=PDD_BLOCK_SIZE, max_t=1.0):
    num_steps, block_size = int(num_steps), int(block_size)
    if block_size < 1 or num_steps % block_size:
        raise ValueError(f"PDD num_steps={num_steps} must be divisible by block_size={block_size}")
    step_sizes = pdd_time_grid(shift, num_steps, max_t).diff()
    return torch.cat([pdd_sampling_plan(step_sizes, start, block_size) for start in range(0, num_steps, block_size)])


def pdd_sampling_plans_for_sigmas(sigmas, shift, num_steps=PDD_NUM_STEPS, max_t=1.0):
    """Fuse fine PDD intervals over arbitrary descending shifted-sigma ranges (sigmas on the grid scaled to max_t)."""
    sigmas = torch.as_tensor(sigmas).flatten().detach().to(device="cpu", dtype=torch.float64, non_blocking=False)
    times = 1.0 - sigmas
    fine = pdd_time_grid(shift, num_steps, max_t)
    starts, ends = fine[:-1], fine[1:]
    plans = []
    for start, end in zip(times[:-1], times[1:]):
        span = end - start
        if span <= 0:
            raise ValueError("MiniMax H3 PDD requires strictly descending sigma boundaries")
        overlap = (torch.minimum(ends, end) - torch.maximum(starts, start)).clamp_min_(0.0)
        if not torch.isclose(overlap.sum(), span, rtol=1e-6, atol=1e-8):
            raise ValueError(f"MiniMax H3 PDD sigma interval [{float(1.0 - start):.6f}, {float(1.0 - end):.6f}] is outside its trained grid")
        plans.append(overlap / span)
    return torch.stack(plans)


class MiniMaxH3ParallelHead(nn.Module):
    """One output head per PDD interval, fused into one velocity at runtime. Head 0 is the base projection, an ordinary Linear
    (`linear`) so that a LoRA of the base model applies to it through the shared LoRA path; `heads` holds the other heads relative
    to it (head n - head 0). As the plan weights sum to one, a delta on the base projection is a delta on every fused head."""

    def __init__(self, in_features, out_features, num_steps=PDD_NUM_STEPS, bias=True, device=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_steps = int(num_steps)
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias, dtype=torch.float32, device=device)
        self.heads = nn.Parameter(torch.empty(self.num_steps - 1, self.out_features, self.in_features, dtype=torch.float32, device=device))
        self.head_biases = nn.Parameter(torch.empty(self.num_steps - 1, self.out_features, dtype=torch.float32, device=device)) if bias else None
        self.plan = torch.zeros(1, self.num_steps)
        self.plan[0, 0] = 1.0

    def set_plan(self, plan):
        if plan.ndim != 2 or plan.shape != (1, self.num_steps):
            raise ValueError(f"A MiniMax H3 PDD plan must have shape (1, {self.num_steps}), got {tuple(plan.shape)}")
        self.plan = plan

    def forward(self, hidden_states):
        plan = self.plan.to(device=self.heads.device, dtype=self.heads.dtype)[0, 1:]
        weight = torch.einsum("n,noi->oi", plan, self.heads)
        bias = None if self.head_biases is None else torch.einsum("n,no->o", plan, self.head_biases)
        return self.linear(hidden_states) + F.linear(hidden_states, weight, bias)


__all__ = ["MiniMaxH3ParallelHead", "PDD_BLOCK_SIZE", "PDD_NUM_STEPS", "pdd_sampling_plan", "pdd_sampling_plans",
           "pdd_sampling_plans_for_sigmas", "pdd_time_grid"]
