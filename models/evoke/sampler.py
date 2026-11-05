# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke, evoke/scheduler/scheduling_evoke.py, evoke/pipelines/pipeline_evoke.py
# stage2_sample / sample_block_noise, evoke/utils/utils_base.py apply_schedule_shift), Copyright 2025 The Helios Team and The HuggingFace
# Team, Copyright 2026 The Evoke Team, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The distilled sampler of EVOKE-Turbo: a three-stage latent pyramid (1/4, 1/2 then full resolution), one forward per stage. The noise
is drawn at full resolution and downsampled; each stage starts from the previous stage's x0 prediction upsampled 2x and re-noised with
noise correlated inside 2x2 patches. The stage's sigma follows the source's EvokeScheduler (shift 1, gamma 1/3, stage ranges of 1/3)
with its dynamic exponential shift by sequence length; one UniPC step to sigma 0 is x0 = x - sigma * v."""
import math

import numpy as np
import torch
import torch.nn.functional as F

STAGES, GAMMA, TRAIN_STEPS, STAGE_RANGE = 3, 1.0 / 3.0, 1000, (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
BASE_SEQ, MAX_SEQ, BASE_SHIFT, MAX_SHIFT, EXP_MAX = 256, 4096, 0.5, 1.15, 7.0


def _stage_tables():
    """(original start sigma, (timestep max, timestep min)) of each stage (EvokeScheduler.init_sigmas_for_each_stage, version v1)."""
    sigmas = np.flip(1.0 - np.linspace(1, 1 / TRAIN_STEPS, TRAIN_STEPS + 1))[:-1].copy()
    timesteps = sigmas * TRAIN_STEPS
    starts, distances = [], []
    for stage in range(STAGES):
        start = sigmas[max(int(STAGE_RANGE[stage] * TRAIN_STEPS), 0)]
        stop_index = min(int(STAGE_RANGE[stage + 1] * TRAIN_STEPS), TRAIN_STEPS)
        end = sigmas[stop_index] if stop_index < TRAIN_STEPS else 0.0
        starts.append(start)
        if stage:
            original = 1 - start
            start = 1 - original / (math.sqrt(1 + 1 / GAMMA) * (1 - original) + original)
        distances.append(start - end)
    total, tables = sum(distances), []
    for stage in range(STAGES):
        ratio_start = sum(distances[:stage]) / total if stage else 0.0
        ratio_end = 0.9999999999999999 if stage == STAGES - 1 else sum(distances[:stage + 1]) / total
        stage_timesteps = np.linspace(min(timesteps[int(ratio_start * TRAIN_STEPS)], 999), timesteps[min(int(ratio_end * TRAIN_STEPS), TRAIN_STEPS - 1)], TRAIN_STEPS + 1)[:-1]
        tables.append((starts[stage], (stage_timesteps[0], stage_timesteps[-1])))
    return tables


_TABLES = _stage_tables()


def stage_step(stage, latent_shape):
    """(sigma, model timestep) of a stage's single step for noise latents of latent_shape (..., frames, height, width): the stage's sigma
    0.999 shifted by mu = exp(min(shift(frames * height * width / 4), ln 7)), its timestep in the stage's range, truncated as the source's
    int64 cast."""
    frames, height, width = latent_shape[-3:]
    sequence = frames * height * width // 4
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ)
    mu = math.exp(min(sequence * slope + (BASE_SHIFT - slope * BASE_SEQ), math.log(EXP_MAX)))
    sigma = torch.tensor(0.999, dtype=torch.float64)
    sigma = float((sigma * mu) / (1 + (mu - 1) * sigma))
    timestep_max, timestep_min = _TABLES[stage][1]
    return sigma, int(timestep_min + sigma * (timestep_max - timestep_min))


def renoise_coefficients(stage):
    """(alpha, beta) of the re-noising before stage > 0: x = alpha * upsampled x0 + beta * block noise."""
    original = 1 - _TABLES[stage][0]
    alpha = 1 / (math.sqrt(1 + 1 / GAMMA) * (1 - original) + original)
    return alpha, alpha * (1 - original) / math.sqrt(GAMMA)


def downsample_noise(noise, stages=STAGES):
    """The full resolution noise (1, C, T, H, W) at the first stage's size: bilinear halvings, each times 2 (unit variance)."""
    frames = noise.shape[2]
    noise = noise.transpose(1, 2).flatten(0, 1)
    height, width = noise.shape[-2:]
    for _ in range(stages - 1):
        height, width = height // 2, width // 2
        noise = F.interpolate(noise, size=(height, width), mode="bilinear") * 2
    return noise.unflatten(0, (-1, frames)).transpose(1, 2)


def upsample(latents):
    """Nearest 2x upsampling of (1, C, T, H, W) latents to the next stage's size."""
    frames = latents.shape[2]
    latents = F.interpolate(latents.transpose(1, 2).flatten(0, 1), scale_factor=2, mode="nearest")
    return latents.unflatten(0, (-1, frames)).transpose(1, 2)


def block_noise(shape, generator, device):
    """Noise of shape (1, C, T, H, W), correlated inside each 2x2 patch with covariance (1 + gamma) I - gamma 11^T."""
    batch, channels, frames, height, width = shape
    covariance = torch.eye(4, device=device) * (1 + GAMMA) - torch.ones(4, 4, device=device) * GAMMA + torch.eye(4, device=device) * 1e-8
    lower = torch.linalg.cholesky(covariance.float())
    z = torch.randn(batch * channels * frames * (height // 2) * (width // 2), 4, generator=generator, device=device)
    noise = (z @ lower.T).view(batch, channels, frames, height // 2, width // 2, 2, 2)
    return noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(shape)


def sample(predict, noise, generator, on_stage=None):
    """The chunk's clean latents (FP32) from full resolution noise (1, C, T, H, W) FP32: predict(latents, stage, timestep) returns the
    velocity at that stage (FP32, the latents' shape); on_stage(stage, latents) receives each stage's clean latents (coarse to full size)."""
    latents = downsample_noise(noise)
    for stage in range(STAGES):
        if stage:
            alpha, beta = renoise_coefficients(stage)
            latents = upsample(latents)
            latents = alpha * latents + beta * block_noise(latents.shape, generator, latents.device)
        sigma, timestep = stage_step(stage, latents.shape)
        velocity = predict(latents, stage, timestep)
        latents = latents - sigma * velocity
        del velocity
        if on_stage is not None:
            on_stage(stage, latents)
    return latents
