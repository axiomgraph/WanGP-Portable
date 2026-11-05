# Adapted from ViGeo (https://github.com/aigc3d/ViGeo, commit 78100ce, vigeo/: vigeo.py, layers/dinov2.py, layers/attention.py,
# layers/block.py, layers/rope.py, transformer_decoder.py, conv_head.py), Apache License 2.0; the DINOv2 layers Copyright (c) Meta
# Platforms, Inc. and affiliates, Apache License 2.0; and from Alaya-EVOKE (https://github.com/AlayaLab/Evoke,
# evoke/modules/geometric_state/vigeo_cloud.py), Copyright 2026 The Evoke Team, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""ViGeo, the video geometry model EVOKE writes its world memory with: a DINOv2-g trunk (40 blocks; from block 13 every other block
attends across the frames of a window and keeps a budgeted cache of earlier windows' keys and values, so one stream shares one coordinate
frame), a 5-block geometry decoder and the point head, whose last channel is log depth. EVOKE reads only the depth: the ray, normal, mask,
confidence and camera heads are not built. The trunk and decoder run under FP16 autocast and the head in FP32, as upstream."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort

PATCH, EMBED, DEPTH, HEADS, ALT_START, REGISTERS = 14, 1536, 40, 24, 13, 4
EXTRA_TOKENS = 6  # cls, camera and 4 registers before the patch tokens
PROCESS_RES = 644  # long side of the frames ViGeo sees (multiples of the 14 pixel patch)
CHUNK_SIZE, CACHE_KEEP_FRAMES, MIN_KEEP_FRAMES = 16, 6, 4  # EVOKE's streaming settings (mode "chunk")
DEPTH_MEDIAN_TARGET = 5.0  # i2v depth scale: the first window's median depth becomes 5 pose units, locked for the session
IMAGE_MEAN, IMAGE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def fit_patch_res(height, width, long_side=PROCESS_RES):
    scale = float(long_side) / float(max(height, width))
    return max(PATCH, int(round(height * scale / PATCH)) * PATCH), max(PATCH, int(round(width * scale / PATCH)) * PATCH)


class Rope2D(nn.Module):
    """Rotate-half 2D RoPE (frequency 100): the first half of the head dimension rotated by the row position, the second by the column."""

    def __init__(self, frequency=100.0):
        super().__init__()
        self.frequency, self.cache = frequency, {}

    def _components(self, dim, positions, device, dtype):
        key = (dim, positions, device, dtype)
        if key not in self.cache:
            inv_freq = 1.0 / (self.frequency ** (torch.arange(0, dim, 2, device=device).float() / dim))
            angles = torch.einsum("i,j->ij", torch.arange(positions, device=device, dtype=inv_freq.dtype), inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.cache[key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.cache[key]

    @staticmethod
    def _rotate(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def forward(self, tokens, positions):
        """tokens (B, heads, N, D), positions (B, N, 2) long."""
        dim = tokens.size(-1) // 2
        cos, sin = self._components(dim, int(positions.max()) + 1, tokens.device, tokens.dtype)
        parts = []
        for part, axis in zip(tokens.chunk(2, dim=-1), (positions[..., 0], positions[..., 1])):
            c, s = F.embedding(axis, cos)[:, None], F.embedding(axis, sin)[:, None]
            parts.append(part * c + self._rotate(part) * s)
        return torch.cat(parts, dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads, qk_norm=False, rope=None):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rope = rope
        self.anchor_tokens = 0  # the first streamed frame's tokens, never evicted

    @staticmethod
    def evict(k, v, budget, anchors):
        """The cache (B, heads, N, D) cut to budget tokens: the anchors, then the candidates least similar to their mean (per head), in
        order. Returns (k, v, mean similarity) (None without eviction)."""
        tokens = k.shape[2]
        keep = budget - anchors
        if tokens <= budget or keep <= 0:
            return k, v, None
        anchor_k, candidate_k = k.split([anchors, tokens - anchors], dim=2)
        anchor_v, candidate_v = v.split([anchors, tokens - anchors], dim=2)
        normed = F.normalize(candidate_k, p=2, dim=-1)
        scores = torch.sum(normed * normed.mean(dim=2, keepdim=True), dim=-1)
        indices = torch.topk(-scores, k=keep, dim=-1).indices.sort(dim=-1).values.unsqueeze(-1).expand(*k.shape[:2], keep, k.shape[-1])
        return (torch.cat([anchor_k, torch.gather(candidate_k, 2, indices)], dim=2), torch.cat([anchor_v, torch.gather(candidate_v, 2, indices)], dim=2),
                scores.mean().item())

    def forward(self, x, pos=None, cache=None, budget=None, tokens_per_frame=None):
        """x (B, N, C). Without cache (None) a plain self-attention; with cache ([] for the first window, else [(k, v)]) the window's
        queries attend to the cached keys / values and its own, and cache is replaced by the evicted union. Returns (y, mean eviction
        score or None)."""
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, pos).to(v.dtype), self.rope(k, pos).to(v.dtype)
        score = None
        if cache is not None:
            if self.anchor_tokens == 0:
                self.anchor_tokens = tokens_per_frame
            if cache:
                k, v = torch.cat([cache[0][0], k], dim=2), torch.cat([cache[0][1], v], dim=2)
            qkv = [q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)]
            del q
            y = pay_attention(qkv)
            if budget is not None and k.shape[2] > budget:  # after the attention, over the union (the source's order)
                k, v, score = self.evict(k, v, budget, self.anchor_tokens)
            cache[:] = [(k, v)]
            del k, v
        else:
            qkv = [q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)]
            del q, k, v
            y = pay_attention(qkv)
        return self.proj(y.reshape(batch, tokens, channels)), score


class Block(nn.Module):
    def __init__(self, dim, heads, swiglu, qk_norm=False, rope=None, layer_scale=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, qk_norm, rope)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        if swiglu:
            hidden = (int(dim * 4 * 2 / 3) + 7) // 8 * 8
            self.mlp = nn.Module()
            self.mlp.w12, self.mlp.w3 = nn.Linear(dim, 2 * hidden), nn.Linear(hidden, dim)
        else:
            self.mlp = nn.Module()
            self.mlp.fc1, self.mlp.fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)
        self.swiglu = swiglu
        if layer_scale:
            self.ls1, self.ls2 = nn.Module(), nn.Module()
            self.ls1.gamma, self.ls2.gamma = nn.Parameter(torch.empty(dim)), nn.Parameter(torch.empty(dim))
        self.layer_scale = layer_scale

    def _mlp(self, x):
        if self.swiglu:
            gate, value = self.mlp.w12(x).chunk(2, dim=-1)
            return self.mlp.w3(F.silu(gate) * value)
        return self.mlp.fc2(F.gelu(self.mlp.fc1(x)))

    def forward(self, x, pos=None, cache=None, budget=None, tokens_per_frame=None):
        y, score = self.attn(self.norm1(x), pos, cache, budget, tokens_per_frame)
        x = x + (y * self.ls1.gamma if self.layer_scale else y)
        y = self._mlp(self.norm2(x))
        return x + (y * self.ls2.gamma if self.layer_scale else y), score


class DINOv2(nn.Module):
    """The ViT-g/14 trunk with 4 registers and a camera token, local (per frame) and global (per window) attention from block 13."""

    def __init__(self, rope):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, EMBED, kernel_size=PATCH, stride=PATCH)
        self.cls_token = nn.Parameter(torch.empty(1, 1, EMBED))
        self.camera_token = nn.Parameter(torch.empty(1, 2, EMBED))
        self.pos_embed = nn.Parameter(torch.empty(1, (518 // PATCH) ** 2 + 1, EMBED))
        self.register_tokens = nn.Parameter(torch.empty(1, REGISTERS, EMBED))
        self.mask_token = nn.Parameter(torch.empty(1, EMBED))
        self.blocks = nn.ModuleList([Block(EMBED, HEADS, True, qk_norm=index >= ALT_START, rope=rope if index >= ALT_START else None) for index in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED, eps=1e-6)
        self.global_blocks = [index for index in range(DEPTH) if index >= ALT_START and index % 2 == 1]
        self.register_buffer("last_scores", torch.zeros(len(self.global_blocks)), persistent=False)

    def reset_cache_state(self):
        for block in self.blocks:
            block.attn.anchor_tokens = 0
        self.last_scores.zero_()

    def _pos_embed(self, height, width):
        """The 37x37 position embedding interpolated (bicubic, antialiased, offset 0.1) to the patch grid (FP32)."""
        pos = self.pos_embed.float()
        grid, dim = int(math.sqrt(pos.shape[1] - 1)), pos.shape[-1]
        rows, cols = height // PATCH, width // PATCH
        patches = F.interpolate(pos[:, 1:].reshape(1, grid, grid, dim).permute(0, 3, 1, 2), mode="bicubic", antialias=True,
                                scale_factor=(float(rows + 0.1) / grid, float(cols + 0.1) / grid))
        return torch.cat((pos[:, :1], patches.permute(0, 2, 3, 1).view(1, -1, dim)), dim=1)

    def forward(self, x, pos, caches, budget):
        """x (B, S, 3, H, W) normalized frames, pos (B, S, tokens, 2); caches: per global block a list (see Attention), or None for an
        offline window. Returns the concatenated normed outputs of the last two blocks (B, S, tokens, 2 * EMBED)."""
        batch, frames, _, height, width = x.shape
        x = self.patch_embed.proj(x.flatten(0, 1)).flatten(2).transpose(1, 2)
        # under autocast the patch embedding is FP16 and the FP32 cls token promotes the residual stream to FP32, as in the source
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self._pos_embed(height, width).to(x.dtype)
        x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1).unflatten(0, (batch, frames))
        streaming = caches is not None
        budgets = [None] * len(self.global_blocks)
        if streaming and budget:
            budgets = (torch.softmax((1.0 - self.last_scores) / 0.5, dim=0) * budget).int().tolist()
        outputs, scores, global_index = [], [], 0
        for index, block in enumerate(self.blocks):
            if index == ALT_START:  # the camera token after the cls token: the first frame of a stream gets its own
                continued = streaming and bool(caches[0])
                camera = self.camera_token[:, 1:2] if continued else self.camera_token[:, 0:1]
                camera = torch.cat([camera, self.camera_token[:, 1:2].expand(-1, frames - 1, -1)], dim=1) if not continued and frames > 1 else camera.expand(-1, frames, -1)
                x = torch.cat((x[:, :, :1], camera.expand(batch, -1, -1).unsqueeze(2).to(x.dtype), x[:, :, 1:]), dim=2)
            tokens = x.shape[2]
            is_global = index >= ALT_START and index % 2 == 1
            block_pos = None if index < ALT_START else pos
            if is_global:
                y, score = block(x.flatten(1, 2), None if block_pos is None else block_pos.flatten(1, 2),
                                 caches[global_index] if streaming else None, budgets[global_index], tokens)
                x = y.unflatten(1, (frames, tokens))
                if streaming:
                    scores.append(self.last_scores[global_index].item() if score is None else score)
                global_index += 1
            else:
                y, _ = block(x.flatten(0, 1), None if block_pos is None else block_pos.flatten(0, 1))
                x = y.unflatten(0, (batch, frames))
            if index >= DEPTH - 2:
                outputs.append(x)
            check_abort()
        if streaming and scores:
            self.last_scores.copy_(torch.tensor(scores, device=self.last_scores.device, dtype=self.last_scores.dtype))
        final = torch.cat(outputs, dim=-1)
        return torch.cat([self.norm(final[..., :EMBED]), self.norm(final[..., EMBED:])], dim=-1)


class Decoder(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.projects = nn.Linear(2 * EMBED, 1024)
        self.blocks = nn.ModuleList([Block(1024, 16, False, rope=rope, layer_scale=False) for _ in range(5)])
        self.linear_out = nn.Linear(1024, 1024)

    def forward(self, hidden, pos):
        """hidden (B, S, patches, 2 * EMBED), pos (B, S, tokens, 2) -> (B * S, patches, 1024)."""
        pos = pos[:, :, -hidden.shape[2]:].flatten(0, 1)
        hidden = self.projects(hidden.flatten(0, 1))
        for block in self.blocks:
            hidden, _ = block(hidden, pos)
        return self.linear_out(hidden)


def view_plane_uv(width, height, aspect_ratio, dtype, device):
    span_x, span_y = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5, 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    return torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Identity(), nn.ReLU(), nn.Conv2d(channels, hidden, 3, padding=1, padding_mode="replicate"),
                                    nn.Identity(), nn.ReLU(), nn.Conv2d(hidden, channels, 3, padding=1, padding_mode="replicate"))

    def forward(self, x):
        return x + self.layers(x)


class PointHead(nn.Module):
    """ViGeo's ConvHead (1024 -> 256 -> 128 -> 64 with view-plane uv channels, 2 residual blocks per stage) with its 3 output channels."""

    def __init__(self, dims=(256, 128, 64)):
        super().__init__()
        def upsampler(cin, cout):
            return nn.Sequential(nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2), nn.Conv2d(cout, cout, 3, padding=1, padding_mode="replicate"))
        self.upsample_blocks = nn.ModuleList([nn.Sequential(upsampler(cin + 2, cout), *(ResidualConvBlock(cout, 2 * cout) for _ in range(2)))
                                              for cin, cout in zip((1024,) + dims[:-1], dims)])
        self.output_block = nn.ModuleList([nn.Sequential(nn.Conv2d(dims[-1] + 2, 32, 3, padding=1, padding_mode="replicate"), nn.ReLU(),
                                                         nn.Conv2d(32, 3, 1, padding=0, padding_mode="replicate"))])

    def _uv(self, x, aspect):
        uv = view_plane_uv(x.shape[-1], x.shape[-2], aspect, x.dtype, x.device).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, uv], dim=1)

    def forward(self, hidden, patch_h, patch_w, chunk=8):
        outputs = []
        for start in range(0, hidden.shape[0], chunk):  # the source's chunks of 8 frames
            x = hidden[start:start + chunk].permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous()
            for block in self.upsample_blocks:
                x = block(self._uv(x, patch_w / patch_h))
            x = F.interpolate(x, (patch_h * PATCH, patch_w * PATCH), mode="bilinear", align_corners=False)
            outputs.append(self.output_block[0](self._uv(x, patch_w / patch_h)))
        return torch.cat(outputs)


class ViGeo(nn.Module):
    def __init__(self):
        super().__init__()
        rope = Rope2D(100.0)
        self.pretrained = DINOv2(rope)
        self.decoder = Decoder(rope)
        self.point_head = PointHead()
        self.register_buffer("image_mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1))

    @staticmethod
    def _positions(frames, patch_h, patch_w, device):
        grid = torch.cartesian_prod(torch.arange(patch_h, device=device), torch.arange(patch_w, device=device)).view(1, -1, 2) + 1
        return torch.cat([torch.zeros(1, EXTRA_TOKENS, 2, dtype=grid.dtype, device=device), grid], dim=1).expand(frames, -1, -1).unsqueeze(0)

    def forward(self, frames, caches=None, budget=0):
        """frames (S, 3, H, W) in [0, 1] at a multiple of the patch -> depth (S, H, W) FP32 (affine-invariant)."""
        count, _, height, width = frames.shape
        patch_h, patch_w = height // PATCH, width // PATCH
        x = ((frames - self.image_mean) / self.image_std).unsqueeze(0)
        pos = self._positions(count, patch_h, patch_w, frames.device)
        with torch.autocast(device_type=frames.device.type, dtype=torch.float16, enabled=frames.device.type == "cuda"):
            hidden = self.pretrained(x, pos, caches, budget)
            hidden = self.decoder(hidden[:, :, EXTRA_TOKENS:], pos)
        with torch.autocast(device_type=frames.device.type, dtype=torch.float32, enabled=frames.device.type == "cuda"):
            logits = self.point_head(hidden.float(), patch_h, patch_w)
        return logits[:, 2].float().clamp(-10, 10).exp()


class DepthStream:
    """ViGeo over the windows of a world, as EVOKE's chunk mode: windows of frames go through the trunk with the global blocks' caches kept
    across calls (the first frame of the stream as anchor, a budget of max(4, 6) frames per global block), depth scaled once for the
    session (depth_median: the first window's median becomes 5 pose units)."""

    def __init__(self, model):
        self.model, self.caches, self.scale = model, None, None

    def reset(self):
        self.caches, self.scale = None, None
        self.model.pretrained.reset_cache_state()

    @staticmethod
    def _prepare(frames):
        """(N, 3, H, W) frames in [0, 1] at ViGeo's processing size (bilinear), their size."""
        size = fit_patch_res(*frames.shape[-2:])
        return (F.interpolate(frames, size=size, mode="bilinear", align_corners=False) if tuple(frames.shape[-2:]) != size else frames), size

    @torch.no_grad()
    def window(self, frames):
        """Depth (N, h, w) FP32 in pose units of a window of frames (N, 3, H, W) in [0, 1] (N >= 3), the frames ViGeo saw (N, 3, h, w) and
        (h, w); continues the stream."""
        frames, (height, width) = self._prepare(frames.float())
        if self.caches is None:
            self.model.pretrained.reset_cache_state()
            self.caches = [[] for _ in self.model.pretrained.global_blocks]
        tokens = (height // PATCH) * (width // PATCH) + EXTRA_TOKENS
        budget = len(self.model.pretrained.global_blocks) * max(MIN_KEEP_FRAMES, CACHE_KEEP_FRAMES) * tokens
        depth = torch.cat([self.model(frames[start:start + CHUNK_SIZE], self.caches, budget) for start in range(0, frames.shape[0], CHUNK_SIZE)])
        if self.scale is None:
            self.scale = float(torch.median(depth).item()) / DEPTH_MEDIAN_TARGET
        return depth / self.scale, frames, (height, width)

    @torch.no_grad()
    def single(self, frame):
        """Depth of one frame (3, H, W) in [0, 1] with its own median scale (EVOKE's chunk 0 reference warp, before any window): the
        stream is reset before and after, as in the source."""
        frames, (height, width) = self._prepare(frame.float().unsqueeze(0))
        self.reset()
        depth = self.model(frames, None)
        self.reset()
        return depth[0] / (float(torch.median(depth).item()) / DEPTH_MEDIAN_TARGET), frames[0], (height, width)
