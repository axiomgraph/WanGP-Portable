# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""VSA for the H3 transformer: trained tile selection with a gated coarse branch, one policy per transformer and one attention per layer.
Prefix tiles (text, audio, image references) are attended by every query; every video query tile keeps its own top-k of each video
region (the reference videos at the reference keep rate, the generated video at the trained sparsity)."""
import math

import torch

from mmgp import offload
from shared.kernels import kernel_policy
from ..veda import preprocess
from . import kernel, sparse_cuda, sparse_int8, tiling
from . import preprocess as fused


class VSAPolicy:
    def __init__(self, inference):
        self.sparsity = float(inference["vsa_sparsity"])
        self.tile = int(inference["vsa_tile_size"])
        if self.tile not in tiling.TILE_SHAPES:
            raise ValueError(f"H3 VSA supports {sorted(tiling.TILE_SHAPES)}-token tiles, the checkpoint was trained with {self.tile}")
        self.ref_keep_rate = float(inference.get("vsa_ref_keep_rate") or 1.0 - self.sparsity)
        self.enabled = False
        self.int8_qk = False
        self.cuda_kind = None
        self.reset()

    def reset(self):
        self.signature = None
        self.layout = None
        self.index_cache = {}

    def begin_forward(self, layout, grid, payload, device, dtype):
        self.enabled = offload.shared_state.get("_attention") == "vsa"
        if not self.enabled:
            return
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8 or dtype != torch.bfloat16:
            raise ValueError("H3 VSA requires BF16 on an NVIDIA SM80+ GPU with Triton")
        if payload.get("target_video_order") is not None:
            raise ValueError("H3 VSA does not support grouped-row mask denoising")
        signature = (payload["layout_signature"], device)
        if signature == self.signature:
            return
        self.reset()
        self.signature = signature
        regions = [(start, region_grid) for start, _, region_grid in layout.reference_video_spans] + [(layout.sequence_length - math.prod(grid), grid)]
        self.layout = tiling.build_tile_layout(layout.sequence_length, layout.dense_spans, regions, self.tile, self.sparsity, self.ref_keep_rate, device)
        self.int8_qk = kernel_policy.allow_approximate()
        self.cuda_kind = sparse_cuda.kernel_kind(device) if self.int8_qk else None
        kernels = (f"WGPKernels CUDA INT8 QK / {self.cuda_kind.upper()} PV" if self.cuda_kind else "Triton INT8 QK / FP16 PV") if self.int8_qk else "Triton exact BF16"
        target = self.layout.regions[-1]
        references = "".join(f", reference video {i + 1}: {r.keep}/{r.stop - r.start} tiles" for i, r in enumerate(self.layout.regions[:-1]))
        print(f"[MiniMax H3] VSA Attention ({kernels}) - {self.tile}-token tiles, {self.layout.n_prefix_tiles} prefix tiles, generated video: "
              f"{target.keep}/{target.stop - target.start} tiles kept per query tile ({100 * (1 - target.keep / (target.stop - target.start)):.0f}% sparse){references}")

    def head_range(self, start, stop, device):
        key = ("heads", start, stop)
        if key not in self.index_cache:
            self.index_cache[key] = torch.arange(start, stop, device=device)
        return self.index_cache[key]

    def index_lut(self, heads, device):
        """Key-block lists of the WGPKernels kernel for a head count: lut [H', n_blocks, n_blocks] int32 (ascending block ids as
        increments, the first absolute) and counts [H', n_blocks]. The prefix query blocks' rows (every block, consecutive) and the
        counts are constant; the video query blocks' rows are rewritten by each call (same stream, so each call reads what it wrote)."""
        key = ("lut", heads)
        if key not in self.index_cache:
            layout = self.layout
            prefix_blocks = layout.n_prefix_tiles * layout.blocks_per_tile
            lut = torch.ones((heads, layout.n_blocks, layout.n_blocks), dtype=torch.int32, device=device)
            lut[:, :, 0] = 0
            count = torch.full((heads, layout.n_blocks), layout.n_blocks, dtype=torch.int32, device=device)
            count[:, prefix_blocks:] = prefix_blocks + layout.keep_total * layout.blocks_per_tile
            self.index_cache[key] = (lut, count)
        return self.index_cache[key]

    def block_index(self, heads, top):
        """Absolute key-block lists of the Triton kernels: (index [H', n_video_blocks, W] int32, count [H', n_video_blocks]) for the
        video query blocks from top [H', n_video_tiles, keep_total] (kept tile ids), and the dense prefix query blocks' lists."""
        layout = self.layout
        bpt, device = layout.blocks_per_tile, top.device
        prefix_blocks = layout.n_prefix_tiles * bpt
        kept = (top.unsqueeze(-1) * bpt + torch.arange(bpt, device=device)).flatten(-2).sort(dim=-1).values.to(torch.int32)
        prefix_ids = torch.arange(prefix_blocks, dtype=torch.int32, device=device).expand(heads, top.shape[1], prefix_blocks)
        index = torch.cat((prefix_ids, kept), dim=-1).repeat_interleave(bpt, dim=1).contiguous()
        count = torch.full((heads, index.shape[1]), index.shape[2], dtype=torch.int32, device=device)
        key = ("prefix", heads)
        if key not in self.index_cache:
            dense = torch.arange(layout.n_blocks, dtype=torch.int32, device=device).expand(heads, prefix_blocks, layout.n_blocks).contiguous()
            self.index_cache[key] = (dense, torch.full((heads, prefix_blocks), layout.n_blocks, dtype=torch.int32, device=device))
        return index, count, self.index_cache[key]


_TILED_BYTES = 128 << 20  # bound of each tiled q/k/v/out copy: heads are processed by chunks, whatever the sequence length


class VSAAttention:
    def __init__(self, policy, layer):
        self.policy = policy
        self.layer = layer

    def __call__(self, qkv_list, group, gate_list):
        """qkv_list: disposable [1, S, H', D] query/key/value of the heads group.q (normalized, rotated); gate_list: the disposable
        [1, S, H', D] gate projection of the same heads."""
        layout = self.policy.layout
        query, key, value = (tensor.squeeze(0) for tensor in qkv_list)
        qkv_list.clear()
        gate = gate_list[0].squeeze(0)
        gate_list.clear()
        seq_len, heads, dim = query.shape
        device = query.device
        result = query.new_empty(seq_len, heads, dim)
        chunk = max(1, min(heads, _TILED_BYTES // (layout.num_slots * dim * query.element_size())))
        for start in range(0, heads, chunk):
            self._heads(query, key, value, gate, start, self.policy.head_range(start, min(heads, start + chunk), device), result)
        query = key = value = gate = None
        return result.unsqueeze(0)

    def _heads(self, query, key, value, gate, first, local, result):
        layout = self.policy.layout
        heads, dim, device = local.numel(), query.shape[-1], query.device
        n_prefix, n_tiles, tile = layout.n_prefix_tiles, layout.n_tiles, layout.tile
        int8_qk, cuda_kind = self.policy.int8_qk, self.policy.cuda_kind
        if cuda_kind:  # WGPKernels kernel: softmax scale applied inside, keys smoothed by their mean
            q_int8, q_scale, q_pooled = fused.gather_int8_pool(query, layout, local)
            k_mean = key.narrow(1, first, heads).mean(dim=0, dtype=torch.float32)  # the chunk's heads are consecutive: a view
            k_int8, k_scale, k_pooled = fused.gather_int8_pool(key, layout, local, mean=k_mean)
            k_mean = None
        elif int8_qk:  # one pass per tensor: tiled INT8 / FP16 copies, their scales and the fp32 tile means
            q_int8, q_scale, q_pooled = fused.gather_int8_pool(query, layout, local, sparse_int8.query_pre_scale(dim))
            k_int8, k_scale, k_pooled = fused.gather_int8_pool(key, layout, local)
        else:
            q = preprocess.gather(query, layout, local)
            k = preprocess.gather(key, layout, local)
            inverse_count = 1.0 / layout.valid_count.clamp_min(1).to(torch.float32)[:, None, None]
            # fp32 masked means over the tiles' real rows (padding slots are zero): [H', n_tiles, D]
            q_pooled = (q.view(n_tiles, tile, heads, dim).sum(1, dtype=torch.float32) * inverse_count).permute(1, 0, 2)
            k_pooled = (k.view(n_tiles, tile, heads, dim).sum(1, dtype=torch.float32) * inverse_count).permute(1, 0, 2)
        scores = torch.baddbmm(q_pooled.new_empty(heads, n_tiles, n_tiles), q_pooled, k_pooled.transpose(1, 2), beta=0, alpha=dim ** -0.5)
        q_pooled = k_pooled = None
        # every video query tile keeps its own top-k of each region: [H', n_video_tiles, keep_total] kept tile ids
        video_scores = scores[:, n_prefix:]
        top = torch.cat([torch.topk(video_scores[:, :, region.start:region.stop], region.keep, dim=-1).indices.add_(region.start)
                         if region.keep < region.stop - region.start else
                         torch.arange(region.start, region.stop, device=device).expand(heads, n_tiles - n_prefix, -1) for region in layout.regions], dim=-1)
        video_scores = None
        out = torch.empty((layout.num_slots, heads, dim), device=device, dtype=query.dtype)
        if cuda_kind:
            lut, count = self.policy.index_lut(heads, device)
            fused.index_lut(top, lut, layout)
            if cuda_kind == "fp8":
                low, high = torch.aminmax(value.narrow(1, first, heads), dim=0)
                v8, v_pooled, v_scale = fused.gather_fp8_pool_t(value, layout, local, low, high, sparse_cuda.V_FP8_RANGE)
                low = high = None
                sparse_cuda.attend(cuda_kind, q_int8, q_scale, k_int8, k_scale, v8, v_scale, out, lut, count, layout.block_valid, dim)
                v8 = v_scale = None
            else:
                v16, v_pooled = fused.gather_fp16_pool(value, layout, local)
                sparse_cuda.attend(cuda_kind, q_int8, q_scale, k_int8, k_scale, v16, None, out, lut, count, layout.block_valid, dim)
                v16 = None
            q_int8 = k_int8 = top = None
        else:
            index, count, (prefix_index, prefix_count) = self.policy.block_index(heads, top)
            top = None
            prefix_blocks = n_prefix * layout.blocks_per_tile
            if int8_qk:  # SageAttention-style INT8 QK / FP16 PV, as the default dense attention
                v16, v_pooled = fused.gather_fp16_pool(value, layout, local)
                quantized = sparse_int8.Quantized(q_int8, q_scale, k_int8, k_scale, v16)
                q_int8 = k_int8 = v16 = None
                sparse_int8.attend(quantized, out, index, count, layout.block_valid, prefix_blocks)
                if prefix_blocks:
                    sparse_int8.attend(quantized, out, prefix_index, prefix_count, layout.block_valid, 0)
                quantized = None
            else:
                v = preprocess.gather(value, layout, local)
                kernel.attend(q, k, v, out, index, count, layout.block_valid, prefix_blocks)
                if prefix_blocks:
                    kernel.attend(q, k, v, out, prefix_index, prefix_count, layout.block_valid, 0)
                q = k = None
                v_pooled = (v.view(n_tiles, tile, heads, dim).sum(1, dtype=torch.float32) * inverse_count).permute(1, 0, 2)
                v = None
            index = count = None
        # coarse branch: dense attention over the pooled tiles [H', n_tiles, D], added to each tile's rows scaled by the trained gate
        # while the fine output is scattered back to the packed rows
        coarse = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)
        scores = v_pooled = None
        fused.scatter_gated(result, out, coarse, gate, layout, local)
        coarse = out = None


def read_fasth3_inference(filename):
    """The FastH3 inference contract stored in the prepared checkpoint's metadata (schedule, shifts, VSA sparsity, tile size and
    reference keep rate)."""
    import json
    from safetensors import safe_open
    with safe_open(filename, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
    if "fasth3_inference" not in metadata:
        raise ValueError(f"Not a WanGP FastH3 checkpoint (no fasth3_inference metadata): {filename}")
    return json.loads(metadata["fasth3_inference"])
