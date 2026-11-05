# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke, evoke/modules/geometric_state/da3_cloud.py: DA3FrameBank,
# _render_backward_multisrc_zbuf, render_cloud_batched, solve_chunk0_disparity_scale, build_single_source_warp_mono;
# evoke/pipelines/pipeline_evoke.py: _geo_da3_ingest; evoke/dataset/online_materialize.py: _geo_resize_visibility_to_latent), Copyright
# 2026 The Evoke Team, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The world memory of EVOKE: after every chunk, 12 of its 36 frames go through ViGeo (one stream for the world) and are stored as depth
maps under their commanded camera poses, nothing fused. The view of the next chunk is rendered from them: for each of its 33 target
poses the 8 stored frames whose sampled points fall most inside the target frustum, depth-scaled against the 16 most recent frames when
older, forward-splatted one pixel per point into a z-buffer and fused by nearest depth. The stored frames live in RAM (depth FP32, RGB
uint8); only the 2,000 co-visibility samples per frame stay on the GPU, and the selected frames are uploaded per render."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

INGEST_FRAMES = 12  # frames of a chunk written to the memory
COVIS_SAMPLES, TOP_K, NEARBY, DEPTH_BAND = 2000, 8, 16, 0.02
FG_COVIS, FG_FACTOR, SCALE_MIN_OVERLAP = 0.3, 1.5, 200
FAR = 1e6
CHUNK0_DISPARITY_PX = 135.0  # p90 parallax of the first chunk's single image warp, in render pixels (0: off)


def scale_intrinsics(K, src_hw, dst_hw):
    """K of src_hw pixels for an image resized to dst_hw (per-axis scaling, as the source)."""
    K = K.clone()
    sy, sx = dst_hw[0] / src_hw[0], dst_hw[1] / src_hw[1]
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def unproject(depth, K, c2w):
    """World points (h, w, 3) of a depth map (h, w) with intrinsics K and camera-to-world pose c2w (OpenCV axes)."""
    h, w = depth.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=depth.device, dtype=torch.float32), torch.arange(w, device=depth.device, dtype=torch.float32), indexing="ij")
    z = depth.float()
    cam = torch.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], dim=-1)
    return cam @ c2w[:3, :3].float().T + c2w[:3, 3].float()


def visibility_to_latent(visibility, latent_frames, latent_hw, temporal=4):
    """The pixel visibility (1, 1, frames, H, W) of a memory view as the latent mask (1, 1, latent_frames, h, w): frames 0, 4, 8, ...,
    averaged over 16 x 16 pixel patches (2 x 2 latents, one token), kept above half coverage."""
    ids = (torch.arange(latent_frames, device=visibility.device) * temporal).clamp(max=visibility.shape[2] - 1)
    sampled = visibility.index_select(2, ids)
    covered = (F.adaptive_avg_pool3d(sampled, (latent_frames, latent_hw[0] // 2, latent_hw[1] // 2)) > 0.5).to(sampled.dtype)
    return F.interpolate(covered, size=(latent_frames, *latent_hw), mode="nearest")


@dataclass
class StoredFrame:
    depth: torch.Tensor  # (h, w) FP32, CPU, pose units
    rgb: torch.Tensor  # (3, h, w) uint8, CPU: the frame ViGeo saw
    K: torch.Tensor  # (3, 3) at (h, w)
    c2w: torch.Tensor  # (4, 4), the commanded pose
    samples: torch.Tensor  # (COVIS_SAMPLES, 3) world points on the device (FAR rows: no valid point)


class WorldBank:
    def __init__(self, device, retention_frames=None):
        self.device, self.retention_frames, self.frames = device, retention_frames, {}

    def __len__(self):
        return len(self.frames)

    def ingest(self, stream, frames, poses, start, K_pix):
        """Writes a decoded chunk (frames (N, 3, H, W) in [-1, 1] on the device, its poses (N, 4, 4), frame ids start..start+N-1, K_pix
        the intrinsics at (H, W)): 12 evenly spaced frames through the depth stream, then retention."""
        count = frames.shape[0]
        local = torch.unique(torch.linspace(0, count - 1, INGEST_FRAMES).round().long().clamp(0, count - 1)).tolist()
        depth, seen, size = stream.window((frames[local].float() * 0.5 + 0.5).clamp(0, 1))
        K = scale_intrinsics(K_pix.float().cpu(), frames.shape[-2:], size)
        for index, (position, frame_depth, frame_rgb) in enumerate(zip(local, depth, seen)):
            gid = start + position
            pose = poses[position].float().cpu()
            world = unproject(frame_depth, K.to(frame_depth.device), pose.to(frame_depth.device)).reshape(-1, 3)
            valid = world[(frame_depth.reshape(-1) > 1e-4) & torch.isfinite(world).all(-1)]
            generator = torch.Generator(device=valid.device).manual_seed(gid)
            samples = (valid[torch.randint(0, valid.shape[0], (COVIS_SAMPLES,), device=valid.device, generator=generator)] if valid.shape[0]
                       else torch.full((COVIS_SAMPLES, 3), FAR, device=valid.device))
            self.frames[gid] = StoredFrame(frame_depth.cpu(), (frame_rgb * 255).round().to(torch.uint8).cpu(), K, pose, samples.to(self.device))
        if self.retention_frames:
            oldest = max(self.frames) - self.retention_frames + 1
            for gid in [gid for gid in self.frames if gid < oldest]:
                del self.frames[gid]

    def state(self):
        """The memory as CPU tensors (world files)."""
        ids = sorted(self.frames)
        if not ids:
            return {}
        return {"bank_ids": torch.tensor(ids, dtype=torch.int64), "bank_depth": torch.stack([self.frames[g].depth for g in ids]).to(torch.float16),
                "bank_rgb": torch.stack([self.frames[g].rgb for g in ids]), "bank_K": torch.stack([self.frames[g].K for g in ids]),
                "bank_c2w": torch.stack([self.frames[g].c2w for g in ids]), "bank_samples": torch.stack([self.frames[g].samples.cpu() for g in ids])}

    def load_state(self, tensors):
        for index, gid in enumerate(tensors["bank_ids"].tolist()):
            self.frames[gid] = StoredFrame(tensors["bank_depth"][index].float(), tensors["bank_rgb"][index], tensors["bank_K"][index].float(),
                                           tensors["bank_c2w"][index].float(), tensors["bank_samples"][index].float().to(self.device))

    # ---- rendering

    def _scale(self, frame, depth, reference):
        """(scale, overlap) of an old frame's depth against the reference points (the nearby frames' samples): the median ratio of the
        nearest reference depth to the frame's depth where both exist, 1 below 200 pixels of overlap or outside (0.2, 5)."""
        h, w = depth.shape
        w2c = torch.linalg.inv(frame.c2w.to(self.device))
        K = frame.K.to(self.device)
        cam = (w2c[:3, :3] @ reference.T).T + w2c[:3, 3]
        z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * K[0, 0] + K[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * K[1, 1] + K[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < SCALE_MIN_OVERLAP:
            return 1.0, int(ok.sum())
        nearest = torch.full((h * w,), float("inf"), device=self.device)
        nearest.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        flat = depth.reshape(-1)
        both = (nearest < float("inf")) & (flat > 1e-4)
        overlap = int(both.sum())
        if overlap < SCALE_MIN_OVERLAP:
            return 1.0, overlap
        scale = float((nearest[both] / flat[both]).median())
        return (scale if 0.2 < scale < 5.0 else 1.0), overlap

    def _splat(self, frame, depth, rgb, target, K, height, width, scale):
        """One stored frame forward-splatted into a target camera: (colours (3, H, W) in [0, 1], z-buffer (H * W,), covered (H, W)), the
        nearest point per pixel through a packed (depth x 1000) << 24 | source pixel int64 minimum, its colour sampled at that pixel."""
        h, w = depth.shape
        z = depth * scale if scale != 1.0 else depth
        ys, xs = torch.meshgrid(torch.arange(h, device=self.device, dtype=torch.float32), torch.arange(w, device=self.device, dtype=torch.float32), indexing="ij")
        frame_K, c2w = frame.K.to(self.device), frame.c2w.to(self.device)
        cam = torch.stack([(xs - frame_K[0, 2]) / frame_K[0, 0] * z, (ys - frame_K[1, 2]) / frame_K[1, 1] * z, z, torch.ones_like(z)], -1).reshape(-1, 4)
        world = (c2w @ cam.T).T[:, :3]
        w2c = torch.linalg.inv(target)
        ct = (w2c[:3, :3] @ world.T).T + w2c[:3, 3]
        zt = ct[:, 2]
        xt = torch.round(ct[:, 0] / zt.clamp(min=1e-6) * K[0, 0] + K[0, 2]).long()
        yt = torch.round(ct[:, 1] / zt.clamp(min=1e-6) * K[1, 1] + K[1, 2]).long()
        ok = (z.reshape(-1) > 1e-4) & (zt > 1e-4) & (xt >= 0) & (xt < width) & (yt >= 0) & (yt < height)
        zbuf = torch.full((height * width,), float("inf"), device=self.device)
        if not bool(ok.any()):
            return None, zbuf
        packed = ((zt[ok].clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1) << 24) | torch.arange(h * w, device=self.device)[ok]
        best = torch.full((height * width,), 1 << 62, dtype=torch.long, device=self.device)
        best.scatter_reduce_(0, (yt * width + xt)[ok], packed, reduce="amin", include_self=True)
        valid = best < (1 << 62)
        owner = (best & ((1 << 24) - 1)).clamp(max=h * w - 1)
        zbuf[valid] = (best[valid] >> 24).float() / 1000.0
        grid = torch.stack([((owner % w).float() / max(w - 1, 1) * 2 - 1).reshape(height, width), ((owner // w).float() / max(h - 1, 1) * 2 - 1).reshape(height, width)], -1)[None]
        sampled = F.grid_sample(rgb[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        return torch.where(valid.reshape(height, width)[None], sampled, torch.zeros_like(sampled)), zbuf

    @torch.no_grad()
    def render(self, targets, K_pix, height, width):
        """The memory view of a chunk: (warp (1, 3, F, H, W) in [-1, 1], visibility (1, 1, F, H, W)) for target poses (F, 4, 4)."""
        frames = len(targets)
        ids = sorted(self.frames)
        if not ids:
            return torch.full((1, 3, frames, height, width), -1.0, device=self.device), torch.zeros((1, 1, frames, height, width), device=self.device)
        K = K_pix.to(self.device).float()
        targets = targets.to(self.device).float()
        samples = torch.stack([self.frames[g].samples for g in ids])
        nearby = ids[-NEARBY:]
        nearby_centres = torch.stack([self.frames[g].c2w[:3, 3] for g in nearby]).to(self.device)
        centres = torch.stack([self.frames[g].c2w[:3, 3] for g in ids]).to(self.device)
        orders, coverages = [], []
        for target in targets:  # co-visibility: the share of each frame's samples inside the target frustum (no occlusion test)
            w2c = torch.linalg.inv(target)
            cam = torch.einsum("cmj,kj->cmk", samples - target[:3, 3], w2c[:3, :3])
            z = cam[..., 2]
            px, py = cam[..., 0] / z.clamp(min=1e-6) * K[0, 0] + K[0, 2], cam[..., 1] / z.clamp(min=1e-6) * K[1, 1] + K[1, 2]
            coverage = ((z > 1e-4) & (px >= 0) & (px < width) & (py >= 0) & (py < height) & (samples[..., 0] < FAR * 0.5)).float().mean(1)
            orders.append(torch.topk(coverage, k=min(TOP_K, len(ids))).indices.tolist())
            coverages.append(coverage.tolist())
        selected = sorted({ids[i] for order in orders for i in order})
        reference = samples[[ids.index(g) for g in nearby]].reshape(-1, 3)
        reference = reference[reference[:, 0] < FAR * 0.5]
        uploaded = {g: (self.frames[g].depth.to(self.device), self.frames[g].rgb.to(self.device).float() / 255) for g in selected}
        scales = {g: (1.0, 0) if g in nearby else self._scale(self.frames[g], uploaded[g][0], reference) for g in selected}
        aligned = {g: g not in nearby and scales[g][1] >= SCALE_MIN_OVERLAP and 0.8 <= scales[g][0] <= 1.25 for g in selected}
        warps, visibility = [], []
        for target, order, coverage in zip(targets, orders, coverages):
            nearest_nearby = float((nearby_centres - target[:3, 3]).norm(dim=1).min())
            candidates = [ids[i] for i in order if coverage[i] >= FG_COVIS or float((centres[i] - target[:3, 3]).norm()) <= FG_FACTOR * nearest_nearby or aligned[ids[i]]]
            fused = torch.zeros(3, height, width, device=self.device)
            fused_depth = torch.full((height * width,), float("inf"), device=self.device)
            covered = torch.zeros(height, width, dtype=torch.bool, device=self.device)
            for g in candidates:
                colour, zbuf = self._splat(self.frames[g], *uploaded[g], target, K, height, width, scales[g][0])
                if colour is None:
                    continue
                update = torch.isfinite(zbuf) & (zbuf < fused_depth - DEPTH_BAND)
                if not bool(update.any()):
                    update = torch.isfinite(zbuf) & (zbuf < fused_depth)
                    if not bool(update.any()):
                        continue
                mask = update.reshape(height, width)
                fused = torch.where(mask[None], colour, fused)
                fused_depth = torch.where(update, zbuf, fused_depth)
                covered |= mask
            warps.append(fused.clamp(0, 1) * 2 - 1)
            visibility.append(covered.float())
        return torch.stack(warps, 1)[None], torch.stack(visibility)[None, None]


def render_cloud(xyz, rgb, targets, K, height, width, radius=1):
    """A point cloud (P, 3) with colours (P, 3) in [0, 1] splatted ((2 radius + 1)^2 pixels per point, nearest wins) into target cameras:
    (warp (1, 3, F, H, W) in [-1, 1], black where empty; visibility (1, 1, F, H, W))."""
    device, points, views, pixels = xyz.device, xyz.shape[0], targets.shape[0], height * width
    w2c_r, t = targets[:, :3, :3], targets[:, :3, 3]
    cam = torch.einsum("vij,vpj->vpi", w2c_r.transpose(1, 2), xyz[None] - t[:, None])
    z = cam[..., 2]
    px = torch.round((cam[..., 0] / z) * K[0, 0] + K[0, 2]).long()
    py = torch.round((cam[..., 1] / z) * K[1, 1] + K[1, 2]).long()
    front = z > 1e-4
    zq = (z.clamp(0, 1e6) * 1000.0).long().clamp(0, (1 << 38) - 1)
    index = torch.arange(points, device=device)[None].expand(views, points)
    view = torch.arange(views, device=device)[:, None].expand(views, points)
    best = torch.full((views * pixels,), 1 << 62, dtype=torch.long, device=device)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            qx, qy = px + dx, py + dy
            ok = front & (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
            best.scatter_reduce_(0, (view * pixels + qy * width + qx)[ok], (zq[ok] << 24) | index[ok], reduce="amin", include_self=True)
    valid = best < (1 << 62)
    winner = (best & ((1 << 24) - 1)).clamp(max=points - 1)
    image = torch.zeros(views * pixels, 3, device=device)
    image[valid] = rgb[winner[valid]]
    image = image.reshape(views, height, width, 3).clamp(0, 1)
    return (image.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0).contiguous(), valid.reshape(views, height, width).float()[None, None].contiguous()


def disparity_scale(depth, K_depth, ref_c2w, targets, K_out, target_px, quantile=0.9, limits=(1.0 / 64.0, 64.0)):
    """Factor k on a single image's depth so that the quantile pixel parallax over the target poses is target_px (the first chunk's
    amplitude control); 1 without commanded translation."""
    h, w = depth.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=depth.device, dtype=torch.float32), torch.arange(w, device=depth.device, dtype=torch.float32), indexing="ij")
    m_x, m_y = (xs - float(K_depth[0, 2])) / float(K_depth[0, 0]), (ys - float(K_depth[1, 2])) / float(K_depth[1, 1])
    fx, fy = float(K_out[0, 0]), float(K_out[1, 1])
    R0, t0 = ref_c2w[:3, :3].float(), ref_c2w[:3, 3].float()
    valid = torch.isfinite(depth) & (depth > 0)
    if not bool(valid.any()):
        return 1.0
    safe = torch.where(valid, depth, torch.ones_like(depth))
    best = 0.0
    for target in targets:
        dx, dy, dz = (float(v) for v in R0.transpose(0, 1) @ (t0 - target[:3, 3].float()))
        spread = torch.sqrt((fx * (dx - m_x * dz)) ** 2 + (fy * (dy - m_y * dz)) ** 2)
        k = torch.where(valid, ((spread / target_px) - dz) / safe, torch.zeros_like(safe)).clamp_min(0.0)
        best = max(best, float(torch.quantile(k.reshape(-1), quantile)))
    if not math.isfinite(best) or best <= 0:
        return 1.0
    return min(max(best, limits[0]), limits[1])


@torch.no_grad()
def single_image_view(stream, image, ref_pose, targets, K_pix, height, width, disparity_px=CHUNK0_DISPARITY_PX):
    """The first chunk's memory view of an image-started world: the start image's own depth (its median at 5 pose units, rescaled to
    disparity_px of parallax over the targets when the camera moves) splatted into the target poses, 3 x 3 pixels per point."""
    depth, seen, size = stream.single(((image.float() * 0.5) + 0.5).clamp(0, 1))
    K_depth = scale_intrinsics(K_pix.float(), (height, width), size).to(depth.device)
    ref_pose, targets = ref_pose.to(depth.device).float(), targets.to(depth.device).float()
    if disparity_px > 0:
        depth = depth * disparity_scale(depth, K_depth, ref_pose, targets, K_pix.to(depth.device).float(), disparity_px)
    xyz = unproject(depth, K_depth, ref_pose).reshape(-1, 3)
    return render_cloud(xyz, seen.permute(1, 2, 0).reshape(-1, 3), targets, K_pix.to(depth.device).float(), height, width)
