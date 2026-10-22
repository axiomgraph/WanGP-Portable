# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Tile permutation of the packed H3 sequence for VSA: [prefix tiles][region tiles]..., `tile` slots per tile.

Dense segments (text, keyframes, image references, audio rows) are chunked into tiles one segment at a time, in sequence order, so no
tile mixes two segments. Every video region (a reference video, then the generated video last) has its (T, H, W) token grid cut into
`tile_shape` boxes, enumerated t, h, w outer to inner, the rows inside a box in t, h, w order; a box at the grid's edge holds fewer rows.
Real rows are compacted to the front of every tile and the remaining slots are padding (-1), so a tile's real rows are the prefix of
length `valid_count[tile]`. The kernels work on blocks of BLOCK rows (two per 128-token tile): `block_valid` is the real rows of every
block, a tile's blocks are consecutive.

Only the membership of a tile matters to the attention (which tokens are pooled and selected together), not the numbering, so this
layout gives the same results as FastVideo's VSA-H3 tiling for the same segments.
"""
import dataclasses
import math

import torch

BLOCK = 64  # rows per kernel block (q/k scale block, key block of the sparse kernels)
TILE_SHAPES = {64: (4, 4, 4), 128: (4, 4, 8)}


@dataclasses.dataclass
class Region:
    start: int  # first tile of the region (tile numbering over the whole layout)
    stop: int  # one past its last tile
    keep: int  # tiles of this region every video query tile keeps


@dataclasses.dataclass
class TileLayout:
    perm: torch.Tensor  # [n_tiles * tile] int64 packed row of every slot, -1 on padding
    valid_count: torch.Tensor  # [n_tiles] int32 real rows per tile (a prefix of the tile)
    block_valid: torch.Tensor  # [n_blocks] int32 real rows per kernel block
    tile: int
    n_prefix_tiles: int
    regions: tuple  # Region per video region, the generated video last; their tiles follow the prefix tiles contiguously
    seq_len: int

    @property
    def n_video_tiles(self):
        return sum(region.stop - region.start for region in self.regions)

    @property
    def n_tiles(self):
        return self.n_prefix_tiles + self.n_video_tiles

    @property
    def blocks_per_tile(self):
        return self.tile // BLOCK

    @property
    def n_blocks(self):
        return self.n_tiles * self.blocks_per_tile

    @property
    def num_slots(self):
        return self.n_tiles * self.tile

    @property
    def keep_total(self):
        return sum(region.keep for region in self.regions)


def keep_count(sparsity, n_tiles):
    """Tiles kept per query at a sparsity (FastVideo's compute_topk)."""
    return max(1, min(math.ceil((1.0 - sparsity) * n_tiles), n_tiles))


def _grid_tiles(start, grid, tile, tile_shape):
    """[n_boxes, tile] rows of a (T, H, W) grid starting at packed row `start`, boxed by tile_shape, real rows first."""
    t, h, w = grid
    tt, th, tw = tile_shape
    tp, hp, wp = -(-t // tt) * tt, -(-h // th) * th, -(-w // tw) * tw
    boxes = torch.full((tp, hp, wp), -1, dtype=torch.long)
    boxes[:t, :h, :w] = start + torch.arange(t * h * w).view(t, h, w)
    tiles = boxes.view(tp // tt, tt, hp // th, th, wp // tw, tw).permute(0, 2, 4, 1, 3, 5).reshape(-1, tile)
    order = torch.argsort((tiles < 0).to(torch.int8), dim=1, stable=True)
    return torch.gather(tiles, 1, order)


def build_tile_layout(seq_len, dense_spans, video_regions, tile, sparsity, ref_keep_rate, device):
    """Layout of a packed sequence: dense_spans are (start, stop) row ranges chunked one at a time; video_regions are
    (start, grid) of every video region in sequence order, the generated video last, each with its own top-k budget: `sparsity`
    for the generated video, 1 - ref_keep_rate for reference videos."""
    tile_shape = TILE_SHAPES[tile]
    prefix = []
    for start, stop in dense_spans:
        rows = torch.arange(start, stop)
        n = -(-(stop - start) // tile)
        chunk = torch.full((n * tile,), -1, dtype=torch.long)
        chunk[:stop - start] = rows
        prefix.append(chunk.view(n, tile))
    n_prefix = sum(part.shape[0] for part in prefix)
    parts, regions, cursor = prefix, [], n_prefix
    for index, (start, grid) in enumerate(video_regions):
        if start + math.prod(grid) > seq_len:
            raise ValueError(f"VSA video region at row {start} with grid {tuple(grid)} exceeds the {seq_len}-row sequence")
        tiles = _grid_tiles(start, grid, tile, tile_shape)
        last = index == len(video_regions) - 1
        regions.append(Region(cursor, cursor + tiles.shape[0], keep_count(sparsity if last else 1.0 - ref_keep_rate, tiles.shape[0])))
        parts.append(tiles)
        cursor += tiles.shape[0]
    perm = torch.cat(parts).reshape(-1)
    valid_count = (perm.view(-1, tile) >= 0).sum(1).to(torch.int32)
    block_valid = (perm.view(-1, BLOCK) >= 0).sum(1).to(torch.int32)
    if int((perm >= 0).sum()) != seq_len:
        raise ValueError("VSA layout: the dense spans and video regions must cover every packed row exactly once")
    return TileLayout(perm=perm.to(device), valid_count=valid_count.to(device), block_valid=block_valid.to(device), tile=tile,
                      n_prefix_tiles=n_prefix, regions=tuple(regions), seq_len=seq_len)
