"""MiniMax H3 tensor-layout transforms for generic ConvRot conversion and loading."""

from functools import partial


def group_qkv_rows(tensor, heads, head_dim):
    tail = tuple(tensor.shape[1:])
    return tensor.reshape(heads, 3, head_dim, *tail).permute(1, 0, 2, *range(3, 3 + len(tail))).reshape_as(tensor).contiguous()


def get_convrot_layout(model):
    """Return H3's optional source-to-ConvRot tensor layout mapping."""
    return {
        f"{name}.qkv_proj.weight": partial(group_qkv_rows, heads=module.heads, head_dim=module.head_dim)
        for name, module in model.named_modules() if hasattr(module, "heads") and hasattr(module, "head_dim")
    }
