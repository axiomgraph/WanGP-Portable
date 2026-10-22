import torch


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in place: the same values without a second copy, and one read and write of the activation
        return torch.ops.aten.gelu_(self.proj(x), approximate="tanh")
