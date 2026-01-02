import torch
from torch import nn
from Conv import Conv


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        # 残差结构
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


"""
Bottleneck input shape: torch.Size([1, 32, 160, 160])
self.add: True
self.cv1: Conv(
  (conv): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
  (act): SiLU(inplace=True)
)
self.cv2: Conv(
  (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
  (act): SiLU(inplace=True)
)
Bottleneck output shape: torch.Size([1, 32, 160, 160])
"""


def test_bottleneck():
    model = Bottleneck(c1=32, c2=32, shortcut=True, g=1, k=(3, 3), e=0.5).to(dtype=torch.float32, device="cuda")
    model.eval()
    x = torch.randn(1, 32, 160, 160).to(dtype=torch.float32, device="cuda")
    x = model(x)
    print(f"Bottleneck output shape: {x.shape}")


if __name__ == "__main__":
    test_bottleneck()
