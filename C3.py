import torch
from torch import nn
from Conv import Conv
from Bottleneck import Bottleneck


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


def test_c3():
    model = C3(c1=256, c2=256, n=1, shortcut=True).to(dtype=torch.float32, device="cuda")
    model.eval()
    x = torch.randn(1, 256, 40, 40).to(dtype=torch.float32, device="cuda")
    with torch.no_grad():
        x = model(x)
        print(f"x shape {x.shape}")


if __name__ == "__main__":
    test_c3()

"""
Output Log:
x shape torch.Size([1, 256, 40, 40])
"""
