import torch
from torch import nn
from Bottleneck import Bottleneck
from C2f import C2f
from C3k import C3k


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        # 构建一个 PyTorch 的模块列表（nn.ModuleList），里面放了 n 个子模块；
        # 每个子模块根据条件 c3k 来决定是用 C3k(...) 还是用 Bottleneck(...)
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))


"""
self.c: 32
Conv(
  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
  (act): SiLU(inplace=True)
)
Conv(
  (conv): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
  (act): SiLU(inplace=True)
)
ModuleList(
  (0): Bottleneck(
    (cv1): Conv(
      (conv): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (cv2): Conv(
      (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
  )
)
<class 'ultralytics.nn.modules.block.C3k2'>
input x shape: torch.Size([1, 64, 160, 160])
output x shape: torch.Size([1, 128, 160, 160])
"""


def test_c3k2():
    # c3k False
    model = C3k2(c1=64, c2=128, n=1, c3k=False, e=0.5, g=1, shortcut=True).to(dtype=torch.float32, device="cuda")
    x = torch.randn(1, 64, 160, 160).to(dtype=torch.float32, device="cuda")
    model.eval()
    with torch.no_grad():
        x = model(x)
        print(f"x shape {x.shape}")

    model = C3k2(c1=256, c2=256, n=1, c3k=True, e=0.25, g=1, shortcut=True).to(dtype=torch.float32, device="cuda")
    x = torch.randn(1, 256, 40, 40).to(dtype=torch.float32, device="cuda")
    model.eval()
    with torch.no_grad():
        x = model(x)
        print(f"x shape {x.shape}")


if __name__ == "__main__":
    test_c3k2()

"""
Output Log:
x shape torch.Size([1, 128, 160, 160])
x shape torch.Size([1, 256, 40, 40])
"""
