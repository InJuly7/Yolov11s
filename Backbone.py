import torch
from torch import nn

from Conv import Conv
from C3k2 import C3k2
from C2PSA import C2PSA
from SPPF import SPPF


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(c1=3, c2=32, k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(c1=32, c2=64, k=3, s=2, p=1))
        self.p2.append(C3k2(c1=64, c2=128, n=1, c3k=False, e=0.25))
        # p3/8
        self.p3.append(Conv(c1=128, c2=128, k=3, s=2, p=1))
        self.p3.append(C3k2(c1=128, c2=256, n=1, c3k=False, e=0.25))
        # p4/16
        self.p4.append(Conv(c1=256, c2=256, k=3, s=2, p=1))
        self.p4.append(C3k2(c1=256, c2=256, n=1, c3k=True, e=0.5))
        # p5/32
        self.p5.append(Conv(c1=256, c2=512, k=3, s=2, p=1))
        self.p5.append(C3k2(c1=512, c2=512, n=1, c3k=True, e=0.5))
        self.p5.append(SPPF(c1=512, c2=512, k=5))
        self.p5.append(C2PSA(c1=512, c2=512, n=1, e=0.5))

        self.p1 = nn.Sequential(*self.p1)
        self.p2 = nn.Sequential(*self.p2)
        self.p3 = nn.Sequential(*self.p3)
        self.p4 = nn.Sequential(*self.p4)
        self.p5 = nn.Sequential(*self.p5)

    def forward(self, x):
        # x = [1, 3, 640, 640]
        p1 = self.p1(x)  # [1, 32, 320, 320]
        p2 = self.p2(p1)  # [1, 128, 160, 160]
        p3 = self.p3(p2)  # [1, 256, 80, 80]
        p4 = self.p4(p3)  # [1, 256, 40, 40]
        p5 = self.p5(p4)  # [1, 512, 20, 20]
        return p3, p4, p5


def test_backbone():
    model = Backbone().to(dtype=torch.float32, device="cuda")
    model.eval()
    x = torch.randn(1, 3, 640, 640).to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        p3, p4, p5 = model(x)
    print("Backbone")
    print("Input:", x.shape)
    print("Output p3:", p3.shape)
    print("Output p4:", p4.shape)
    print("Output p5:", p5.shape)


if __name__ == "__main__":
    test_backbone()
