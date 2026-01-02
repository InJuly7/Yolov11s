import torch
from torch import nn
from Conv import Conv
from Concat import Concat
from C3k2 import C3k2


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.h1 = C3k2(c1=768, c2=256, n=1, c3k=False)
        self.h2 = C3k2(c1=512, c2=128, n=1, c3k=False)
        self.h3 = Conv(c1=128, c2=128, k=3, s=2, p=1)
        self.h4 = C3k2(c1=384, c2=256, n=1, c3k=False)
        self.h5 = Conv(c1=256, c2=256, k=3, s=2, p=1)
        self.h6 = C3k2(c1=768, c2=512, n=1, c3k=True)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


def test_head():
    model = Head().to(dtype=torch.float32, device="cuda")
    model.eval()
    p3 = torch.randn(1, 256, 80, 80).to(device="cuda", dtype=torch.float32)
    p4 = torch.randn(1, 256, 40, 40).to(device="cuda", dtype=torch.float32)
    p5 = torch.randn(1, 512, 20, 20).to(device="cuda", dtype=torch.float32)
    x = [p3, p4, p5]
    with torch.no_grad():
        p3, p4, p5 = model(x)
    print("Output p3:", p3.shape)
    print("Output p4:", p4.shape)
    print("Output p5:", p5.shape)


if __name__ == "__main__":
    test_head()

"""
Output Log:
Output p3: torch.Size([1, 128, 80, 80])
Output p4: torch.Size([1, 256, 40, 40])
Output p5: torch.Size([1, 512, 20, 20])
"""
