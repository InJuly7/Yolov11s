import torch
from Backbone import Backbone
from Head import Head
from Detect import Detect


class YOLOV11S(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.Backbone = Backbone()
        self.head = Head()
        self.Detect = Detect(nc=80, ch=(128, 256, 512))
        self.Detect.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x):
        p3, p4, p5 = self.Backbone(x)
        p3, p4, p5 = self.head([p3, p4, p5])
        return self.Detect([p3, p4, p5])



def test_yolov11s(seed=77, model_path=None):
    model = YOLOV11S().to(dtype=torch.float32, device="cuda")
    model.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(1, 3, 640, 640).to(dtype=torch.float32, device="cuda")
    with torch.no_grad():
        x, _ = model(x)
    # print(f"output x shape: {x.shape}")
    return x


if __name__ == "__main__":
    test_yolov11s(seed=77)
