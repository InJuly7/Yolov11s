import torch
from torch import nn
from Conv import Conv


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)  # [B, h, H, W] h: (key_dim + key_dim + head_dim)*num_heads
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        # [B, num_heads, key_dim, N], [B, num_heads, key_dim, N], [B, num_heads, head_dim, N]
        attn = (q.transpose(-2, -1) @ k) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))  # [B, num_heads, head_dim, N] -> [B, C, H, W]
        x = self.proj(x)  # [B, C, H, W]
        return x


def test_attention():
    model = Attention(dim=512).to(dtype=torch.float32, device="cuda")
    model.eval()
    x = torch.randn(1, 512, 20, 20).to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        x = model(x)
        print(x.shape)


if __name__ == "__main__":
    test_attention()
