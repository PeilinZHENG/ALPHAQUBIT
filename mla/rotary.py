import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        assert dim % 2 == 0, "旋转维度必须是偶数"
        self.dim = dim
        self.theta = theta
        
        # 初始化时先不构建缓存，等到第一次forward时按需构建
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)
        self.max_seq_len = max_seq_len
    
    def _build_cache(self, seq_len, device):
        # 动态构建缓存
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        t = torch.arange(seq_len, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self.max_seq_len = seq_len
    
    def forward(self, x, seq_len=None):
        """输入: [batch, seq, dim]
           输出: [batch, seq, dim]
        """
        seq_len = seq_len if seq_len else x.size(1)
        device = x.device
        
        # 按需构建缓存
        if self.cos_cache is None or seq_len > self.max_seq_len:
            self._build_cache(seq_len, device)
        
        # 确保缓存维度匹配
        cos = self.cos_cache[:seq_len, :].to(device)
        sin = self.sin_cache[:seq_len, :].to(device)
        
        # 只旋转前dim个维度
        x_rot = x[..., :self.dim]
        x_pass = x[..., self.dim:]
        
        # 应用旋转
        x_rot = x_rot * cos.unsqueeze(0) + self._rotate_half(x_rot) * sin.unsqueeze(0)
        
        return torch.cat([x_rot, x_pass], dim=-1)
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

if __name__ == "__main__":
    # 测试不同序列长度
    rotary = RotaryEmbedding(dim=32)
    
    # 测试短序列
    x_short = torch.randn(2, 50, 64)  # [batch, seq, dim]
    out_short = rotary(x_short)
    print(f"短序列输入: {x_short.shape}, 输出: {out_short.shape}")
    
    # 测试长序列（超过初始max_seq_len）
    x_long = torch.randn(2, 3000, 64)
    out_long = rotary(x_long)
    print(f"长序列输入: {x_long.shape}, 输出: {out_long.shape}")
    
    # 验证形状一致性
    assert out_short.shape == x_short.shape
    assert out_long.shape == x_long.shape