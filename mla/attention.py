import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这行
import math

class SafeAttention(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, Q, K, V, attn_mask=None):
        """添加attn_mask参数"""
        B = Q.size(0)
        
        Q = self._reshape(Q)
        K = self._reshape(K)
        V = self._reshape(V)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)
        
        attn = F.softmax(attn, dim=-1)
        return (attn @ V).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
    
    def _reshape(self, x):
        return x.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

if __name__ == "__main__":
    # 测试注意力
    attn = SafeAttention(num_heads=4, head_dim=32)
    Q = K = V = torch.randn(2, 224, 128)  # [batch, seq, d_model]
    print(f"输入形状: Q{K.shape}, K{K.shape}, V{V.shape}")
    out = attn(Q, K, V)
    print(f"输出形状: {out.shape}")
    assert out.shape == (2, 224, 128), "输出形状验证失败!"