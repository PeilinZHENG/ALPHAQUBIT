import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这行关键导入
from mla.projection import ProjectionSystem
from mla.attention import SafeAttention
from mla.rotary import RotaryEmbedding

class DeepSeekMLA(nn.Module):
    def __init__(self, d_model, num_heads, d_c=64, d_rotate=32):
        super().__init__()
        self.proj = ProjectionSystem(d_model, d_c, d_rotate)
        self.rotary = RotaryEmbedding(d_rotate)
        self.attention = SafeAttention(num_heads, d_model//num_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x):
        # 1. 投影
        proj_out = self.proj(x)
        C = proj_out['compressed']
        
        # 2. 旋转
        C_rot = self.rotary(C[..., :self.rotary.dim], seq_len=x.size(1))
        
        # 3. 合并
        if proj_out['residual_dim'] > 0:
            C = torch.cat([C_rot, C[..., self.rotary.dim:]], dim=-1)
        
        # 4. 解压和注意力
        H = self.proj.decompress(C).chunk(3, dim=-1)
        output = self.attention(*H)
        
        return self.out_proj(output)

if __name__ == "__main__":
    # 测试
    mla = DeepSeekMLA(d_model=128, num_heads=4)
    x = torch.randn(2, 224, 128)
    print(f"输入形状: {x.shape}")
    
    out = mla(x)
    print(f"输出形状: {out.shape}")
    assert out.shape == x.shape, f"形状不匹配! 输入{x.shape}, 输出{out.shape}"