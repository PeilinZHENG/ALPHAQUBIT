import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionSystem(nn.Module):
    """处理所有维度变换的安全投影系统"""
    def __init__(self, d_model, d_c, d_rotate):
        super().__init__()
        self.d_model = d_model
        self.d_c = d_c
        self.d_rotate = min(d_rotate, d_c//2)
        
        self.compress = nn.Linear(d_model, d_c)
        self.decompress = nn.Linear(d_c, d_model * 3)
        
        nn.init.orthogonal_(self.compress.weight, gain=0.02)
        nn.init.xavier_uniform_(self.decompress.weight)
    
    def forward(self, x):
        """输入: [batch, seq, d_model]"""
        # 安全压缩
        C = self.compress(x)
        
        # 返回压缩结果和解压所需形状
        return {
            'compressed': C,
            'rotate_dim': self.d_rotate,
            'residual_dim': self.d_c - self.d_rotate
        }

if __name__ == "__main__":
    # 测试投影系统
    proj = ProjectionSystem(d_model=128, d_c=64, d_rotate=32)
    x = torch.randn(2, 224, 128)
    out = proj(x)
    print(f"输入形状: {x.shape}")
    print(f"压缩输出形状: {out['compressed'].shape}")
    print(f"旋转维度: {out['rotate_dim']}, 残差维度: {out['residual_dim']}")