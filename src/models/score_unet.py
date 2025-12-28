import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Time Embedding: Scale=5.0 유지 (전체적인 형태 학습 유도)"""
    def __init__(self, embed_dim, scale=5.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, t):
        t_proj = t * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.dense(self.act(x))

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(1, 2).view(B, C, H, W)

class ScoreUNet(nn.Module):
    def __init__(self, channels=[64, 128, 256], embed_dim=256):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim, scale=5.0),
            Dense(embed_dim, embed_dim)
        )

        # Encoder
        self.conv1 = nn.Conv2d(1, channels[0], 3, 1, 1)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gn1 = nn.GroupNorm(32, channels[0])
        
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, 2, 1)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gn2 = nn.GroupNorm(32, channels[1])
        
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, 2, 1)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gn3 = nn.GroupNorm(32, channels[2])
        
        self.attn1 = SelfAttention(channels[2]) 
        
        # Decoder
        self.tconv2 = nn.ConvTranspose2d(channels[2], channels[1], 3, 2, 1, 1)
        self.dense4 = Dense(embed_dim, channels[1])
        self.gn4 = nn.GroupNorm(32, channels[1])
        
        self.attn2 = SelfAttention(channels[1])
        
        self.tconv1 = nn.ConvTranspose2d(channels[1]*2, channels[0], 3, 2, 1, 1)
        self.dense5 = Dense(embed_dim, channels[0])
        self.gn5 = nn.GroupNorm(32, channels[0])
        
        self.final_conv = nn.Conv2d(channels[0]*2, 1, 3, 1, 1)
        self.act = nn.SiLU()

        # ★★★ 핵심 추가: Zero Initialization ★★★
        # 마지막 레이어를 0으로 초기화하여 학습 초기 안정을 유도
        self.final_conv.weight.data.zero_()
        self.final_conv.bias.data.zero_()

    def forward(self, x, t):
        embed = self.time_embed(t)
        
        # Encoding
        h1 = self.conv1(x)
        h1 += self.dense1(embed)[..., None, None]
        h1 = self.act(self.gn1(h1))
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)[..., None, None]
        h2 = self.act(self.gn2(h2))
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)[..., None, None]
        h3 = self.act(self.gn3(h3))
        
        h3 = self.attn1(h3)
        
        # Decoding
        h2_up = self.tconv2(h3)
        h2_up += self.dense4(embed)[..., None, None]
        h2_up = self.act(self.gn4(h2_up))
        
        h2_up = self.attn2(h2_up)
        
        h1_up = self.tconv1(torch.cat([h2_up, h2], dim=1))
        h1_up += self.dense5(embed)[..., None, None]
        h1_up = self.act(self.gn5(h1_up))
        
        out = self.final_conv(torch.cat([h1_up, h1], dim=1))
        return out