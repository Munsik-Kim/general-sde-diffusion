# src/models/score_mlp.py
import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    """
    시간 t를 고차원 벡터로 매핑하여 모델이 '시간'을 훨씬 예민하게 인지하도록 함.
    Transformer의 Positional Encoding과 유사한 역할.
    """
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # 학습되지 않는(Random Fixed) 가중치 사용
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        # t: [Batch, 1] -> [Batch, Dim]
        t_proj = t * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class ScoreMLP(nn.Module):
    def __init__(self, hidden_dim=256): # hidden_dim을 좀 더 키움
        super().__init__()
        
        # 1. 강력해진 Time Embedding
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU() # Swish
        )
        
        # 2. 메인 네트워크 (ResNet 구조처럼 Skip Connection을 추가하면 좋지만, 일단 깊게)
        self.input_layer = nn.Linear(2 + hidden_dim, hidden_dim)
        
        self.layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t):
        # x: [B, 2], t: [B, 1]
        if t.dim() == 1: t = t.unsqueeze(1)
        
        # Time Embedding
        t_emb = self.time_embed(t)
        
        # x와 t 정보를 결합
        xt = torch.cat([x, t_emb], dim=1)
        
        out = self.input_layer(xt)
        out = self.layers(out)
        
        # (선택 사항) 마지막에 아주 작은 값으로 나누어 초기 학습 안정화
        return out