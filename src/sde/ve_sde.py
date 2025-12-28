import torch
import numpy as np
from .base_sde import SDE

class VESDE(SDE):
    """
    Variance Exploding SDE (a.k.a Score Matching with Langevin Dynamics, NCSN)
    dx = sqrt(d(sigma^2)/dt) dw
    """
    def __init__(self, sigma_min=0.01, sigma_max=50.0, T=1.0):
        super().__init__(T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        # sigma(t) = sigma_min * (sigma_max / sigma_min)^t
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x, t):
        # Drift = 0 (데이터를 건드리지 않음)
        # Diffusion = sqrt( d(sigma^2)/dt )
        sigma_t = self.sigma(t)
        
        # t가 텐서일 경우 차원 맞추기
        if isinstance(t, torch.Tensor):
            view_shape = [t.shape[0]] + [1] * (x.dim() - 1)
            sigma_t = sigma_t.view(*view_shape)
            
        drift = torch.zeros_like(x)
        
        # d(sigma^2)/dt = 2 * sigma * d(sigma)/dt
        #               = sigma^2 * 2 * ln(sigma_max/sigma_min)
        diffusion = sigma_t * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))))
        
        return drift, diffusion

    def marginal_prob(self, x0, t):
        # p_t(x) = N(x | x0, sigma(t)^2 I)
        std = self.sigma(t)
        
        if isinstance(t, torch.Tensor):
            view_shape = [t.shape[0]] + [1] * (x0.dim() - 1)
            std = std.view(*view_shape)
            
        mean = x0 # Drift가 0이므로 평균은 변하지 않음
        return mean, std

    def prior_sampling(self, shape):
        # Prior p_T(x) approx N(0, sigma_max^2 I)
        return torch.randn(*shape) * self.sigma_max