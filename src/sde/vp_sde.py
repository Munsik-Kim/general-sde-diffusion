import torch
from .base_sde import SDE

class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=5.0, T=1.0):
        super().__init__(T)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x, t):
        beta_t = self.beta(t)
        if isinstance(beta_t, torch.Tensor) and beta_t.dim() > 0:
            view_shape = [beta_t.shape[0]] + [1] * (x.dim() - 1)
            beta_t = beta_t.view(*view_shape)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x0, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        if isinstance(t, torch.Tensor):
            view_shape = [t.shape[0]] + [1] * (x0.dim() - 1)
            log_mean_coeff = log_mean_coeff.view(*view_shape)
        mean = torch.exp(log_mean_coeff) * x0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std