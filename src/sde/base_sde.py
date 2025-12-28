import abc
import torch

class SDE(abc.ABC):
    """SDE 추상 클래스: dX_t = f(x, t)dt + g(t)dW_t"""
    def __init__(self, T=1.0):
        self.T = T

    @abc.abstractmethod
    def sde(self, x, t):
        """Drift(f)와 Diffusion(g) 계수 반환"""
        pass

    @abc.abstractmethod
    def marginal_prob(self, x0, t):
        """p(x_t | x_0)의 평균과 표준편차 반환"""
        pass

    def prior_sampling(self, shape):
        """
        Prior 분포 p(x_T)에서 샘플링.
        VP-SDE의 경우 Standard Normal Distribution N(0, I)입니다.
        """
        return torch.randn(*shape)