import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1

def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)

class PixelFlowMatchingTrainer(BaseTrainer):
    """
    Flow Matching Trainer for Pixel Space models.
    Always uses x_pred parameterization (predicting x_0 directly) which is converted
    to velocity target during training.
    """
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            lognorm_t=False,
            timeshift=1.0,
            t_eps=0.05,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.t_eps = t_eps
        
    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch_size = x.shape[0]
        if self.lognorm_t:
            t = torch.randn(batch_size).to(x.device, x.dtype).sigmoid()
        else:
            t = torch.rand(batch_size).to(x.device, x.dtype)
        t = time_shift_fn(t, self.timeshift)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)
        # w = self.scheduler.w(t) # Unused in this logic

        # Linear Scheduler convention: alpha=t, sigma=1-t
        x_t = alpha * x + noise * sigma
        
        # Target velocity v_t = dx_t/dt = x - noise (for Linear Schedule)
        v_t = dalpha * x + dsigma * noise
        
        out = net(x_t, t, y)

        # Convert x_0 prediction to velocity v
        out = (out - x_t) / sigma.clamp_min(self.t_eps)

        weight = self.loss_weight_fn(alpha, sigma)

        loss = weight*(out - v_t)**2

        out = dict(
            loss=loss.mean(),
            fm_loss=loss.mean().detach(),
        )
        return out
