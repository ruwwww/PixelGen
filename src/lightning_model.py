from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback


from src.models.autoencoder.base import BaseAE, fp2uint8
from src.models.conditioner.base import BaseConditioner
from src.utils.model_loader import ModelLoader
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

torch._functorch.config.donated_buffer = False

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]

class LightningModel(pl.LightningModule):
    def __init__(self,
                 vae: BaseAE,
                 conditioner: BaseConditioner,
                 denoiser: nn.Module,
                 diffusion_trainer: BaseTrainer,
                 diffusion_sampler: BaseSampler,
                 ema_tracker: SimpleEMA=None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 eval_original_model: bool = False,
                 muon_cfg: Optional[Mapping] = None,
                 ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        # optional muon settings (populated from config)
        self.muon_cfg = dict(muon_cfg) if muon_cfg is not None else {}

        self._strict_loading = False

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        # disable grad for conditioner and vae
        no_grad(self.conditioner)
        no_grad(self.vae)
        # no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)

        # torch.compile
        self.denoiser.compile()
        self.ema_denoiser.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())

        # Try to detect Muon and prepare Muon-style param groups (hidden weights -> Muon, others -> Adam)
        try:
            import muon
            muon_class = getattr(muon, "MuonWithAuxAdam", None)
        except Exception:
            muon_class = None

        is_muon_configured = False
        try:
            if muon_class is not None:
                opt_name = getattr(self.optimizer, "__name__", None) or getattr(self.optimizer, "__class__", None)
                if self.optimizer is muon_class or (isinstance(opt_name, str) and "MuonWithAuxAdam" in opt_name) or getattr(self.optimizer, "__module__", "")=="muon":
                    is_muon_configured = True
        except Exception:
            is_muon_configured = False

        # Resolve hyperparameters from config (defaults follow recommended values)
        muon_defaults = {
            "lr": 0.02,
            "weight_decay": 0.01,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
        }
        aux_defaults = {
            "lr": 3e-4,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
        }
        # override with provided muon_cfg
        muon_h = dict(muon_defaults)
        aux_h = dict(aux_defaults)
        if isinstance(self.muon_cfg, Mapping):
            muon_h.update(self.muon_cfg.get("muon_group", {}))
            aux_h.update(self.muon_cfg.get("aux_group", {}))

        # default param groups (non-muon)
        param_groups = [
            {"params": params_denoiser, },
            {"params": params_trainer,},
            {"params": params_sampler, "lr": 1e-3},
        ]

        # If the optimizer appears to be Muon (class or module name contains 'muon'/'Muon'),
        # build muon-style groups (hidden weights + aux) proactively.
        try:
            import muon as _muon
            muon_class = getattr(_muon, "MuonWithAuxAdam", None)
        except Exception:
            muon_class = None

        opt_name = getattr(self.optimizer, "__name__", None) or getattr(self.optimizer, "__class__", None)
        looks_like_muon = False
        try:
            if muon_class is not None and (self.optimizer is muon_class or (isinstance(opt_name, str) and "MuonWithAuxAdam" in opt_name) or getattr(self.optimizer, "__module__", "").startswith("muon") or "muon" in str(self.optimizer).lower()):
                looks_like_muon = True
        except Exception:
            looks_like_muon = False

        if looks_like_muon:
            hidden_weights = [p for p in params_denoiser if getattr(p, "ndim", 0) >= 2 and p.requires_grad]
            aux_from_denoiser = [p for p in params_denoiser if not (getattr(p, "ndim", 0) >= 2 and p.requires_grad)]
            aux = aux_from_denoiser + list(params_trainer) + list(params_sampler) + list(filter_nograd_tensors(self.vae.parameters())) + list(filter_nograd_tensors(self.conditioner.parameters()))

            new_groups = []
            if hidden_weights:
                new_groups.append({
                    "params": hidden_weights,
                    "use_muon": True,
                    "lr": float(muon_h.get("lr", 0.02)),
                    "momentum": float(muon_h.get("momentum", 0.95)),
                    "weight_decay": float(muon_h.get("weight_decay", 0.01)),
                })
            if aux:
                new_groups.append({
                    "params": aux,
                    "use_muon": False,
                    "lr": float(aux_h.get("lr", 3e-4)),
                    "betas": tuple(aux_h.get("betas", (0.9, 0.95))),
                    "eps": float(aux_h.get("eps", 1e-10)),
                    "weight_decay": float(aux_h.get("weight_decay", 0.01)),
                })
            if new_groups:
                param_groups = new_groups

        # instantiate optimizer; if Muon complains about missing keys, rebuild muon-style groups and retry
        try:
            optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        except AssertionError as e:
            msg = str(e)
            should_retry = False
            if muon_class is not None:
                try:
                    if self.optimizer is muon_class or (isinstance(opt_name, str) and "MuonWithAuxAdam" in opt_name) or getattr(self.optimizer, "__module__", "").startswith("muon") or "muon" in str(self.optimizer).lower() or "use_muon" in msg:
                        should_retry = True
                except Exception:
                    should_retry = True
            if should_retry:
                # rebuild muon-style groups from denoiser params
                hidden_weights = [p for p in params_denoiser if getattr(p, "ndim", 0) >= 2 and p.requires_grad]
                aux_from_denoiser = [p for p in params_denoiser if not (getattr(p, "ndim", 0) >= 2 and p.requires_grad)]
                aux = aux_from_denoiser + list(params_trainer) + list(params_sampler) + list(filter_nograd_tensors(self.vae.parameters())) + list(filter_nograd_tensors(self.conditioner.parameters()))

                param_groups = []
                if hidden_weights:
                    param_groups.append({
                        "params": hidden_weights,
                        "use_muon": True,
                        "lr": float(muon_h.get("lr", 0.02)),
                        "momentum": float(muon_h.get("momentum", 0.95)),
                        "weight_decay": float(muon_h.get("weight_decay", 0.01)),
                    })
                if aux:
                    param_groups.append({
                        "params": aux,
                        "use_muon": False,
                        "lr": float(aux_h.get("lr", 3e-4)),
                        "betas": tuple(aux_h.get("betas", (0.9, 0.95))),
                        "eps": float(aux_h.get("eps", 1e-10)),
                        "weight_decay": float(aux_h.get("weight_decay", 0.01)),
                    })
                optimizer = self.optimizer(param_groups)
            else:
                raise

        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler={
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "learning_rate"
                }
            )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    # sanity check before training start
    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

        # Ensure a default process group exists for optimizers that expect distributed groups (e.g., Muon)
        try:
            import torch.distributed as dist
            if dist.is_available() and not dist.is_initialized():
                # Use NCCL if CUDA is available, otherwise GLOO
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                # Provide sane defaults for single-process init
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29500")
                try:
                    from datetime import timedelta
                    dist.init_process_group(backend=backend, rank=0, world_size=1, timeout=timedelta(seconds=30))
                    print(f"[Info] Initialized default process group with backend={backend} world_size=1")
                except Exception as e:
                    # If init fails, don't crash — Muon will still try and may raise a more specific error
                    print(f"[Warning] init_process_group failed: {e}")
        except Exception:
            pass

        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def on_load_checkpoint(self, checkpoint):
        keys_to_check = [
            "denoiser.pos_embed", 
            "ema_denoiser.pos_embed"
        ]
        ckpt_state_dict = checkpoint["state_dict"]
      
        current_state_dict = self.state_dict()

        for key in keys_to_check:
            if key in ckpt_state_dict and key in current_state_dict:
                ckpt_shape = ckpt_state_dict[key].shape
                curr_shape = current_state_dict[key].shape
                if ckpt_shape != curr_shape:
                    print(f"[Warning] Shape mismatch for '{key}': "
                          f"Checkpoint {ckpt_shape} vs Current {curr_shape}. "
                          f"Dropping from checkpoint to avoid RuntimeError.")
                    del ckpt_state_dict[key]
                else:
                    pass

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        if metadata is None:
            metadata = {}
        metadata['global_step'] = self.global_step
        with torch.no_grad():
            x = self.vae.encode(x)
            condition, uncondition = self.conditioner(y, metadata)
        loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)
        # to be do! fix the bug in tqdm iteration when enabling accumulate_grad_batches>1
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        # sample images
        if self.eval_original_model:
            samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)
        else:
            samples = self.diffusion_sampler(self.ema_denoiser, xT, condition, uncondition)

        samples = self.vae.decode(samples)
        # fp32 -1,1 -> uint8 0,255
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"ema_denoiser.",
            keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix+"diffusion_trainer.",
            keep_vars=keep_vars)
        return destination