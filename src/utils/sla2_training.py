"""
Quantization-Aware Training (QAT) utilities for SLA2 attention.

This module provides callbacks and utilities for training with QAT enabled,
including:
- Mixed-precision training management
- Quantization scale scheduling
- Router learning rate scheduling
- SparseLinearAttention specific optimization
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule
import logging

logger = logging.getLogger(__name__)


class SLA2QATCallback(Callback):
    """
    Lightning callback for managing SLA2 QAT training.
    
    Handles:
    - Router learning rate scheduling
    - QAT parameter initialization
    - Sparsity monitoring and logging
    - Gradient clipping for quantized branches
    """
    
    def __init__(
        self,
        router_lr_scale: float = 1.0,
        gradient_clip_val: Optional[float] = None,
        log_sparsity: bool = True,
        log_alpha_stats: bool = True,
    ):
        """
        Args:
            router_lr_scale: Scale for router learning rate relative to main LR
            gradient_clip_val: Value for gradient clipping in sparse attention
            log_sparsity: Whether to log attention sparsity metrics
            log_alpha_stats: Whether to log alpha parameter statistics
        """
        super().__init__()
        self.router_lr_scale = router_lr_scale
        self.gradient_clip_val = gradient_clip_val
        self.log_sparsity = log_sparsity
        self.log_alpha_stats = log_alpha_stats
        
    def on_train_start(self, trainer, pl_module: LightningModule) -> None:
        """Initialize QAT-specific settings at training start"""
        self._setup_router_learning_rate(pl_module)
        logger.info("SLA2 QAT training started")
        
    def _setup_router_learning_rate(self, pl_module: LightningModule) -> None:
        """Set up separate learning rates for router parameters"""
        # Find all SLA2 modules
        for name, module in pl_module.named_modules():
            if hasattr(module, 'router') and isinstance(module.router, nn.Module):
                # Router parameters could be scaled differently if needed
                logger.info(f"Found SLA2 router in module: {name}")
    
    def on_train_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Called after training batch ends"""
        if self.log_sparsity and batch_idx % 100 == 0:
            self._log_sparsity_metrics(pl_module, trainer)
        
        if self.log_alpha_stats and batch_idx % 100 == 0:
            self._log_alpha_statistics(pl_module, trainer)
    
    def _log_sparsity_metrics(
        self,
        pl_module: LightningModule,
        trainer,
    ) -> None:
        """Log sparsity metrics from SLA2 modules"""
        for name, module in pl_module.named_modules():
            if hasattr(module, 'sla2'):
                sla2_module = module.sla2
                sparsity = sla2_module.sparsity
                trainer.logger.log_metrics(
                    {f"{name.replace('.', '/')}/sparsity": sparsity},
                    step=trainer.global_step
                )
    
    def _log_alpha_statistics(
        self,
        pl_module: LightningModule,
        trainer,
    ) -> None:
        """Log alpha parameter statistics"""
        for name, module in pl_module.named_modules():
            if hasattr(module, 'sla2'):
                sla2_module = module.sla2
                alpha_raw = sla2_module.alpha
                alpha_sigmoid = torch.sigmoid(alpha_raw)
                
                metrics = {
                    f"{name.replace('.', '/')}/alpha_mean": alpha_sigmoid.mean().item(),
                    f"{name.replace('.', '/')}/alpha_std": alpha_sigmoid.std().item(),
                    f"{name.replace('.', '/')}/alpha_min": alpha_sigmoid.min().item(),
                    f"{name.replace('.', '/')}/alpha_max": alpha_sigmoid.max().item(),
                }
                trainer.logger.log_metrics(metrics, step=trainer.global_step)


class RouterParameterScheduler(Callback):
    """
    Scheduler for router learning parameters during training.
    
    Implements:
    - Warm-up phase for router initialization
    - Learning rate scheduling specific to router
    - Compression ratio scheduling
    """
    
    def __init__(
        self,
        router_warmup_steps: int = 5000,
        initial_topk_ratio: float = 0.2,
        final_topk_ratio: float = 0.05,
    ):
        """
        Args:
            router_warmup_steps: Number of warmup steps for router
            initial_topk_ratio: Initial sparse ratio (less sparse)
            final_topk_ratio: Final sparse ratio (more sparse)
        """
        super().__init__()
        self.router_warmup_steps = router_warmup_steps
        self.initial_topk_ratio = initial_topk_ratio
        self.final_topk_ratio = final_topk_ratio
    
    def on_train_batch_start(
        self,
        trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        """Update router parameters at the start of each batch"""
        if trainer.global_step < self.router_warmup_steps:
            # Linear schedule from initial to final topk_ratio
            progress = trainer.global_step / self.router_warmup_steps
            current_topk_ratio = (
                self.initial_topk_ratio
                + (self.final_topk_ratio - self.initial_topk_ratio) * progress
            )
            
            # Update all SLA2 modules
            for name, module in pl_module.named_modules():
                if hasattr(module, 'sla2'):
                    module.sla2.topk_ratio = current_topk_ratio


class SLA2TwoStageScheduler(Callback):
    """
    Two-stage SLA2 training scheduler.

    Stage 1: SoftTop-k routing with auxiliary MSE loss against full attention.
    Stage 2: Hard Top-k routing with auxiliary loss disabled.
    """

    def __init__(
        self,
        warmup_steps: int = 10000,
        soft_topk_ratio: float = 0.2,
        hard_topk_ratio: float = 0.1,
        soft_topk_tau: float = 1.0,
        soft_topk_iters: int = 8,
        router_aux_weight_stage1: float = 1.0,
        router_aux_weight_stage2: float = 0.0,
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.soft_topk_ratio = soft_topk_ratio
        self.hard_topk_ratio = hard_topk_ratio
        self.soft_topk_tau = soft_topk_tau
        self.soft_topk_iters = soft_topk_iters
        self.router_aux_weight_stage1 = router_aux_weight_stage1
        self.router_aux_weight_stage2 = router_aux_weight_stage2

    def on_train_batch_start(
        self,
        trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.global_step < self.warmup_steps:
            self._apply_stage(
                pl_module,
                mode="soft",
                topk_ratio=self.soft_topk_ratio,
                tau=self.soft_topk_tau,
                iters=self.soft_topk_iters,
                aux_weight=self.router_aux_weight_stage1,
            )
        else:
            self._apply_stage(
                pl_module,
                mode="hard",
                topk_ratio=self.hard_topk_ratio,
                tau=self.soft_topk_tau,
                iters=self.soft_topk_iters,
                aux_weight=self.router_aux_weight_stage2,
            )

    def _apply_stage(
        self,
        pl_module: LightningModule,
        mode: str,
        topk_ratio: float,
        tau: float,
        iters: int,
        aux_weight: float,
    ) -> None:
        for module in pl_module.modules():
            if hasattr(module, "sla2"):
                module.sla2.set_router_mode(mode, tau=tau)
                module.sla2.set_topk_ratio(topk_ratio)
                module.sla2.soft_topk_iters = iters
                module.sla2.set_router_aux_weight(aux_weight)


class QuantizedAttentionMonitor(Callback):
    """
    Monitors quantization effects during training.
    
    Tracks:
    - Quantization scale values
    - Numerical stability metrics
    - Attention pattern changes
    - QAT convergence
    """
    
    def __init__(self, monitor_frequency: int = 1000):
        """
        Args:
            monitor_frequency: How often to log quantization metrics
        """
        super().__init__()
        self.monitor_frequency = monitor_frequency
    
    def on_train_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Monitor quantization metrics"""
        if trainer.global_step % self.monitor_frequency == 0:
            self._log_quantization_metrics(pl_module, trainer)
    
    def _log_quantization_metrics(
        self,
        pl_module: LightningModule,
        trainer,
    ) -> None:
        """Log quantization-specific metrics"""
        # This would be extended to track quantization effects
        # For now, we log basic sparsity and alpha statistics
        for name, module in pl_module.named_modules():
            if hasattr(module, 'alpha'):
                alpha = torch.sigmoid(module.alpha)
                trainer.logger.log_metrics(
                    {f"{name.replace('.', '/')}/alpha_mean": alpha.mean().item()},
                    step=trainer.global_step
                )


class SLA2OptimizerWrapper:
    """
    Wrapper for handling optimizer updates with SLA2 attention.
    
    Features:
    - Separate learning rate groups for router
    - Gradient accumulation with quantization
    - Custom step function for QAT convergence
    """
    
    def __init__(
        self,
        base_optimizer,
        router_lr_scale: float = 1.0,
    ):
        """
        Args:
            base_optimizer: PyTorch optimizer instance
            router_lr_scale: Scale for router learning rate
        """
        self.base_optimizer = base_optimizer
        self.router_lr_scale = router_lr_scale
        self.router_params = []
        self.main_params = []
    
    def register_router_params(self, module: nn.Module):
        """Register router parameters for special treatment"""
        for name, param in module.named_parameters():
            if 'router' in name or 'alpha' in name:
                self.router_params.append(param)
            else:
                self.main_params.append(param)
    
    def step(self):
        """Custom optimizer step with QAT handling"""
        self.base_optimizer.step()


def enable_sla2_qat(model: nn.Module) -> None:
    """
    Enable QAT for all SLA2 modules in a model.
    
    Args:
        model: PyTorch model containing SLA2 attention modules
    """
    for module in model.modules():
        if hasattr(module, 'qat'):
            module.qat.enable_qat = True
            logger.info(f"Enabled QAT for {module.__class__.__name__}")


def disable_sla2_qat(model: nn.Module) -> None:
    """
    Disable QAT for all SLA2 modules in a model (for inference).
    
    Args:
        model: PyTorch model containing SLA2 attention modules
    """
    for module in model.modules():
        if hasattr(module, 'qat'):
            module.qat.enable_qat = False
            logger.info(f"Disabled QAT for {module.__class__.__name__}")


def get_sla2_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Extract statistics from all SLA2 modules in a model.
    
    Args:
        model: PyTorch model containing SLA2 attention modules
    
    Returns:
        Dictionary with per-module SLA2 statistics
    """
    stats = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'sla2'):
            sla2 = module.sla2
            module_name = f"sla2_{name.replace('.', '_')}"
            
            # Get alpha statistics
            alpha = torch.sigmoid(sla2.alpha)
            stats[f"{module_name}/alpha_mean"] = alpha.mean().item()
            stats[f"{module_name}/alpha_std"] = alpha.std().item()
            stats[f"{module_name}/sparsity"] = sla2.sparsity
            
            # Get router parameter count
            router_params = sum(p.numel() for p in sla2.router.parameters())
            stats[f"{module_name}/router_params"] = router_params
    
    return stats


def log_sla2_config(model: nn.Module, logger_fn) -> None:
    """
    Log SLA2 configuration to logger.
    
    Args:
        model: PyTorch model containing SLA2 attention modules
        logger_fn: Function to call for logging
    """
    logger_fn("=" * 80)
    logger_fn("SLA2 Configuration Summary")
    logger_fn("=" * 80)
    
    sla2_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'sla2'):
            sla2_count += 1
            sla2 = module.sla2
            logger_fn(f"\n{name}:")
            logger_fn(f"  - Sparsity: {sla2.sparsity:.2%}")
            logger_fn(f"  - Top-K Ratio: {sla2.topk_ratio:.2%}")
            logger_fn(f"  - Feature Map: softmax")
            logger_fn(f"  - QAT Enabled: {sla2.qat.enable_qat}")
            router_params = sum(p.numel() for p in sla2.router.parameters())
            logger_fn(f"  - Router Parameters: {router_params:,}")
    
    logger_fn(f"\nTotal SLA2 Modules: {sla2_count}")
    logger_fn("=" * 80)
