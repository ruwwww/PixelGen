"""
Example: Training and Using SLA2-enabled JiT Model

This script demonstrates how to:
1. Create a JiT model with SLA2 attention
2. Configure QAT training
3. Log SLA2 metrics
4. Run inference with SLA2
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models.transformer.JiT_SLA2 import JiTSLA2, JiTSLA2_B_16
from src.utils.sla2_training import (
    SLA2QATCallback,
    RouterParameterScheduler,
    enable_sla2_qat,
    disable_sla2_qat,
    get_sla2_statistics,
    log_sla2_config,
)


def example_basic_model_creation():
    """Example 1: Create a JiT model with SLA2"""
    print("\n" + "="*80)
    print("Example 1: Creating JiTSLA2 Model")
    print("="*80)
    
    # Create model with SLA2 enabled
    model = JiTSLA2_B_16(
        use_sla2=True,
        sla2_topk_ratio=0.15,
        sla2_compression_ratio=8.0,
        sla2_enable_qat=True,
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Log SLA2 configuration
    log_sla2_config(model, print)
    
    return model


def example_forward_pass(model, device='cuda'):
    """Example 2: Run forward pass with SLA2"""
    print("\n" + "="*80)
    print("Example 2: Forward Pass with SLA2")
    print("="*80)
    
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, 20, (batch_size,), device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t, y)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Forward pass successful!")
    
    return output


def example_sla2_statistics(model):
    """Example 3: Monitor SLA2 statistics"""
    print("\n" + "="*80)
    print("Example 3: SLA2 Statistics")
    print("="*80)
    
    stats = get_sla2_statistics(model)
    
    if stats:
        print("\nSLA2 Module Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("No SLA2 modules found in model")


def example_training_setup():
    """Example 4: Setup Lightning trainer with SLA2 callbacks"""
    print("\n" + "="*80)
    print("Example 4: Training Setup with SLA2 Callbacks")
    print("="*80)
    
    # Create callbacks
    sla2_callback = SLA2QATCallback(
        router_lr_scale=1.0,
        log_sparsity=True,
        log_alpha_stats=True,
    )
    
    router_scheduler = RouterParameterScheduler(
        router_warmup_steps=5000,
        initial_topk_ratio=0.25,  # Start less sparse
        final_topk_ratio=0.10,    # End more sparse
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="sla2-jit-{epoch:02d}",
        save_top_k=3,
        monitor="val_loss",
    )
    
    # Example trainer config
    trainer_config = {
        "max_steps": 1000000,
        "precision": "bf16-mixed",
        "callbacks": [sla2_callback, router_scheduler, checkpoint_callback],
        "log_every_n_steps": 50,
    }
    
    print("Trainer will be created with SLA2 callbacks:")
    print(f"  - SLA2QATCallback (log_sparsity={sla2_callback.log_sparsity})")
    print(f"  - RouterParameterScheduler (warmup={router_scheduler.router_warmup_steps})")
    print(f"  - ModelCheckpoint")
    
    return trainer_config


def example_inference(model, device='cuda'):
    """Example 5: Inference with optimized settings"""
    print("\n" + "="*80)
    print("Example 5: Optimized Inference with SLA2")
    print("="*80)
    
    model = model.to(device)
    model.eval()
    
    # Disable QAT for inference (use learned quantization)
    disable_sla2_qat(model)
    print("Disabled QAT for inference mode")
    
    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, 20, (batch_size,), device=device)
    
    # Inference timing
    import time
    
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(x, t, y)
        
        # Timing
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            _ = model(x, t, y)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
    
    per_image_time = (elapsed / 10 / batch_size) * 1000  # ms
    
    print(f"Inference Time Statistics:")
    print(f"  Total time for 10 batches: {elapsed:.2f}s")
    print(f"  Per image latency: {per_image_time:.2f}ms")
    print(f"  Throughput: {1000/per_image_time:.1f} images/s")


def example_sla2_vs_standard():
    """Example 6: Compare SLA2 vs Standard attention"""
    print("\n" + "="*80)
    print("Example 6: SLA2 vs Standard Attention Comparison")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create both versions
    model_sla2 = JiTSLA2_B_16(use_sla2=True)
    model_standard = JiTSLA2_B_16(use_sla2=False)
    
    model_sla2.to(device)
    model_standard.to(device)
    
    # Count parameters
    sla2_params = sum(p.numel() for p in model_sla2.parameters())
    standard_params = sum(p.numel() for p in model_standard.parameters())
    
    print(f"Model Comparison:")
    print(f"  SLA2 Parameters: {sla2_params:,}")
    print(f"  Standard Parameters: {standard_params:,}")
    print(f"  Difference: {sla2_params - standard_params:,}")
    
    # Get SLA2 statistics
    sla2_stats = get_sla2_statistics(model_sla2)
    
    print(f"\nSLA2 Efficiency Metrics:")
    if sla2_stats:
        for key, value in sla2_stats.items():
            if 'sparsity' in key:
                print(f"  {key}: {value:.2%}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")


def example_custom_config():
    """Example 7: Create models with different SLA2 configurations"""
    print("\n" + "="*80)
    print("Example 7: Custom SLA2 Configurations")
    print("="*80)
    
    configs = [
        {
            "name": "Maximum Sparsity",
            "sla2_topk_ratio": 0.05,
            "sla2_compression_ratio": 16.0,
        },
        {
            "name": "Balanced",
            "sla2_topk_ratio": 0.15,
            "sla2_compression_ratio": 8.0,
        },
        {
            "name": "High Quality",
            "sla2_topk_ratio": 0.25,
            "sla2_compression_ratio": 4.0,
        },
    ]
    
    for config in configs:
        name = config.pop("name")
        model = JiTSLA2_B_16(use_sla2=True, **config)
        
        params = sum(p.numel() for p in model.parameters())
        stats = get_sla2_statistics(model)
        
        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        for key, value in stats.items():
            if 'sparsity' in key:
                print(f"  Sparsity: {value:.2%}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("SLA2-Enabled JiT Model Examples")
    print("="*80)
    
    # Example 1: Create model
    model = example_basic_model_creation()
    
    # Example 3: Check statistics
    example_sla2_statistics(model)
    
    # Example 4: Training setup
    example_training_setup()
    
    # Example 6: Comparison
    example_sla2_vs_standard()
    
    # Example 7: Custom configs
    example_custom_config()
    
    # Only run GPU examples if available
    if torch.cuda.is_available():
        print("\nGPU Available - Running GPU Examples:")
        # Example 2: Forward pass
        example_forward_pass(model)
        
        # Example 5: Inference
        example_inference(model)
    else:
        print("\nGPU not available - skipping GPU examples")
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
