"""
Integration tests for SLA2 in PixelGen

Tests verify:
1. Model creation and initialization
2. Forward pass with different configurations
3. Gradient flow and backpropagation
4. Parameter count and structure
5. Attention output shapes and values
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from src.models.layers.sla2_attention import (
    SparseLinearAttention,
    LearnableRouter,
    QuantizationAwareTraining,
)
from src.models.transformer.JiT_SLA2 import (
    JiTSLA2,
    JiTBlockSLA2,
    JiTSLA2_B_16,
)
from src.utils.sla2_training import (
    get_sla2_statistics,
    enable_sla2_qat,
    disable_sla2_qat,
)


class TestSparseLinearAttention:
    """Tests for SparseLinearAttention module"""
    
    @pytest.fixture
    def attention_module(self):
        """Create a SparseLinearAttention module"""
        return SparseLinearAttention(
            dim=768,
            num_heads=12,
            topk_ratio=0.15,
            compression_ratio=8.0,
            enable_qat=True,
        )
    
    def test_attention_creation(self, attention_module):
        """Test attention module creation"""
        assert attention_module is not None
        assert attention_module.dim == 768
        assert attention_module.num_heads == 12
        assert attention_module.topk_ratio == 0.15
    
    def test_forward_pass_shape(self, attention_module):
        """Test forward pass output shapes"""
        batch_size = 2
        seq_len = 256
        dim = 768
        
        x = torch.randn(batch_size, seq_len, dim)
        output = attention_module(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_forward_pass_dtype(self, attention_module):
        """Test output dtype matches input"""
        x = torch.randn(2, 256, 768, dtype=torch.float32)
        output = attention_module(x)
        assert output.dtype == x.dtype
    
    def test_alpha_parameter(self, attention_module):
        """Test learnable alpha parameter"""
        assert hasattr(attention_module, 'alpha')
        assert attention_module.alpha.shape[0] == 12
        assert attention_module.alpha.requires_grad
    
    def test_gradient_flow(self, attention_module):
        """Test gradients flow through attention"""
        x = torch.randn(2, 256, 768, requires_grad=True)
        output = attention_module(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert attention_module.alpha.grad is not None
    
    def test_qat_enable_disable(self, attention_module):
        """Test QAT enable/disable functionality"""
        assert attention_module.qat.enable_qat == True
        
        disable_sla2_qat(attention_module)
        assert attention_module.qat.enable_qat == False
        
        enable_sla2_qat(attention_module)
        assert attention_module.qat.enable_qat == True


class TestLearnableRouter:
    """Tests for LearnableRouter module"""
    
    @pytest.fixture
    def router(self):
        """Create a LearnableRouter"""
        return LearnableRouter(
            head_dim=64,
            compression_ratio=8.0,
            topk_ratio=0.05,
        )
    
    def test_router_creation(self, router):
        """Test router module creation"""
        assert router is not None
        assert router.head_dim == 64
        assert router.compression_ratio == 8.0
    
    def test_compress_tokens(self, router):
        """Test token compression"""
        x = torch.randn(2, 12, 256, 64)  # (B, H, L, D)
        x_compressed = router.compress_tokens(x)
        
        expected_compressed_len = (256 + 7) // 8  # ceil(256 / 8)
        assert x_compressed.shape == (2, 12, expected_compressed_len, 64)
    
    def test_forward_output(self, router):
        """Test router forward pass output"""
        q = torch.randn(2, 12, 256, 64)
        k = torch.randn(2, 12, 256, 64)
        
        routing_scores, topk_ratio = router(q, k)
        
        # Scores should be probabilistic
        assert routing_scores.shape[0] == 2  # Batch
        assert routing_scores.shape[1] == 12  # Heads
        assert (routing_scores >= 0).all()
        assert (routing_scores <= 1).all()
        assert topk_ratio == router.topk_ratio


class TestJiTBlockSLA2:
    """Tests for JiTBlockSLA2 module"""
    
    @pytest.fixture
    def block_sla2(self):
        """Create a JiTBlockSLA2"""
        return JiTBlockSLA2(
            hidden_size=768,
            num_heads=12,
            use_sla2=True,
        )
    
    @pytest.fixture
    def block_standard(self):
        """Create a JiTBlockSLA2 with standard attention"""
        return JiTBlockSLA2(
            hidden_size=768,
            num_heads=12,
            use_sla2=False,
        )
    
    def test_block_creation_sla2(self, block_sla2):
        """Test SLA2 block creation"""
        assert block_sla2 is not None
        assert hasattr(block_sla2, 'attn')
    
    def test_block_forward_sla2(self, block_sla2):
        """Test SLA2 block forward pass"""
        x = torch.randn(2, 256, 768)
        c = torch.randn(2, 768)
        
        output = block_sla2(x, c, feat_rope=None)
        
        assert output.shape == x.shape
    
    def test_block_forward_standard(self, block_standard):
        """Test standard block forward pass"""
        from src.models.transformer.JiT import VisionRotaryEmbeddingFast
        
        x = torch.randn(2, 256, 768)
        c = torch.randn(2, 768)
        rope = VisionRotaryEmbeddingFast(dim=32, pt_seq_len=16)
        
        output = block_standard(x, c, feat_rope=rope)
        
        assert output.shape == x.shape


class TestJiTSLA2:
    """Tests for complete JiTSLA2 model"""
    
    @pytest.fixture
    def model_sla2(self):
        """Create a JiTSLA2 model with SLA2"""
        return JiTSLA2_B_16(use_sla2=True)
    
    @pytest.fixture
    def model_standard(self):
        """Create a JiTSLA2 model without SLA2"""
        return JiTSLA2_B_16(use_sla2=False)
    
    def test_model_creation_sla2(self, model_sla2):
        """Test SLA2 model creation"""
        assert model_sla2 is not None
        assert model_sla2.use_sla2 == True
    
    def test_model_creation_standard(self, model_standard):
        """Test standard model creation"""
        assert model_standard is not None
        assert model_standard.use_sla2 == False
    
    def test_forward_pass_sla2(self, model_sla2):
        """Test SLA2 model forward pass"""
        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)
        t = torch.randint(0, 1000, (batch_size,))
        y = torch.randint(0, 20, (batch_size,))
        
        output = model_sla2(x, t, y)
        
        assert output.shape == x.shape
    
    def test_forward_pass_standard(self, model_standard):
        """Test standard model forward pass"""
        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)
        t = torch.randint(0, 1000, (batch_size,))
        y = torch.randint(0, 20, (batch_size,))
        
        output = model_standard(x, t, y)
        
        assert output.shape == x.shape
    
    def test_gradient_flow_sla2(self, model_sla2):
        """Test SLA2 model gradient flow"""
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        t = torch.randint(0, 1000, (1,))
        y = torch.randint(0, 20, (1,))
        
        output = model_sla2(x, t, y)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check some block parameters have gradients
        for block in model_sla2.blocks:
            if hasattr(block.attn, 'sla2'):
                sla2 = block.attn.sla2
                assert sla2.alpha.grad is not None
    
    def test_parameter_count(self, model_sla2, model_standard):
        """Test parameter counts"""
        sla2_params = sum(p.numel() for p in model_sla2.parameters())
        standard_params = sum(p.numel() for p in model_standard.parameters())
        
        # SLA2 should have more parameters (alpha + router)
        assert sla2_params > standard_params
    
    def test_intermediate_features(self, model_sla2):
        """Test returning intermediate features"""
        x = torch.randn(2, 3, 256, 256)
        t = torch.randint(0, 1000, (2,))
        y = torch.randint(0, 20, (2,))
        
        output, feat = model_sla2(x, t, y, return_layer=6)
        
        assert output.shape == x.shape
        assert feat.dim() == 3  # (B, N, C)


class TestSLA2Utilities:
    """Tests for SLA2 utility functions"""
    
    @pytest.fixture
    def model_sla2(self):
        """Create a test model"""
        return JiTSLA2_B_16(use_sla2=True)
    
    def test_get_sla2_statistics(self, model_sla2):
        """Test getting SLA2 statistics"""
        stats = get_sla2_statistics(model_sla2)
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        assert any('sparsity' in key for key in stats.keys())
        assert any('alpha' in key for key in stats.keys())
    
    def test_enable_disable_qat(self, model_sla2):
        """Test QAT enable/disable"""
        # Check initial state
        has_sla2 = any(hasattr(m, 'sla2') for m in model_sla2.modules())
        assert has_sla2
        
        # Disable QAT
        disable_sla2_qat(model_sla2)
        
        # Enable QAT
        enable_sla2_qat(model_sla2)


class TestSLA2Configurations:
    """Tests for different SLA2 configurations"""
    
    def test_max_sparsity_config(self):
        """Test maximum sparsity configuration"""
        model = JiTSLA2(
            hidden_size=768,
            depth=12,
            num_heads=12,
            use_sla2=True,
            sla2_topk_ratio=0.05,
            sla2_compression_ratio=16.0,
        )
        
        x = torch.randn(1, 3, 256, 256)
        t = torch.randint(0, 1000, (1,))
        y = torch.randint(0, 20, (1,))
        
        output = model(x, t, y)
        assert output.shape == x.shape
    
    def test_balanced_config(self):
        """Test balanced configuration"""
        model = JiTSLA2(
            hidden_size=768,
            depth=12,
            num_heads=12,
            use_sla2=True,
            sla2_topk_ratio=0.15,
            sla2_compression_ratio=8.0,
        )
        
        x = torch.randn(1, 3, 256, 256)
        t = torch.randint(0, 1000, (1,))
        y = torch.randint(0, 20, (1,))
        
        output = model(x, t, y)
        assert output.shape == x.shape
    
    def test_high_quality_config(self):
        """Test high quality configuration"""
        model = JiTSLA2(
            hidden_size=768,
            depth=12,
            num_heads=12,
            use_sla2=True,
            sla2_topk_ratio=0.25,
            sla2_compression_ratio=4.0,
        )
        
        x = torch.randn(1, 3, 256, 256)
        t = torch.randint(0, 1000, (1,))
        y = torch.randint(0, 20, (1,))
        
        output = model(x, t, y)
        assert output.shape == x.shape
    
    def test_hybrid_sla2_config(self):
        """Test hybrid SLA2 (only in later layers)"""
        model = JiTSLA2(
            hidden_size=768,
            depth=12,
            num_heads=12,
            use_sla2=True,
            sla2_start_layer=6,  # Only in last 6 layers
        )
        
        x = torch.randn(1, 3, 256, 256)
        t = torch.randint(0, 1000, (1,))
        y = torch.randint(0, 20, (1,))
        
        output = model(x, t, y)
        assert output.shape == x.shape


# Pytest configuration
@pytest.mark.cuda
class TestGPUCompatibility:
    """Tests for GPU compatibility"""
    
    def test_device_transfer(self):
        """Test model transfer to GPU"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = JiTSLA2_B_16(use_sla2=True)
        model = model.cuda()
        
        x = torch.randn(1, 3, 256, 256, device='cuda')
        t = torch.randint(0, 1000, (1,), device='cuda')
        y = torch.randint(0, 20, (1,), device='cuda')
        
        output = model(x, t, y)
        
        assert output.device.type == 'cuda'
        assert output.shape == x.shape
    
    def test_mixed_precision(self):
        """Test with mixed precision"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = JiTSLA2_B_16(use_sla2=True).cuda()
        
        x = torch.randn(1, 3, 256, 256, device='cuda')
        t = torch.randint(0, 1000, (1,), device='cuda')
        y = torch.randint(0, 20, (1,), device='cuda')
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(x, t, y)
        
        assert output.shape == x.shape


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
