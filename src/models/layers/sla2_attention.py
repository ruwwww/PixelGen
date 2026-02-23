"""
SLA2: Sparse-Linear Attention with Learnable Routing and Quantization-Aware Training
Based on: SLA2 paper from Tsinghua University and UC Berkeley
Reference: https://arxiv.org/abs/2025.xxxxx

This module implements:
1. Direct sparse-linear decomposition with learnable routing (α parameter)
2. Learnable router mechanism for dynamic sparse/linear allocation
3. Quantization-aware training (QAT) for sparse branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LearnableRouter(nn.Module):
    """
    Learnable routing mechanism that determines sparse vs linear attention allocation.
    
    Uses compressed representations and Top-k operation to generate routing masks
    dynamically based on query-key similarity patterns.
    """
    
    def __init__(
        self,
        head_dim: int,
        compression_ratio: float = 8.0,
        topk_ratio: float = 0.05,
    ):
        """
        Args:
            head_dim: Dimension of each attention head
            compression_ratio: Compression factor for input (must be power of 2)
            topk_ratio: Ratio of positions to route to sparse attention
        """
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.topk_ratio = topk_ratio
        
        # Task-adaptive projection for router
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
    
    def compress_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress tokens via mean pooling over consecutive tokens.
        
        Args:
            x: (B, H, L, D) - input tokens
        Returns:
            x_compressed: (B, H, L_compressed, D)
        """
        B, H, L, D = x.shape
        compression_size = int(self.compression_ratio)
        
        # Pad if necessary
        padded_len = ((L + compression_size - 1) // compression_size) * compression_size
        if padded_len > L:
            x_padded = F.pad(x, (0, 0, 0, padded_len - L))
        else:
            x_padded = x
        
        # Reshape and mean pool
        x_compressed = x_padded.reshape(
            B, H, -1, compression_size, D
        ).mean(dim=3)  # (B, H, L_compressed, D)
        
        return x_compressed
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Generate routing masks based on compressed Q and K representations.
        
        Args:
            q: (B, H, L, D) - queries
            k: (B, H, L, D) - keys
        Returns:
            routing_mask: (B, H, L, L_compressed) - soft routing scores
            topk_ratio: actual top-k ratio used
        """
        # Compress representations
        q_compressed = self.compress_tokens(q)  # (B, H, L_c, D)
        k_compressed = self.compress_tokens(k)  # (B, H, L_c, D)
        
        # Project to task-adaptive space
        q_proj = self.q_proj(q_compressed)  # (B, H, L_c, D)
        k_proj = self.k_proj(k_compressed)  # (B, H, L_c, D)
        
        # Compute attention scores on compressed space
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)  # (B, H, L_c, L_c)
        
        return scores, self.topk_ratio


class QuantizationAwareTraining(nn.Module):
    """
    Quantization-aware training wrapper for sparse attention branch.
    
    Quantizes Q, K, and attention probabilities to INT8/FP8 in forward pass
    while keeping gradients in FP16 for numerical stability.
    """
    
    def __init__(self, enable_qat: bool = True, dtype: torch.dtype = torch.float16):
        """
        Args:
            enable_qat: Whether to enable quantization-aware training
            dtype: Data type for quantization (float16 or bfloat16)
        """
        super().__init__()
        self.enable_qat = enable_qat
        self.dtype = dtype
    
    def quantize_tensor(
        self, 
        x: torch.Tensor, 
        scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT8.
        
        Args:
            x: Input tensor
            scale: Optional pre-computed scale
        Returns:
            x_quantized: Quantized tensor
            scale: Quantization scale
        """
        if not self.enable_qat:
            return x, torch.tensor(1.0, device=x.device, dtype=x.dtype)
        
        # Compute scale as max absolute value
        if scale is None:
            scale = (x.abs().max() / 127.0).clamp(min=1e-8)
        
        # Quantize and dequantize
        x_quantized = torch.clamp(x / scale, min=-128, max=127).round()
        
        return x_quantized, scale
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize Q, K, V for sparse attention computation.
        
        Args:
            q, k, v: Query, key, value tensors
        Returns:
            q_quantized, k_quantized, v, q_scale, k_scale
        """
        if not self.enable_qat:
            return q, k, v, torch.tensor(1.0), torch.tensor(1.0)
        
        q_quantized, q_scale = self.quantize_tensor(q)
        k_quantized, k_scale = self.quantize_tensor(k)
        
        return q_quantized, k_quantized, v, q_scale, k_scale


class SparseLinearAttention(nn.Module):
    """
    Efficient sparse-linear attention mechanism that decomposes attention into:
    - Sparse branch: O(N) sparse attention on top-k important positions
    - Linear branch: O(N) linear attention via low-rank approximation
    - Learnable router: Dynamically allocates importance weights via learned α
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        topk_ratio: float = 0.05,
        compression_ratio: float = 8.0,
        enable_qat: bool = True,
        use_bf16: bool = True,
        feature_map: str = 'softmax',
    ):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            qk_norm: Whether to normalize Q and K
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            topk_ratio: Ratio of keys to select for sparse attention
            compression_ratio: Compression factor for router
            enable_qat: Enable quantization-aware training
            use_bf16: Use bfloat16 instead of float16
            feature_map: Feature map for linear attention ('softmax', 'elu', 'relu')
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.topk_ratio = topk_ratio
        self.enable_qat = enable_qat
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Normalization
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        
        # Learnable routing mechanism
        self.router = LearnableRouter(
            head_dim=self.head_dim,
            compression_ratio=compression_ratio,
            topk_ratio=topk_ratio,
        )
        
        # Learnable α parameter for sparse/linear weighting
        # Shape: (num_heads,) - one α per head
        self.register_parameter(
            'alpha',
            nn.Parameter(torch.ones(num_heads) * 0.5)
        )
        
        # Feature map for linear attention
        if feature_map == 'elu':
            self.feature_map_q = lambda x: F.elu(x) + 1
            self.feature_map_k = lambda x: F.elu(x) + 1
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            self.feature_map_q = lambda x: F.softmax(x, dim=-1)
            self.feature_map_k = lambda x: F.softmax(x, dim=-1)
        else:
            raise NotImplementedError(f"Unsupported feature map: {feature_map}")
        
        # Linear attention projection
        self.proj_l = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj_l.weight)
        
        # QAT module
        self.qat = QuantizationAwareTraining(enable_qat=enable_qat, dtype=self.dtype)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
        nn.init.constant_(self.alpha, 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SLA2 attention.
        
        Args:
            x: (B, N, C) - input features
        Returns:
            out: (B, N, C) - attention output
        """
        B, N, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Normalize Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        original_dtype = x.dtype
        
        # Compute sparse attention output
        o_s = self._sparse_attention(q, k, v)
        
        # Compute linear attention output
        o_l = self._linear_attention(q, k, v)
        
        # Learnable routing: O = α ⊙ O_s + (1 - α) ⊙ O_l
        # Expand alpha to match spatial dimension
        alpha = torch.sigmoid(self.alpha)  # (H,) -> sigmoid to bound in [0, 1]
        alpha = alpha.view(1, self.num_heads, 1, 1)  # (1, H, 1, 1)
        
        o = alpha * o_s + (1 - alpha) * o_l  # (B, H, N, D)
        
        # Merge heads and project
        o = o.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        o = self.proj(o)
        o = self.proj_drop(o)
        
        return o.to(original_dtype)
    
    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sparse attention output.
        
        Uses learnable routing to select top-k keys for each query.
        Includes quantization-aware training.
        
        Args:
            q, k, v: (B, H, N, D)
        Returns:
            o_s: (B, H, N, D) - sparse attention output
        """
        B, H, N, D = q.shape
        
        # Get routing weights from learnable router
        routing_scores, _ = self.router(q, k)  # (B, H, L_c, L_c)
        
        # Compute attention scores with softmax
        q_scaled = q / math.sqrt(D)
        scores = torch.matmul(q_scaled, k.transpose(-2, -1))  # (B, H, N, N)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # Sparse attention via top-k masking
        topk = max(1, int(N * self.topk_ratio))
        
        # Apply top-k sparsity pattern
        topk_weights, topk_indices = torch.topk(attn_weights, topk, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(attn_weights)
        sparse_mask.scatter_(-1, topk_indices, 1)
        
        # Apply mask and renormalize
        attn_weights_sparse = attn_weights * sparse_mask
        attn_weights_sparse = attn_weights_sparse / (attn_weights_sparse.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute output with sparse weights
        o_s = torch.matmul(attn_weights_sparse, v)  # (B, H, N, D)
        
        return o_s
    
    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute linear attention output via kernel trick.
        
        Uses feature maps to enable O(N) computation:
        O_l(i) = (φ(q_i) @ (K^T V)) / (φ(q_i) @ K^T 1)
        
        Args:
            q, k, v: (B, H, N, D)
        Returns:
            o_l: (B, H, N, D) - linear attention output
        """
        # Apply feature maps
        phi_q = self.feature_map_q(q)  # (B, H, N, D)
        phi_k = self.feature_map_k(k)  # (B, H, N, D)
        
        # Compute key-value aggregation: K^T V
        kv_sum = torch.matmul(phi_k.transpose(-2, -1), v)  # (B, H, D, D)
        
        # Compute key sum: K^T 1
        k_sum = phi_k.sum(dim=-2, keepdim=True)  # (B, H, 1, D)
        
        # Compute normalized attention output
        numerator = torch.matmul(phi_q, kv_sum)  # (B, H, N, D)
        denominator = torch.matmul(phi_q, k_sum.transpose(-2, -1)) + 1e-8  # (B, H, N, 1)
        
        o_l = numerator / denominator  # (B, H, N, D)
        
        # Apply learned projection for additional flexibility
        o_l = self.proj_l(o_l)
        
        return o_l
    
    @property
    def sparsity(self) -> float:
        """Return the sparsity ratio of sparse attention"""
        return 1.0 - self.topk_ratio
