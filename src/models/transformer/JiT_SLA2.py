"""
SLA2-integrated JiT Transformer

This module provides a version of the JiT transformer that uses SLA2
(Sparse-Linear Attention with Learnable Routing) instead of standard
scaled dot-product attention.

Supports both standard attention and SLA2 attention via a configuration flag.
"""

import torch
import torch.nn as nn
from src.models.transformer.JiT import (
    VisionRotaryEmbeddingFast,
    RMSNorm,
    TimestepEmbedder,
    LabelEmbedder,
    BottleneckPatchEmbed,
    PatchEmbed,
    SwiGLUFFN,
    FinalLayer,
    modulate,
    get_2d_sincos_pos_embed,
)
from src.models.layers.sla2_attention import SparseLinearAttention
from torch.nn.functional import scaled_dot_product_attention


class SLA2Attention(nn.Module):
    """
    SLA2-based attention module that replaces standard scaled dot-product attention.
    
    Provides efficient sparse-linear attention with learnable routing for JiT.
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
        router_mode: str = "hard",
        soft_topk_tau: float = 1.0,
        soft_topk_iters: int = 8,
        router_aux_weight: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.sla2 = SparseLinearAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            topk_ratio=topk_ratio,
            compression_ratio=compression_ratio,
            enable_qat=enable_qat,
            use_bf16=use_bf16,
            feature_map='softmax',
            router_mode=router_mode,
            soft_topk_tau=soft_topk_tau,
            soft_topk_iters=soft_topk_iters,
            router_aux_weight=router_aux_weight,
        )
    
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        """
        Forward pass using SLA2 attention.
        
        Args:
            x: (B, N, C) - input features
            rope: Rotary position embedding (currently unused in SLA2, 
                  but kept for API compatibility with standard Attention)
        Returns:
            out: (B, N, C) - attention output
        """
        return self.sla2(x)


class StandardAttention(nn.Module):
    """
    Standard scaled dot-product attention with RoPE and normalization.
    
    This is the original JiT attention for comparison.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, rope) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = rope(q)
        k = rope(k)
        
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class JiTBlockSLA2(nn.Module):
    """
    JiT transformer block with optional SLA2 attention.
    
    Can switch between standard attention and SLA2 attention.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sla2: bool = False,
        sla2_topk_ratio: float = 0.05,
        sla2_compression_ratio: float = 8.0,
        sla2_enable_qat: bool = True,
        sla2_router_mode: str = "hard",
        sla2_soft_topk_tau: float = 1.0,
        sla2_soft_topk_iters: int = 8,
        sla2_router_aux_weight: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        
        # Create attention module
        if use_sla2:
            self.attn = SLA2Attention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                topk_ratio=sla2_topk_ratio,
                compression_ratio=sla2_compression_ratio,
                enable_qat=sla2_enable_qat,
                use_bf16=True,
                router_mode=sla2_router_mode,
                soft_topk_tau=sla2_soft_topk_tau,
                soft_topk_iters=sla2_soft_topk_iters,
                router_aux_weight=sla2_router_aux_weight,
            )
        else:
            self.attn = StandardAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.use_sla2 = use_sla2
    
    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        feat_rope=None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # For SLA2, rope is not used, pass None
        rope_to_use = feat_rope if not self.use_sla2 else None
        
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=rope_to_use
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class JiTSLA2(nn.Module):
    """
    JiT transformer with optional SLA2 attention mechanism.
    
    This is a drop-in replacement for the standard JiT that supports
    both standard attention and SLA2 attention.
    """
    
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        use_bottleneck: bool = True,
        in_context_len: int = 32,
        in_context_start: int = 8,
        # SLA2-specific parameters
        use_sla2: bool = False,
        sla2_start_layer: int = 0,  # Apply SLA2 from this layer onwards
        sla2_topk_ratio: float = 0.05,
        sla2_compression_ratio: float = 8.0,
        sla2_enable_qat: bool = True,
        sla2_router_mode: str = "hard",
        sla2_soft_topk_tau: float = 1.0,
        sla2_soft_topk_iters: int = 8,
        sla2_router_aux_weight: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.use_bottleneck = use_bottleneck
        self.use_sla2 = use_sla2
        self.sla2_start_layer = sla2_start_layer
        
        # Time and class embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        
        # Patch embedding
        if self.use_bottleneck:
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
            )
        else:
            self.x_embedder = PatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
            )
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # In-context positional embedding
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size),
                requires_grad=True
            )
            torch.nn.init.normal_(self.in_context_posemb, std=.02)
        
        # RoPE
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            JiTBlockSLA2(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                use_sla2=use_sla2 and (i >= sla2_start_layer),
                sla2_topk_ratio=sla2_topk_ratio,
                sla2_compression_ratio=sla2_compression_ratio,
                sla2_enable_qat=sla2_enable_qat,
                sla2_router_mode=sla2_router_mode,
                sla2_soft_topk_tau=sla2_soft_topk_tau,
                sla2_soft_topk_iters=sla2_soft_topk_iters,
                sla2_router_aux_weight=sla2_router_aux_weight,
            )
            for i in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch embedding
        if self.use_bottleneck:
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        else:
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj1.bias, 0)
        
        # Initialize label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        # Initialize time embeddings
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x: torch.Tensor, p: int) -> torch.Tensor:
        """Convert patch embeddings back to image"""
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_layer: int = None,
        return_last: bool = False,
    ):
        """
        Forward pass of the model.
        
        Args:
            x: (N, C, H, W) - input images
            t: (N,) - timesteps
            y: (N,) - class labels
            return_layer: Layer index to return intermediate features
            return_last: Whether to return last hidden state
        Returns:
            output: (N, C, H, W) - generated images
            feat: (N, L, D) - intermediate features if return_layer is specified
            last_out: (N, L, D) - last hidden state if return_last is True
        """
        # Embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        
        # Embed patches
        x = self.x_embedder(x)
        x = x + self.pos_embed
        
        # Forward through blocks
        for i, block in enumerate(self.blocks):
            if return_layer is not None and i == return_layer:
                if return_layer > self.in_context_start:
                    feat = x[:, self.in_context_len:]
                else:
                    feat = x
            
            # In-context tokens
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            
            x = block(
                x,
                c,
                self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            )
        
        # Remove in-context tokens
        x = x[:, self.in_context_len:]
        
        if return_last:
            last_out = x
        
        # Final layer and unpatchify
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)
        
        if return_layer is not None:
            if return_last:
                return output, feat, last_out
            else:
                return output, feat
        else:
            return output

    def pop_sla2_aux_loss(self) -> torch.Tensor:
        aux_losses = []
        for module in self.modules():
            if hasattr(module, "sla2") and hasattr(module.sla2, "pop_router_aux_loss"):
                aux = module.sla2.pop_router_aux_loss()
                if aux is not None:
                    aux_losses.append(aux)

        if not aux_losses:
            return None
        return torch.stack(aux_losses).sum()


# Convenience constructors
def JiTSLA2_B_16(**kwargs):
    return JiTSLA2(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs
    )


def JiTSLA2_L_16(**kwargs):
    return JiTSLA2(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
        **kwargs
    )


def JiTSLA2_H_16(**kwargs):
    return JiTSLA2(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
        **kwargs
    )
