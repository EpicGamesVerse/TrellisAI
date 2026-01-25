from __future__ import annotations

from typing import Literal, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def _linear_sparse(module: nn.Linear, x: SparseTensor) -> SparseTensor:
        return x.replace(module(x.feats))

    @staticmethod
    def _linear_dense(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return module(x)

    @staticmethod
    def _reshape_chs(x: SparseTensor, shape: Tuple[int, ...]) -> SparseTensor:
        return x.reshape(*shape)

    def _fused_pre(self, x: SparseTensor, num_fused: int) -> SparseTensor:
        x_feats = x.feats.unsqueeze(0)
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0))

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
        return qkv
    
    def forward(self, x: SparseTensor, context: Optional[Union[SparseTensor, torch.Tensor]] = None) -> SparseTensor:
        if self._type == "self":
            qkv = self._linear_sparse(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = cast(SparseTensor, self.q_rms_norm(q))
                k = cast(SparseTensor, self.k_rms_norm(k))
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                if self.window_size is None:
                    raise ValueError("window_size must be set for serialized attention")
                window_size = self.window_size
                serialize_mode = self.serialize_mode or SerializeMode.Z_ORDER
                shift_sequence = self.shift_sequence or 0
                shift_window = self.shift_window or (0, 0, 0)
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv,
                    window_size,
                    serialize_mode=serialize_mode,
                    shift_sequence=shift_sequence,
                    shift_window=shift_window,
                )
            elif self.attn_mode == "windowed":
                if self.window_size is None:
                    raise ValueError("window_size must be set for windowed attention")
                window_size = self.window_size
                shift_window = self.shift_window or (0, 0, 0)
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv,
                    window_size,
                    shift_window=shift_window,
                )
        else:
            q = self._linear_sparse(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            if context is None:
                raise ValueError("context must be provided for cross-attention")
            if isinstance(context, SparseTensor):
                kv = self._linear_sparse(self.to_kv, context)
                kv = self._fused_pre(kv, num_fused=2)
                if self.qk_rms_norm:
                    q = cast(SparseTensor, self.q_rms_norm(q))
                    k, v = kv.unbind(dim=1)
                    k = cast(SparseTensor, self.k_rms_norm(k))
                    kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
                h = sparse_scaled_dot_product_attention(q, kv)
            else:
                context_t = cast(torch.Tensor, context)
                kv_dense = self._linear_dense(self.to_kv, context_t)
                # [N, L, 2*C] -> [N, L, 2, H, C_head]
                N, L, _ = kv_dense.shape
                kv_dense = kv_dense.view(N, L, 2, self.num_heads, -1)
                if self.qk_rms_norm:
                    q = cast(SparseTensor, self.q_rms_norm(q))
                    k_dense = kv_dense[:, :, 0]
                    v_dense = kv_dense[:, :, 1]
                    k_dense = cast(torch.Tensor, self.k_rms_norm(k_dense))
                    kv_dense = torch.stack([k_dense, v_dense], dim=2)
                h = sparse_scaled_dot_product_attention(q, kv_dense)
        h = self._reshape_chs(h, (-1,))
        h = self._linear_sparse(self.to_out, h)
        return h
