# mlx contentvec / hubert base

from __future__ import annotations

import math
from typing import Dict

import mlx.core as mx
import mlx.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int, has_gn: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=stride, bias=False)
        self.has_gn = has_gn
        if has_gn:
            # groupnorm with groups == c_out
            self.gn_weight = mx.ones((c_out,), dtype=mx.float32)
            self.gn_bias = mx.zeros((c_out,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.has_gn:
            orig_dtype = x.dtype
            xf = x.astype(mx.float32)
            mean = mx.mean(xf, axis=1, keepdims=True)
            var = mx.var(xf, axis=1, keepdims=True)
            xf = (xf - mean) * mx.rsqrt(var + 1e-5)
            xf = xf * self.gn_weight + self.gn_bias
            x = xf.astype(orig_dtype)
        return nn.gelu(x)


class _TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int = 768, ffn_dim: int = 3072, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        residual = x
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)
        x = self.out_proj(attn_out)
        x = self.self_attn_layer_norm((residual + x).astype(mx.float32)).astype(x.dtype)

        residual = x
        x = self.fc2(nn.gelu(self.fc1(x)))
        x = self.final_layer_norm((residual + x).astype(mx.float32)).astype(x.dtype)
        return x
    
        
    
    
    
class ContentVecMLX(nn.Module):
    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.compute_dtype = dtype
        conv_spec = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        self.conv_layers = []
        c_in = 1
        for i, (c_out, k, s) in enumerate(conv_spec):
            self.conv_layers.append(_ConvBlock(c_in, c_out, k, s, has_gn=(i == 0)))
            c_in = c_out

        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 768, bias=True)
        self.pos_conv = nn.Conv1d(768, 768, kernel_size=128, padding=64, groups=16, bias=True)
        self.enc_layer_norm = nn.LayerNorm(768)
        self.layers = [_TransformerLayer(768, 3072, 12) for _ in range(12)]

    @classmethod
    def from_safetensors(
        cls, path: str, use_fp16: bool = True, quantize_8bit: bool = False
    ) -> "ContentVecMLX":
        dtype = mx.float16 if use_fp16 else mx.float32
        model = cls(dtype=dtype)
        raw: Dict[str, mx.array] = mx.load(path)
        
        for i, blk in enumerate(model.conv_layers):
            blk.conv.weight = raw[f"feature_extractor.conv_layers.{i}.conv.weight"].astype(dtype)
            if blk.has_gn:
                blk.gn_weight = raw[f"feature_extractor.conv_layers.{i}.norm.weight"].astype(mx.float32)
                blk.gn_bias = raw[f"feature_extractor.conv_layers.{i}.norm.bias"].astype(mx.float32)

        model.layer_norm.weight = raw["layer_norm.weight"].astype(mx.float32)
        model.layer_norm.bias = raw["layer_norm.bias"].astype(mx.float32)
        model.post_extract_proj.weight = raw["post_extract_proj.weight"].astype(dtype)
        model.post_extract_proj.bias = raw["post_extract_proj.bias"].astype(dtype)

        # bake pos_conv weight norm
        g = raw["encoder.pos_conv.layers.0.g"].astype(mx.float32)
        v = raw["encoder.pos_conv.layers.0.v"].astype(mx.float32)
        v_norm = mx.maximum(mx.sqrt(mx.sum(v * v, axis=(0, 2), keepdims=True)), 1e-5)
        w_pos = g * (v / v_norm)
        model.pos_conv.weight = w_pos.astype(dtype)
        model.pos_conv.bias = raw["encoder.pos_conv.layers.0.module.bias"].astype(dtype)

        model.enc_layer_norm.weight = raw["encoder.layer_norm.weight"].astype(mx.float32)
        model.enc_layer_norm.bias = raw["encoder.layer_norm.bias"].astype(mx.float32)

        # fuse q/k/v into one linear per layer
        for i, layer in enumerate(model.layers):
            p = f"encoder.layers.{i}"
            qw = raw[f"{p}.self_attn.q_proj.weight"]
            kw = raw[f"{p}.self_attn.k_proj.weight"]
            vw = raw[f"{p}.self_attn.v_proj.weight"]
            qb = raw[f"{p}.self_attn.q_proj.bias"]
            kb = raw[f"{p}.self_attn.k_proj.bias"]
            vb = raw[f"{p}.self_attn.v_proj.bias"]

            layer.qkv_proj.weight = mx.concatenate([qw, kw, vw], axis=0).astype(dtype)
            layer.qkv_proj.bias = mx.concatenate([qb, kb, vb], axis=0).astype(dtype)
            layer.out_proj.weight = raw[f"{p}.self_attn.out_proj.weight"].astype(dtype)
            layer.out_proj.bias = raw[f"{p}.self_attn.out_proj.bias"].astype(dtype)

            layer.self_attn_layer_norm.weight = raw[f"{p}.self_attn_layer_norm.weight"].astype(mx.float32)
            layer.self_attn_layer_norm.bias = raw[f"{p}.self_attn_layer_norm.bias"].astype(mx.float32)
            layer.fc1.weight = raw[f"{p}.fc1.weight"].astype(dtype)
            layer.fc1.bias = raw[f"{p}.fc1.bias"].astype(dtype)
            layer.fc2.weight = raw[f"{p}.fc2.weight"].astype(dtype)
            layer.fc2.bias = raw[f"{p}.fc2.bias"].astype(dtype)
            layer.final_layer_norm.weight = raw[f"{p}.final_layer_norm.weight"].astype(mx.float32)
            layer.final_layer_norm.bias = raw[f"{p}.final_layer_norm.bias"].astype(mx.float32)

        mx.eval(model.parameters())
        del raw
        if quantize_8bit:
            nn.quantize(
                model,
                group_size=64,
                bits=8,
                class_predicate=lambda _p, m: isinstance(m, nn.Linear),
            )
            mx.eval(model.parameters())
        mx.clear_cache()
        model.eval()
        return model

    def __call__(self, audio_16k: mx.array, output_layer: int = 12) -> mx.array:
        x = mx.expand_dims(audio_16k, axis=-1).astype(self.compute_dtype)
        for blk in self.conv_layers:
            x = blk(x)
        x = self.layer_norm(x.astype(mx.float32)).astype(self.compute_dtype)
        x = self.post_extract_proj(x)
        x = x + nn.gelu(self.pos_conv(x)[:, :-1, :])
        x = self.enc_layer_norm(x.astype(mx.float32)).astype(self.compute_dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 == output_layer:
                break
        return x
