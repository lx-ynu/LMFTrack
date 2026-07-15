from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import make_divisible
from timm.layers.create_act import create_act_layer
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d
from timm.models.layers import Mlp


def patch_to_tokens(feature: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = feature.shape
    return feature.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()


def tokens_to_patch(tokens: torch.Tensor) -> torch.Tensor:
    batch, num_tokens, channels = tokens.shape
    if num_tokens in (64, 128):
        height = width = 8
    elif num_tokens in (256, 512):
        height = width = 16
    else:
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Cannot reshape {num_tokens} tokens into a square feature map.")
        height = width = side
    return tokens.permute(0, 2, 1).reshape(batch, channels, height, width).contiguous()


class ConvolutionalFeedForward(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv_up = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, groups=2)
        self.bn_up = nn.BatchNorm2d(in_channels * 2, eps=1e-5)
        self.activation = nn.GELU()
        self.conv_down = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, groups=2)
        self.bn_down = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        feature = tokens.permute(0, 2, 1).reshape(batch, channels, height, width).contiguous()
        residual = self.residual(feature)
        feature = feature + self.bn1(self.conv1(feature)) + self.bn3(self.conv3(feature))
        feature = feature + self.bn_down(self.conv_down(self.activation(self.bn_up(self.conv_up(feature)))))
        return self.norm(residual + self.adjust(feature))


class LocalPerceptionUnit(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.depthwise_conv(feature)


class TopKSparseCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, bias: bool = True, topk_ratio: float = 0.5) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1].")
        self.num_heads = num_heads
        self.topk_ratio = topk_ratio
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias, groups=3)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.0)

    def forward(self, query_feature: torch.Tensor, source_feature: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = query_feature.shape
        qkv_query = self.qkv_dwconv(self.qkv(query_feature))
        qkv_source = self.qkv_dwconv(self.qkv(source_feature))
        query, _, _ = qkv_query.chunk(3, dim=1)
        _, key, value = qkv_source.chunk(3, dim=1)

        query = rearrange(query, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        key = rearrange(key, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        value = rearrange(value, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attention = (query @ key.transpose(-2, -1)) * self.temperature
        head_channels = attention.shape[-1]
        retained = max(1, int(head_channels * self.topk_ratio))
        indices = torch.topk(attention, k=retained, dim=-1, largest=True).indices
        mask = torch.zeros_like(attention, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        attention = attention.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = attention @ value
        output = rearrange(output, 'b head c (h w) -> b (head c) h w', h=height, w=width)
        return self.project_out(output)


class MMSCF(nn.Module):
    """Multi-modal Sparse Cross Fusion."""

    def __init__(self, dim: int, num_heads: int = 12, topk_ratio: float = 0.5) -> None:
        super().__init__()
        self.cfn = ConvolutionalFeedForward(dim, dim)
        self.lpu = LocalPerceptionUnit(dim)
        self.sparse_cross_attention = TopKSparseCrossAttention(dim, num_heads=num_heads, topk_ratio=topk_ratio)

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor) -> torch.Tensor:
        rgb = self.lpu(rgb)
        tir = self.lpu(tir)
        _, _, height, width = rgb.shape
        rgb_cross = self.sparse_cross_attention(rgb, tir)
        tir_cross = self.sparse_cross_attention(tir, rgb)
        merged = patch_to_tokens(rgb_cross) + patch_to_tokens(tir_cross)
        return self.cfn(merged, height, width)


class ScaleAwareSpatialAttention(nn.Module):
    def __init__(self, dim: int, group_kernel_sizes: Optional[List[int]] = None) -> None:
        super().__init__()
        kernels = group_kernel_sizes or [1, 3, 5, 7]
        if dim % 4 != 0:
            raise ValueError("The feature dimension must be divisible by four.")
        group_channels = dim // 4
        self.group_channels = group_channels
        self.local_dwc = nn.Conv1d(group_channels, group_channels, kernels[0], padding=kernels[0] // 2, groups=group_channels)
        self.global_dwc_s = nn.Conv1d(group_channels, group_channels, kernels[1], padding=kernels[1] // 2, groups=group_channels)
        self.global_dwc_m = nn.Conv1d(group_channels, group_channels, kernels[2], padding=kernels[2] // 2, groups=group_channels)
        self.global_dwc_l = nn.Conv1d(group_channels, group_channels, kernels[3], padding=kernels[3] // 2, groups=group_channels)
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.gate = nn.Sigmoid()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        feature = tokens_to_patch(tokens)
        batch, channels, height, width = feature.shape
        height_descriptor = feature.mean(dim=3)
        width_descriptor = feature.mean(dim=2)
        height_groups = torch.split(height_descriptor, self.group_channels, dim=1)
        width_groups = torch.split(width_descriptor, self.group_channels, dim=1)
        height_attention = self.gate(self.norm_h(torch.cat((
            self.local_dwc(height_groups[0]), self.global_dwc_s(height_groups[1]),
            self.global_dwc_m(height_groups[2]), self.global_dwc_l(height_groups[3]),
        ), dim=1))).view(batch, channels, height, 1)
        width_attention = self.gate(self.norm_w(torch.cat((
            self.local_dwc(width_groups[0]), self.global_dwc_s(width_groups[1]),
            self.global_dwc_m(width_groups[2]), self.global_dwc_l(width_groups[3]),
        ), dim=1))).view(batch, channels, 1, width)
        return patch_to_tokens(feature * height_attention * width_attention)


class NonLocalContextAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 8,
        gate_layer: str = 'sigmoid',
    ) -> None:
        super().__init__()
        rd_channels = make_divisible(channels * rd_ratio, 1, round_limit=0.0)
        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=nn.ReLU, norm_layer=LayerNorm2d)
        self.gate = create_act_layer(gate_layer)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.conv_attn.bias is not None:
            nn.init.zeros_(self.conv_attn.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        attention = self.conv_attn(feature).reshape(batch, 1, height * width)
        attention = F.softmax(attention, dim=-1).unsqueeze(3)
        context = feature.reshape(batch, channels, height * width).unsqueeze(1) @ attention
        context = context.view(batch, channels, 1, 1) + self.global_pool(feature)
        return patch_to_tokens(feature * self.gate(self.mlp_scale(context)))


class MCSAII(nn.Module):
    """Multi-modal Context-Scale Aware Interaction Integration."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.nlca = NonLocalContextAttention(dim)
        self.smsa = ScaleAwareSpatialAttention(dim)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor):
        batch, channels, height, width = rgb.shape
        fused_tokens = self.linear(patch_to_tokens(torch.cat((rgb, tir), dim=1)))
        nlca_tokens = self.nlca(fused_tokens.transpose(1, 2).reshape(batch, channels, height, width))
        smsa_tokens = self.smsa(fused_tokens)
        fused = (nlca_tokens + smsa_tokens).reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return rgb + fused, tir + fused


class GlobalInteraction(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 2, act_layer=nn.GELU)
        self.norm = nn.LayerNorm(dim)

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor):
        batch, _, height, width = rgb.shape
        num_tokens = height * width
        tokens = torch.cat((patch_to_tokens(rgb), patch_to_tokens(tir)), dim=1)
        tokens = tokens + self.norm(self.mlp(tokens))
        rgb_tokens, tir_tokens = torch.split(tokens, (num_tokens, num_tokens), dim=1)
        return tokens_to_patch(rgb_tokens), tokens_to_patch(tir_tokens)


class ContextSparseFusionBlock(nn.Module):
    """Apply MCSAII and MMSCF separately to template and search tokens."""

    def __init__(self, dim: int = 768, topk_ratio: float = 0.5) -> None:
        super().__init__()
        self.mcsaii = MCSAII(dim)
        self.template_mixing = GlobalInteraction(dim)
        self.search_mixing = GlobalInteraction(dim)
        self.mmscf = MMSCF(dim, num_heads=12, topk_ratio=topk_ratio)

    def forward(self, rgb_tokens: torch.Tensor, tir_tokens: torch.Tensor):
        rgb_template, rgb_search = torch.split(rgb_tokens, (64, 256), dim=1)
        tir_template, tir_search = torch.split(tir_tokens, (64, 256), dim=1)

        rgb_template = tokens_to_patch(rgb_template)
        rgb_search = tokens_to_patch(rgb_search)
        tir_template = tokens_to_patch(tir_template)
        tir_search = tokens_to_patch(tir_search)

        rgb_template_residual, tir_template_residual = rgb_template, tir_template
        rgb_search_residual, tir_search_residual = rgb_search, tir_search

        rgb_template, tir_template = self.mcsaii(rgb_template, tir_template)
        rgb_search, tir_search = self.mcsaii(rgb_search, tir_search)
        rgb_template, tir_template = self.template_mixing(rgb_template, tir_template)
        rgb_search, tir_search = self.search_mixing(rgb_search, tir_search)

        template_fusion = self.mmscf(rgb_template, tir_template)
        search_fusion = self.mmscf(rgb_search, tir_search)
        rgb_template = template_fusion + rgb_template_residual
        tir_template = template_fusion + tir_template_residual
        rgb_search = search_fusion + rgb_search_residual
        tir_search = search_fusion + tir_search_residual

        rgb_output = torch.cat((patch_to_tokens(rgb_template), patch_to_tokens(rgb_search)), dim=1)
        tir_output = torch.cat((patch_to_tokens(tir_template), patch_to_tokens(tir_search)), dim=1)
        return rgb_output, tir_output


# Backward-compatible aliases for earlier source-level imports.
cstnet_fusion = ContextSparseFusionBlock
CFM_woGIM = MCSAII
SFM = MMSCF
Top_K_Sparse_cross_Attention = TopKSparseCrossAttention
