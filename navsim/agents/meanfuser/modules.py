import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Attention, Mlp
from navsim.agents.meanfuser.utils import BoundingBox2DIndex


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dp_nlayers, input_dim, obs_len):
        super().__init__()

        # 1. use TransformerDecoder
        self.blocks = nn.ModuleList([nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, d_ffn,dropout=0.0, batch_first=True), 
            dp_nlayers
        )])

        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_t_emb = SinusoidalPosEmb(d_model)
        self.time_r_emb = SinusoidalPosEmb(d_model)
        token_len = obs_len + 1

        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        # nn.ModuleList,
                        nn.SiLU,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ModuleList):
            for submodule in module:
                self._init_weights(submodule)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SimpleDiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self,
                sample,
                time_r,
                time_t,
                bev_query=None, 
                context_query=None,
                training=True,
                ):
        B, HORIZON, DIM = sample.shape
        sample = sample.view(B, -1).float()
        input_emb = self.input_emb(sample)

        timesteps_t = time_t
        timesteps_r = time_r

        if not torch.is_tensor(timesteps_t):
            timesteps_t = torch.tensor([timesteps_t], dtype=torch.long, device=sample.device)
            timesteps_r = torch.tensor([timesteps_r], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps_t) and len(timesteps_t.shape) == 0:
            timesteps_t = timesteps_t[None].to(sample.device)
            timesteps_r = timesteps_r[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps_t = timesteps_t.expand(sample.shape[0])
        timesteps_r = timesteps_r.expand(sample.shape[0])

        time_t_emb = self.time_t_emb(timesteps_t).unsqueeze(1)
        time_r_emb = self.time_r_emb(timesteps_t - timesteps_r).unsqueeze(1)

        time_emb = time_t_emb + time_r_emb
        # time_emb = torch.cat([time_t_emb, time_r_emb], dim=1)
        
        cond_embeddings = time_emb
        if bev_query is not None:
            cond_embeddings = torch.cat([time_emb, bev_query], dim=1)
        if context_query is not None:
            cond_embeddings = torch.cat([cond_embeddings, context_query], dim=1)
        
        position_embeddings = self.cond_pos_emb[:, :cond_embeddings.shape[1], :]  # each position maps to a (learnable) vector
        c = cond_embeddings + position_embeddings

        # decoder
        token_embeddings = input_emb.unsqueeze(1)
        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        
        down_latents = []
        for block in self.blocks:
            if isinstance(block, nn.MultiheadAttention):
                x = block(x, c, c)[0]
            elif isinstance(block, nn.TransformerDecoder):
                x = block(tgt=x, memory=c)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
            down_latents.append(x)
        
        x = self.ln_f(x)
        x = self.output_emb(x)
        velocity_pred = x.squeeze(1).view(B, HORIZON, DIM)

        if training:
            up_features_tensor = torch.stack([f.flatten(1) for f in down_latents], dim=0)
            return velocity_pred, up_features_tensor
        else:
            return velocity_pred


class SemanticMapHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

    def forward(self, bev_feature, targets):
        bev_semantic_map = self._bev_semantic_head(bev_feature)
        
        # compute loss
        bev_semantic_loss = F.cross_entropy(bev_semantic_map, targets["bev_semantic_map"].long())

        return bev_semantic_loss, bev_semantic_map
