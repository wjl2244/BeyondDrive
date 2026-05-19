import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.agents.meanfuser.utils import cumsum_traj, diff_traj, HORIZON, ACTION_DIM_DELTA
from navsim.agents.meanfuser.meanfuser_config import MeanfuserConfig


class ARMModel(nn.Module):
    def __init__(self, config: MeanfuserConfig):
        super().__init__()
        self._config = config

        self.traj_encoder = nn.Sequential(
            nn.Linear(HORIZON * ACTION_DIM_DELTA, config.tf_d_model*2, bias=False),
            nn.ReLU(),
            nn.Linear(config.tf_d_model*2, config.tf_d_model, bias=False),
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.bev_cross_attn = nn.TransformerDecoder(
            decoder_layer, 3)

        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.trajectory_recon = nn.Sequential(
            nn.Linear(config.num_proposals * config.tf_d_model, config.tf_d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.tf_d_model, config.tf_d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.tf_d_model, HORIZON * ACTION_DIM_DELTA, bias=False),
        )
        
        self.loss_fn = nn.L1Loss(reduction='mean')

        self.apply(self._init_weights)

    def _init_weights(self, module):
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
    
    def get_arm_loss(self, predictions, targets):
        pred_delta_traj = predictions["diff_trajectory"]

        dtype = pred_delta_traj.dtype
        gt_delta_trajectory = diff_traj(targets["trajectory"].to(dtype))

        arm_loss = self.loss_fn(pred_delta_traj, gt_delta_trajectory)
        return arm_loss
    
    def forward(self, outputs_dict, encoder_outputs):
        context_query = encoder_outputs.get('context_query', None)
        trajectorys = outputs_dict['pred_diff_traj']
        bs, num_proposals, num_poses, dim = trajectorys.shape

        embedded_vocab = self.traj_encoder(trajectorys.view(bs, num_proposals, -1))
        cross_attn_output = self.bev_cross_attn(embedded_vocab, context_query).contiguous()
        
        cross_attn_output = self.norm1(cross_attn_output)
        embedded_vocab = embedded_vocab + cross_attn_output
        waypoints = self.trajectory_recon(embedded_vocab.view(bs,-1)).view(bs, num_poses, -1)
        
        trajectory = cumsum_traj(waypoints)

        return {"trajectory": trajectory, 'diff_trajectory': waypoints}
