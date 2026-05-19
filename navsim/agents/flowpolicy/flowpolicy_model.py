from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from navsim.agents.flowpolicy.flowpolicy_config import TransfuserConfig
from navsim.agents.flowpolicy.flowpolicy_backbone import ResNetBackbone
from navsim.agents.flowpolicy.modules import SimpleDiffusionTransformer, SemanticMapHead
from navsim.agents.flowpolicy.utils import diff_traj, cumsum_traj
from navsim.common.enums import StateSE2Index


class FlowPolicyModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        
        self._backbone = ResNetBackbone(config)
        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._semantic_map_head = SemanticMapHead(config)
        self._trajectory_head = ReFlowModel(config)

    def get_flowpolicy_loss(self, predictions, targets):
        results = {}
        encoder_output = predictions.pop('encoder_output')

        flowmatching_loss = self._trajectory_head.fm_loss(targets['trajectory'], encoder_output['context_query'])
        bev_semantic_loss, bev_semantic_map = self._semantic_map_head(encoder_output['bev_feature_upscale'], targets)

        loss = self._config.flowmatching_loss_weight * flowmatching_loss \
               + self._config.bev_semantic_loss_weight * bev_semantic_loss
        
        results.update({
            "loss": loss,
            "flowmatching_loss": self._config.flowmatching_loss_weight * flowmatching_loss,
            "bev_semantic_loss": self._config.bev_semantic_loss_weight * bev_semantic_loss,
        })

        return results

    def encoder(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        encoder_output = {}

        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        if isinstance(status_feature, list):
            status_feature = status_feature[-1]

        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
            if isinstance(lidar_feature, list):
                lidar_feature = lidar_feature[-1]
        
        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature).unsqueeze(-2)

        keyval = torch.concatenate([bev_feature, status_encoding], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]  # bs*65*256
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)
        trajectory_query, context_query = query_out.split(self._query_splits, dim=1)  # bs*30*256, bs*1*256
        
        encoder_output['bev_feature_upscale'] = bev_feature_upscale.clone()
        encoder_output['context_query'] = context_query.clone()
        
        return encoder_output

    def forward(self, features: Dict[str, torch.Tensor]):
        results = {}

        encoder_output = self.encoder(features)
        results['encoder_output'] = encoder_output
        
        if not self.training:
            with torch.no_grad():
                trajectorys = self._trajectory_head.generate(encoder_output['context_query'])
                results.update(trajectorys)

        return results


class ReFlowModel(nn.Module):
    def __init__(self, config):
        super(ReFlowModel, self).__init__()
        
        self._config = config
        self.P_mean = -0.8
        self.P_std = 0.8
        self.t_eps = 5e-2
        self.label_drop_prob = 0.1
        self.noise_scale = config.noise_scale
        self.cfg_scale = config.cfg_scale
        self.cfg_interval = (0.0, 1.0)
        self.steps = config.num_sampling_steps
        self.num_proposals = config.num_sampling_proposals
        self.num_poses = config.trajectory_sampling.num_poses

        self.dit_model = SimpleDiffusionTransformer(config)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def drop_labels(self, context):
        drop = torch.rand(context.shape[0], device=context.device) < self.label_drop_prob
        drop = drop.view(-1, *([1] * (context.ndim - 1)))
        out = torch.where(drop, torch.full_like(context, 0), context)
        return out

    def fm_loss(self, x, context):
        x = diff_traj(x.to(context.dtype))
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        context_dropped = self.drop_labels(context) if self.training else context

        z = t * x + (1 - t) * e
        
        if self._config.flow_pred_x:
            # jit
            v = (x - z) / (1 - t).clamp_min(self.t_eps)
            x_pred = self.dit_model(z, t, context_dropped)
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        else:
            # reflow
            v = x - e
            v_pred = self.dit_model(z, t, context_dropped)
        
        # loss = nn.functional.mse_loss(v, v_pred)
        loss = nn.functional.l1_loss(v, v_pred)

        return loss

    @torch.no_grad()
    def generate(self, context):
        device = context.device
        bs = context.shape[0]
        context_repeat = context.repeat_interleave(self.num_proposals, dim=0)
        z = self.noise_scale * torch.randn(context_repeat.size(0), self.num_poses, 4, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bs, -1, -1)
        timesteps = timesteps.repeat_interleave(self.num_proposals, dim=1)

        # ode
        for i in range(self.steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = self._euler_step(z, t, t_next, context_repeat)
        
        traj = cumsum_traj(z)
        sampled_trajectories = traj.view(bs, self.num_proposals, self.num_poses, -1)

        outputs = {
            "pred_trajectorys": sampled_trajectories,
            "trajectory": sampled_trajectories.mean(dim=1),
        }
        
        return outputs

    @torch.no_grad()
    def _forward_sample(self, z, t, context):
        # conditional
        if self._config.flow_pred_x:
            # jit
            x_cond_pred = self.dit_model(z, t, context)
            v_cond_pred = (x_cond_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        else:
            # reflow
            v_cond_pred = self.dit_model(z, t, context)

        # unconditional
        if self._config.flow_pred_x:
            x_uncond_pred = self.dit_model(z, t, torch.full_like(context, 0))
            v_uncond_pred = (x_uncond_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        else:
            v_uncond_pred = self.dit_model(z, t, torch.full_like(context, 0))

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        v_pred = v_uncond_pred + cfg_scale_interval * (v_cond_pred - v_uncond_pred)
        
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, context):
        v_pred = self._forward_sample(z, t, context)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, context):
        v_pred_t = self._forward_sample(z, t, context)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, context)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next
