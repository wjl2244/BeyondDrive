import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.agents.flowpolicy.utils import RMSNorm, Attention, SwiGLUFFN


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


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.timestep_embedding = SinusoidalPosEmb(hidden_size)

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        # 1. use TransformerDecoder
        self.blocks = nn.ModuleList([nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config.tf_d_model, 
                config.tf_num_head, 
                config.tf_d_ffn,
                config.tf_dropout, 
                batch_first=True), 
            1
        )])
        input_dim = config.trajectory_sampling.num_poses * 4
        token_len = 31

        self.input_emb = nn.Linear(input_dim, config.tf_d_model)
        self.time_t_emb = SinusoidalPosEmb(config.tf_d_model)
        
        self.layer_norm = nn.LayerNorm(config.tf_d_model)
        self.output_emb = nn.Linear(config.tf_d_model, input_dim)
        
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, config.tf_d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, config.tf_d_model))
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
        elif isinstance(module, TimestepEmbedder):
            for submodule in module.mlp._modules.values():
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

    def forward(self, sample, timesteps_t, context):
        B, HORIZON, DIM = sample.shape
        sample = sample.view(B, -1).float()

        input_emb = self.input_emb(sample).unsqueeze(1)
        time_emb = self.time_t_emb(timesteps_t.squeeze(1))
        
        cond_embeddings = time_emb
        cond_embeddings = torch.cat([time_emb, context], dim=1)
        
        position_embeddings = self.cond_pos_emb[:, :cond_embeddings.shape[1], :]  # each position maps to a (learnable) vector
        c = cond_embeddings + position_embeddings

        # decoder_emb
        position_embeddings = self.pos_emb[:, :input_emb.shape[1], :]  # each position maps to a (learnable) vector
        x = input_emb + position_embeddings
        
        for block in self.blocks:
            if isinstance(block, nn.MultiheadAttention):
                x = block(x, c, c)[0]
            elif isinstance(block, nn.TransformerDecoder):
                x = block(tgt=x, memory=c)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
        
        x = self.layer_norm(x)
        x = self.output_emb(x)
        x = x.squeeze(1).view(B, HORIZON, DIM)

        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    # @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(self.modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp * self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTModel(nn.Module):
    def __init__(self, config):
        super(DiTModel, self).__init__()

        self._config = config
        self.t_embedder = SinusoidalPosEmb(config.tf_d_model)

        input_dim = config.trajectory_sampling.num_poses * 4
        self.x_embedder = nn.Sequential(
            nn.Linear(input_dim, config.tf_d_model),
            nn.SiLU(),
            nn.Linear(config.tf_d_model, config.tf_d_model),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, config.tf_d_model), requires_grad=True)
        self.blocks = nn.ModuleList([
            JiTBlock(config.tf_d_model, config.tf_num_head, mlp_ratio=2,
                     attn_drop=config.tf_dropout,
                     proj_drop=0.0)
            for i in range(config.tf_num_layers)
        ])

        self.final_layer = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_model),
            nn.SiLU(),
            nn.Linear(config.tf_d_model, input_dim),
        )

    def forward(self, x, t, context):
        """
        x: bs*num_poses*4
        t: bs*1*1
        context: bs*1*256
        """
        BS, HORIZON, DIM = x.shape
        t_emb = self.t_embedder(t.squeeze(1))
        # c = torch.concat([t_emb, context], dim=-2)
        c = t_emb + context

        x = x.view(BS, -1).unsqueeze(1)
        x = self.x_embedder(x)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            x = block(x, c)
        x = self.final_layer(x).squeeze(1)
        x = x.view(BS, HORIZON, DIM)
        return x


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