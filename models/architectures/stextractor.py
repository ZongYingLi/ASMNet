from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
from models.models_utils import Block
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, device='cuda:0', mocap_frames=60, acc_frames=150, num_joints=21, in_chans=3, acc_coords=3,
                 acc_features=18, spatial_embed=64,
                 sdepth=4, adepth=4, tdepth=4, num_heads=8, mlp_ratio=2., qkv_bias=True,
                 qk_scale=None, op_type='cls', embed_type='lin', fuse_acc_features=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, num_classes=6, dropout=0.1):

        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            mocap_frames (int): input frame number for skeletal joints
            acc_frames (int): input num frames for acc sensor
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            acc_coords(int): number of coords in one acc reading from meditag sensor: (x,y,z)=3
            spatial_embed (int): spatial patch embedding dimension
            sdepth (int): depth of spatial  transformer
            tdepth (int): depth of temporal transformer
            adepth (int): depth of acc transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            op_type(string): 'cls' or 'gap', output of temporal and acc encoder is cls token or global avg pool of encoded features.
            embed_type(string): convolutional 'conv' or linear 'lin'
            acc_features(int): number of features extracted from acc signal
            fuse_acc_features(bool): Wether to fuse acceleration feature into the acc feature or not!
            acc_coords (int) = 3(xyz) or 4(xyz, magnitude)
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm,
                                           eps=1e-6)
        temp_embed = spatial_embed * (num_joints)
        temp_frames = mocap_frames
        acc_embed = temp_embed
        self.op_type = op_type
        self.embed_type = embed_type;

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, spatial_embed)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints + 2, spatial_embed))
        self.spat_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.proj_up_clstoken = nn.Linear(mocap_frames * spatial_embed, num_joints * spatial_embed)
        self.sdepth = sdepth
        self.num_joints = num_joints
        self.joint_coords = in_chans

        # Temporal embedding
        self.Temporal_pos_embed = nn.Parameter(
            torch.zeros(1, temp_frames + 2, temp_embed))
        self.temp_frames = mocap_frames
        self.tdepth = tdepth

        # Acceleration patch and pos embeddings
        self.Acc_coords_to_embedding = nn.Linear(acc_coords, acc_embed)

        self.Acc_pos_embed = nn.Parameter(
            torch.zeros(1, acc_frames + 2, acc_embed))
        self.acc_token = nn.Parameter(torch.zeros(1, 1, acc_embed))
        self.acc_frames = acc_frames
        self.adepth = adepth
        self.acc_features = acc_features
        self.fuse_acc_features = fuse_acc_features
        self.acc_coords = acc_coords

        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]
        adpr = [x.item() for x in torch.linspace(0, drop_path_rate, adepth)]
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=temp_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed)
        self.Temporal_norm = norm_layer(temp_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.class_head = nn.Sequential(
            nn.LayerNorm(temp_embed),
            nn.Linear(temp_embed, num_classes)
        )

        self.num_classes = num_classes
        self.latent_dim = spatial_embed
        self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim*num_joints))
        self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim*num_joints))
        self.Spatial_patch_to_embedding_01 = nn.Linear(63, spatial_embed)
        self.mu_token = nn.Parameter(torch.randn(self.num_classes, 1, self.latent_dim))
        self.sigma_token = nn.Parameter(torch.randn(self.num_classes, 1, self.latent_dim))

    def Spatial_forward_features(self, batch):
        x = batch["x"]
        y = batch["y"]
        x = rearrange(x, 'b j c t -> b t j c')
        b, f, p, c = x.shape
        x = rearrange(x, 'b f p c  -> (b f) p c', )

        x = self.Spatial_patch_to_embedding(x)
        mu_token = torch.tile(self.mu_token[y], (f, 1, 1))
        logvar_token = torch.tile(self.sigma_token[y], (f, 1, 1))
        x = torch.cat((mu_token, logvar_token, x), dim=1)

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)

        Se = x.shape[-1]
        mu = torch.reshape(x[:, 0, :], (b, f * Se))
        logvar = torch.reshape(x[:, 1, :], (b, f * Se))


        temp_mu_token = self.proj_up_clstoken(mu)
        mu = torch.unsqueeze(temp_mu_token, dim=1)

        temp_logvar_token = self.proj_up_clstoken(logvar)
        logvar = torch.unsqueeze(temp_logvar_token, dim=1)

        x = x[:, :p, :]
        x = rearrange(x, '(b f) p Se-> b f (p Se)', f=f)

        xseq = torch.cat((mu, logvar, x), dim=1)

        return {"spatial_output": xseq}


    def Temp_forward_features(self, batch):
        xseq = batch["spatial_output"]

        xseq += self.Temporal_pos_embed
        xseq = self.pos_drop(xseq)

        for blk in self.Temporal_blocks:
            xseq = blk(xseq)

        xseq = self.Temporal_norm(xseq)
        mu = xseq[:, 0, :]
        logvar = xseq[:, 1, :]
        return {"temporal_output": xseq, "mu": mu, "logvar": logvar}

    def forward(self, batch):
        batch.update(self.Spatial_forward_features(batch))

        batch.update(self.Temp_forward_features(batch))

        return batch

class Decoder(nn.Module):
    def __init__(self, device='cuda:0', mocap_frames=60, acc_frames=150, num_joints=21, in_chans=3, acc_coords=3,
                 acc_features=18, spatial_embed=64,
                 sdepth=4, adepth=4, tdepth=4, num_heads=8, mlp_ratio=2., qkv_bias=True,
                 qk_scale=None, op_type='cls', embed_type='lin', fuse_acc_features=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, num_classes=6, dropout=0.1):

        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            mocap_frames (int): input frame number for skeletal joints
            acc_frames (int): input num frames for acc sensor
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            acc_coords(int): number of coords in one acc reading from meditag sensor: (x,y,z)=3
            spatial_embed (int): spatial patch embedding dimension
            sdepth (int): depth of spatial  transformer
            tdepth (int): depth of temporal transformer
            adepth (int): depth of acc transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            op_type(string): 'cls' or 'gap', output of temporal and acc encoder is cls token or global avg pool of encoded features.
            embed_type(string): convolutional 'conv' or linear 'lin'
            acc_features(int): number of features extracted from acc signal
            fuse_acc_features(bool): Wether to fuse acceleration feature into the acc feature or not!
            acc_coords (int) = 3(xyz) or 4(xyz, magnitude)
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        temp_embed = spatial_embed * (num_joints)
        temp_frames = mocap_frames
        acc_embed = temp_embed
        self.op_type = op_type
        self.embed_type = embed_type;

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, spatial_embed)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, spatial_embed))
        self.spat_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.proj_up_clstoken = nn.Linear(mocap_frames * spatial_embed, num_joints * spatial_embed)
        self.sdepth = sdepth
        self.num_joints = num_joints
        self.joint_coords = in_chans

        # Temporal embedding
        self.Temporal_pos_embed = nn.Parameter(
            torch.zeros(1, temp_frames + 1, temp_embed))
        self.temp_frames = mocap_frames
        self.tdepth = tdepth


        self.Acc_coords_to_embedding = nn.Linear(acc_coords, acc_embed)

        self.Acc_pos_embed = nn.Parameter(
            torch.zeros(1, acc_frames + 1, acc_embed))
        self.acc_token = nn.Parameter(torch.zeros(1, 1, acc_embed))
        self.acc_frames = acc_frames
        self.adepth = adepth
        self.acc_features = acc_features
        self.fuse_acc_features = fuse_acc_features
        self.acc_coords = acc_coords

        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]
        adpr = [x.item() for x in torch.linspace(0, drop_path_rate, adepth)]
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=temp_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed)
        self.Temporal_norm = norm_layer(temp_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.class_head = nn.Sequential(
            nn.LayerNorm(temp_embed),
            nn.Linear(temp_embed, num_classes)
        )

        self.num_classes = num_classes
        self.latent_dim = spatial_embed
        self.Spatial_patch_to_embedding_01 = nn.Linear(63, spatial_embed)
        self.actionBias = nn.Parameter(torch.randn(self.num_classes, self.latent_dim*num_joints))
        self.finallayer = nn.Linear(self.latent_dim*num_joints, 63)
        self.final_linear = nn.Linear(spatial_embed, in_chans)

    def Spatial_forward_features(self, batch):
        x = batch["decoder_temporal_output"][:, 1:, :]
        x = rearrange(x, 'b t (j Se) -> (b t) j Se', j=self.num_joints)

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)

        output = self.final_linear(x)

        # Reshape input
        output = rearrange(output, '(b t) j c-> b t j c', t=self.temp_frames)
        output = rearrange(output, 'b t j c -> b j c t')

        return {"output": output}

    def Temp_forward_features(self, batch):
        y = batch["y"]
        z = batch["z"]
        nframes = self.temp_frames
        bs = z.shape[0]
        latent_dim = z.shape[1]

        z = z + self.actionBias[y]
        z = z.unsqueeze(dim=1)

        timequeries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        xseq = torch.cat((z, timequeries), dim=1)

        xseq += self.Temporal_pos_embed
        xseq = self.pos_drop(xseq)

        for blk in self.Temporal_blocks:
            xseq = blk(xseq)

        xseq = self.Temporal_norm(xseq)

        return {"decoder_temporal_output": xseq}

    def forward(self, batch):
        batch.update(self.Temp_forward_features(batch))

        batch.update(self.Spatial_forward_features(batch))

        return batch