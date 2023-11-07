import torch
import torch.nn as nn
from models.modeltype.styleencoder import EncoderStyle
from models.modeltype.mlp import MLP
from models.modeltype import decoder as dec
from models.rotation2xyz import Rotation2xyz
from models.tools.losses import compute_rcxyz_loss, compute_kl_loss, compute_recon_loss
from models.architectures.stextractor import Encoder, Decoder
from models.architectures.gru import Encoder_GRU, Decoder_GRU
from models.architectures.fc import Encoder_FC, Decoder_FC

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features: 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features:]

class Model(nn.Module):
    def __init__(self, config, parameters):
        super(Model, self).__init__()

        parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
        self.encoder = Encoder(spatial_embed=parameters["latent_dim"])
        self.decoder = Decoder(spatial_embed=parameters["latent_dim"])
        # self.encoder = Encoder_TRANSFORMER(**parameters)
        # self.decoder = Decoder_TRANSFORMER(**parameters)
        # self.encoder = Encoder_GRU(**parameters)
        # self.decoder = Decoder_GRU(**parameters)
        # self.encoder = Encoder_FC(**parameters)
        # self.decoder = Decoder_FC(**parameters)
        self.style_encoder = EncoderStyle(config)
        self.style_decoder = dec.Decoder(config)
        self.mlp = MLP(config, get_num_adain_params(self.style_decoder))

        self.num_frames = parameters["num_frames"]
        self.lambdas = parameters['lambdas']
        self.w_rcxyz = 1.0
        self.w_rc = 1.5
        self.w_kl = 1e-5
        self.w_style = 0.3
        self.losses = list(self.lambdas) + ["mixed"]

        self.outputxyz = parameters['outputxyz']

        self.batch_size = parameters["batch_size"]
        self.latent_dim = parameters["njoints"]*parameters["latent_dim"]
        self.pose_rep = parameters['pose_rep']
        self.glob = parameters['glob']
        self.glob_rot = parameters['glob_rot']
        self.device = 'cuda:0'
        self.translation = parameters['translation']
        self.jointstype = parameters['jointstype']
        self.vertstrans = parameters['vertstrans']

        self.losses = ["rcxyz_loss", "recon_style", "recon_style_features", "kl_loss", "mixed"]

        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}

    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def decode_rot(self, content, model_code):
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.style_decoder)
        position = self.style_decoder(content)
        return position

    def decode(self, content, model_code):
        content = content.transpose(1, 2)
        content = content.reshape([content.shape[0], 63, content.shape[3]])
        position = self.decode_rot(content, model_code)
        position = position.reshape([position.shape[0], 21, 3, self.num_frames])
        return position

    def compute_loss(self, content_batch, style_batch):
        rcxyz = compute_rcxyz_loss(content_batch["x"], content_batch["output_xyz"], content_batch["mask"])

        rcxyz_content = compute_recon_loss(content_batch["x"], content_batch["rxr_content"])
        rcxyz_content_same = compute_recon_loss(content_batch["x"], content_batch["rxr_same_content"])

        rcxyz_style = compute_recon_loss(style_batch["x"], style_batch["rxr_style"])
        rcxyz_style_same = compute_recon_loss(style_batch["x"], style_batch["rxr_same_style"])
        recon_style = 0.25*(rcxyz_content + rcxyz_content_same + rcxyz_style + rcxyz_style_same)

        recon_cl_style = compute_recon_loss(style_batch["encoded_style"], style_batch["encoded_cl_same"])
        recon_co_style = compute_recon_loss(content_batch["encoded_content"], content_batch["encoded_co_same"])
        recon_style_features = 0.5*(recon_co_style + recon_cl_style)

        # kl
        kl_content_loss = compute_kl_loss(content_batch)

        mixed_loss = self.w_rc*rcxyz + \
                     self.w_rcxyz * recon_style + \
                     self.w_kl * kl_content_loss + \
                     self.w_style * recon_style_features

        losses = {
            "rcxyz_loss": rcxyz.item(),
            "recon_style": recon_style.item(),
            "recon_style_features":  recon_style_features.item(),
            "kl_loss": kl_content_loss.item(),
            "mixed": mixed_loss.item()
        }

        return mixed_loss, losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def forward(self, content_batch, style_batch):

        content_batch["x_xyz"] = content_batch["x"]

        # encode
        content_batch.update(self.encoder(content_batch))
        content_batch["z"] = self.reparameterize(content_batch)

        # decode
        content_batch.update(self.decoder(content_batch))

        # if we want to output xyz
        content_batch["output_xyz"] = content_batch["output"]

        encoded_style = self.style_encoder(style_batch["x"])
        encoded_cl_same = self.style_encoder(style_batch["same_style"])

        content_style = self.style_encoder(content_batch["x"])
        encoded_co_same = self.style_encoder(content_batch["same_style"])

        rxr_content_target = self.decode(content_batch["x"], encoded_style)
        rxr_target = self.decode(content_batch["output"], encoded_style)
        rxr_style = self.decode(style_batch["x"], encoded_style)
        rxr_content = self.decode(content_batch["x"], content_style)
        rxr_same_style = self.decode(style_batch["x"], encoded_cl_same)
        rxr_same_content = self.decode(content_batch["x"], encoded_co_same)

        style_batch["encoded_style"] = encoded_style
        style_batch["encoded_cl_same"] = encoded_cl_same
        content_batch["encoded_content"] = content_style
        content_batch["encoded_co_same"] = encoded_co_same

        style_batch["rxr_content_target"] = rxr_content_target
        style_batch["rxr_target"] = rxr_target
        style_batch["rxr_style"] = rxr_style
        content_batch["rxr_content"] = rxr_content
        style_batch["rxr_same_style"] = rxr_same_style
        content_batch["rxr_same_content"] = rxr_same_content

        return content_batch, style_batch

    def generate(self, classes,
                 durations,
                 nspa=1,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1):
        if nspa is None:
            nspa = 1
        nats = len(classes)

        y = classes.to(self.device).repeat(nspa)

        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(y.shape)

        mask = self.lengths_to_mask(lengths)

        if noise_same_action == "random":
            if noise_diff_action == "random":
                z = torch.randn(nspa * nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_same_action = torch.randn(nspa, self.latent_dim, device=self.device)
                z = z_same_action.repeat_interleave(nats, axis=0)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
        elif noise_same_action == "interpolate":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            interpolation_factors = torch.linspace(-1, 1, nspa, device=self.device)
            z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa * nats, -1)
        elif noise_same_action == "same":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            z = z_diff_action.repeat((nspa, 1))
        else:
            raise NotImplementedError("Noise same action must be random, same or interpolate.")

        batch = {"z": fact * z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        batch["output_xyz"] = batch["output"]

        return batch

    def evaluate(self, content_batch, style_batch, nspa=1):
        fact = 1
        if nspa is None:
            nspa = 1

        lengths = content_batch["lengths"].repeat((nspa, 1))
        y = content_batch["y"].repeat(nspa)
        if len(lengths.shape) == 1:
            lengths = lengths.to(self.device).repeat(nspa)
        else:
            lengths = lengths.to(self.device).reshape(y.shape)
        mask = self.lengths_to_mask(lengths)

        z = torch.randn(len(y), self.latent_dim, device=self.device)
        batch = {"z": fact * z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)
        batch["output_xyz"] = batch["output"]
        encoded_style = self.style_encoder(style_batch["x"])
        encoded_style_repeated = encoded_style.repeat(nspa, 1, 1)
        target = self.decode(batch["output"], encoded_style_repeated)
        batch["rxr_target"] = target
        return batch

    def evaluate_v2(self, content_batch, style_batch):
        fact = 1

        lengths = content_batch["lengths"]
        y = content_batch["y"]
        mask = self.lengths_to_mask(lengths)

        z = torch.randn(len(y), self.latent_dim, device=self.device)
        batch = {"z": fact * z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)
        batch["output_xyz"] = batch["output"]
        encoded_style = self.style_encoder(style_batch["x"])
        target = self.decode(content_batch["x"], encoded_style)
        batch["rxr_target"] = target
        return batch

    def evaluate_v3(self, a, content, style, lengths):
        fact = 1
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(len(a), self.latent_dim, device=self.device)
        batch = {"z": fact * z, "y": a, "mask": mask, "lengths": lengths}

        batch = self.decoder(batch)

        batch["action2motion"] = batch["output"]

        encoded_style = self.style_encoder(style)
        content_style = self.decode(content, encoded_style)
        batch["motion2style"] = content_style

        actor2style = self.decode(batch["output"], encoded_style)
        batch["action2style"] = actor2style

        return batch

    def evaluate_style(self, content, style):
        batch = {}
        encoded_style = self.style_encoder(style)
        content_style = self.decode(content, encoded_style)
        batch["motion2style"] = content_style
        return batch

    def evaluate_motion(self, a, lengths):
        fact = 1
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(len(a), self.latent_dim, device=self.device)
        batch = {"z": fact * z, "y": a, "mask": mask, "lengths": lengths}

        batch = self.decoder(batch)
        batch["actor2motion"] = batch["output"]

        return batch


