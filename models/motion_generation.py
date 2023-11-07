import torch
import torch.nn as nn
from models.rotation2xyz import Rotation2xyz
from models.tools.losses import compute_rcxyz_loss, compute_kl_loss
from models.architectures.stextractor import Encoder, Decoder
from models.architectures.gru import Encoder_GRU, Decoder_GRU
from models.architectures.fc import Encoder_FC, Decoder_FC

class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()

        parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
        # st extractor
        self.encoder = Encoder(spatial_embed=parameters["latent_dim"])
        self.decoder = Decoder(spatial_embed=parameters["latent_dim"])
        # self.encoder = Encoder_TRANSFORMER(**parameters)
        # self.decoder = Decoder_TRANSFORMER(**parameters)
        # self.encoder = Encoder_TRANSFORMER(**parameters)
        # self.decoder = Decoder_TRANSFORMER(**parameters)
        # self.encoder = Encoder_GRU(**parameters)
        # self.decoder = Decoder_GRU(**parameters)
        # self.encoder = Encoder_FC(**parameters)
        # self.decoder = Decoder_FC(**parameters)

        self.num_frames = parameters["num_frames"]
        self.lambdas = parameters['lambdas']
        self.w_rcxyz = 1.0
        self.w_rc = 1.5
        self.w_kl = 1e-5
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

        self.losses = ["rcxyz_loss", "kl_loss", "mixed"]

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

    def compute_loss(self, batch):
        rcxyz = compute_rcxyz_loss(batch["x"], batch["output_xyz"], batch["mask"])

        # kl
        kl_content_loss = compute_kl_loss(batch)

        mixed_loss = self.w_rc*rcxyz + \
                     self.w_kl * kl_content_loss

        losses = {
            "rcxyz_loss": rcxyz.item(),
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

    def forward(self, batch):
        batch["x_xyz"] = batch["x"]

        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)

        # decode
        batch.update(self.decoder(batch))

        # if we want to output xyz
        batch["output_xyz"] = batch["output"]

        return batch

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

        batch["output_xyz"] = batch["output"]   # [bs, 21, 3, 80]

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

    def evaluate_motion(self, a, lengths):
        fact = 1
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(len(a), self.latent_dim, device=self.device)
        batch = {"z": fact * z, "y": a, "mask": mask, "lengths": lengths}

        batch = self.decoder(batch)

        batch["actor2motion"] = batch["output"]

        return batch


