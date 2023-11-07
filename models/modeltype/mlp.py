from torch import nn
from models.blocks import LinearBlock

class MLP(nn.Module):
    def __init__(self, config, out_dim):
        super(MLP, self).__init__()
        dims = config.mlp_dims      # 144*192*256
        n_blk = len(dims)       # 3
        norm = 'none'
        acti = 'lrelu'

        layers = []
        for i in range(n_blk - 1):
            layers += LinearBlock(dims[i], dims[i + 1], norm=norm, acti=acti)
        layers += LinearBlock(dims[-1], out_dim, norm='none', acti='none')
        self.model = nn.Sequential(*layers)

    def forward(self, x):        # 128*96
        return self.model(x.view(x.size(0), -1))