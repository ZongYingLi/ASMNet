import os

import matplotlib.pyplot as plt
import torch
from src_parser.visualize import parser
from visualization.visualize import viz_epoch
from data.get_dataset import get_datasets
import utils.fixseed  # noqa

plt.switch_backend('agg')
from models.model import Model
from models.config import Config
def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    config = Config()

    datasets = get_datasets(parameters)
    model = Model(config, parameters).to('cuda:0')
    dataset = datasets["train"]

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)

    folder_actor = '../pretrained/motiongenera/humanact12'
    checkpointname_actor = 'checkpoint_4000.pth.tar'
    checkpointpath_actor = os.path.join(folder_actor, checkpointname_actor)
    state_dict_actor = torch.load(checkpointpath_actor, map_location=parameters["device"])
    model.cave.load_state_dict(state_dict_actor)

    viz_epoch(model, dataset, epoch, parameters, folder=folder, writer=None)

if __name__ == '__main__':
    main()
