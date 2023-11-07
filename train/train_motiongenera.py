import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from trainer.trainer import train_motion
from utils.tensors import collate
from src_parser.training import parser
from models.config import Config
from models.motion_generation import Model
from data.get_dataset import get_datasets
def do_epochs(model, optimizer, writer):
    dataset = datasets["train"]
    iterator = DataLoader(dataset, batch_size=parameters["batch_size"], shuffle=True, num_workers=8, collate_fn=collate, drop_last=True)

    logpath = os.path.join(parameters["folder"], "training.log")
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"] + 1):
            dict_loss = train_motion(model, optimizer, iterator, device='cuda:0')

            for key in dict_loss.keys():
                dict_loss[key] /= len(iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"], 'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                torch.save(model.state_dict(), checkpoint_path)

            writer.flush()

if __name__ == '__main__':
    parameters = parser()
    config = Config()

    writer = SummaryWriter(log_dir=parameters["folder"])

    datasets = get_datasets(parameters)
    model = Model(parameters).to('cuda:0')

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    do_epochs(model, optimizer, writer)
    writer.close()
