import os
from src_parser.base import add_misc_options, add_cuda_options, adding_cuda, ArgumentParser
from src_parser.tools import save_args
from src_parser.dataset import add_dataset_options
from src_parser.model import add_model_options, parse_modelname
from src_parser.checkpoint import construct_checkpointname

def add_training_options(parser):
    group = parser.add_argument_group('Training options')
    group.add_argument("--batch_size", type=int, required=True, default=20, help="size of the batches")
    group.add_argument("--num_epochs", type=int, required=True, default=200, help="number of epochs of training")
    group.add_argument("--lr", type=float, required=True, default=0.0001, help="AdamW: learning rate")
    group.add_argument("--snapshot", type=int, required=True, default=500, help="frequency of saving model/viz")

def parser():
    parser = ArgumentParser()

    add_misc_options(parser)

    add_cuda_options(parser)

    add_training_options(parser)

    add_dataset_options(parser)

    add_model_options(parser)

    opt = parser.parse_args()

    parameters = {key: val for key, val in vars(opt).items() if val is not None}

    parameters["losses"] = parameters["losses"].split("_")

    lambdas = {}
    for loss in parameters["losses"]:
        lambdas[loss] = opt.__getattribute__(f"lambda_{loss}")
    parameters["lambdas"] = lambdas
    
    if "folder" not in parameters:
        parameters["folder"] = construct_checkpointname(parameters, parameters["expname"])

    os.makedirs(parameters["folder"], exist_ok=True)
    save_args(parameters, folder=parameters["folder"])

    adding_cuda(parameters)
    
    return parameters
