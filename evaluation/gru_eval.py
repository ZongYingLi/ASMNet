import torch
from tqdm import tqdm
from utils.fixseed import fixseed
from evaluation.action2motion.evaluate import A2MEvaluation
from torch.utils.data import DataLoader
from utils.tensors import collate
import os
from evaluation.tools import save_metrics, format_metrics
from models.motion_generation import Model
from data.get_dataset import get_datasets

class NewDataloader:
    def __init__(self, mode, model, dataiterator, device):
        assert mode in ["gen", "rc", "gt"]
        self.batches = []
        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    classes = databatch["y"]
                    gendurations = databatch["lengths"]
                    batch = model.generate(classes, gendurations)
                    batch = {key: val.to(device) for key, val in batch.items()}
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    batch["x_xyz"] = model.rot2xyz(batch["x"].to(device),
                                                   batch["mask"].to(device))
                    batch["output"] = batch["x"]
                    batch["output_xyz"] = batch["x_xyz"]
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    batch["output_xyz"] = model.rot2xyz(batch["output"],
                                                        batch["mask"])
                    batch["x_xyz"] = model.rot2xyz(batch["x"],
                                                   batch["mask"])

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


def evaluate(parameters, folder, checkpointname, epoch, niter):
    num_frames = 60

    parameters["num_frames"] = num_frames
    if parameters["dataset"] == "humanact12":
        parameters["jointstype"] = "smpl"
        parameters["vertstrans"] = True
    elif parameters["dataset"] == "sxia":
        parameters["jointstype"] = "s2m"
        parameters["vertstrans"] = True
    else:
        raise NotImplementedError("Not in this file.")

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    get_datasets(parameters)
    model = Model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.outputxyz = True

    a2mevaluation = A2MEvaluation(dataname, device)
    a2mmetrics = {}

    datasetGT1 = get_datasets(parameters)["train"]
    datasetGT2 = get_datasets(parameters)["train"]

    allseeds = list(range(niter))

    try:
        for index, seed in enumerate(allseeds):
            print(f"Evaluation number: {index+1}/{niter}")
            fixseed(seed)

            dataiterator = DataLoader(datasetGT1, batch_size=parameters["batch_size"],
                                      shuffle=True, num_workers=8, collate_fn=collate)
            dataiterator2 = DataLoader(datasetGT2, batch_size=parameters["batch_size"],
                                       shuffle=True, num_workers=8, collate_fn=collate)
            motionloader = NewDataloader("gen", model, dataiterator, device)
            gt_motionloader = NewDataloader("gt", model, dataiterator, device)
            gt_motionloader2 = NewDataloader("gt", model, dataiterator2, device)

            loaders = {"gen": motionloader,
                       # "recons": reconstructedloader,
                       "gt": gt_motionloader,
                       "gt2": gt_motionloader2}

            a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
