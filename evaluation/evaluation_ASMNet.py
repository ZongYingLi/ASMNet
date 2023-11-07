import matplotlib.pyplot as plt
from src_parser.visualize import parser
import utils.fixseed  # noqa
plt.switch_backend('agg')
from models.model import Model
from models.config import Config
from torch.utils.data import DataLoader
from utils.tensors import collate
from itertools import cycle
from data.xia import Xia
from os.path import join as pjoin
import numpy as np
import os
import torch
idx_to_content = {0: "walk", 1: "run", 2: "jump", 3: "punch", 4: "kick", 5: "transition"}

def val():
    parameters, folder, checkpointname, epoch = parser()
    parameters["num_classes"] = 6
    parameters["nfeats"] = 3
    parameters["njoints"] = 21
    config = Config()

    dataset = Xia(datapath="../xia/", split='val')
    model = Model(config, parameters).to('cuda:0')
    path = "../pretrained/ASMNet/"
    checkpointpath = path + "checkpoint_5500.pth.tar"

    state_dict = torch.load(checkpointpath, map_location='cuda:0')
    model.load_state_dict(state_dict)

    content_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate, drop_last=True)
    style_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate, drop_last=True)

    cyc_val_style_loader = cycle(style_loader)
    cyc_val_content_loader = cycle(content_loader)

    num = 0
    while True:
        cl_data = next(cyc_val_style_loader)
        co_data = next(cyc_val_content_loader)
        batch = model.evaluate_v3(co_data["y"].to('cuda:0'), co_data["x"].to('cuda:0'), cl_data["x"].to('cuda:0'),
                                      co_data["lengths"].to('cuda:0'))
        num += 1
        print(num)

        motion2style = batch["motion2style"].view(21, 3, 60).data.cpu().numpy()
        actor2motion = batch["action2motion"].view(21, 3, 60).data.cpu().numpy()
        actor2style = batch["action2style"].view(21, 3, 60).data.cpu().numpy()

        style_name = dataset.ids_to_style_name(str(cl_data["ids"]).split("\'")[1])
        action = idx_to_content.get(int(co_data["y"]), None)
        content_id = str(int(co_data["y"]))
        style_id = str(cl_data["ids"]).split("\'")[1]

        save_path = path + "./v3"
        actor2motion_pth = pjoin(save_path, "action2motion")
        if not os.path.isdir(actor2motion_pth):
            os.makedirs(actor2motion_pth)
        motion2style_pth = pjoin(save_path, "motion2style")
        if not os.path.isdir(motion2style_pth):
            os.makedirs(motion2style_pth)
        actor2style_pth = pjoin(save_path, "action2style")
        if not os.path.isdir(actor2style_pth):
            os.makedirs(actor2style_pth)
        save_path0 = pjoin(actor2motion_pth, content_id + '_' + style_id + '_' + action + '_' + style_name + '_a2m.npy')
        save_path1 = pjoin(motion2style_pth, content_id + '_' + style_id + '_' + action + '_' + style_name + '_m2s.npy')
        save_path2 = pjoin(actor2style_pth, content_id + '_' + style_id + '_' + action + '_' + style_name + style_id + '_a2s.npy')

        np.save(save_path0, actor2motion)
        np.save(save_path1, motion2style)
        np.save(save_path2, actor2style)

        if str(cl_data["ids"]) == '[\'404_5\']':
            break

    print("finished!")
    print(num)

if __name__ == '__main__':
    val()
