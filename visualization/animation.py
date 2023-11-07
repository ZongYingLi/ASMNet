import os
from tqdm import tqdm
import numpy as np
from os.path import join as pjoin
from visualization.anim import plot_3d_motion
from einops import rearrange

src_dir = "../pretrained/ASMNet/v3/action2style"
tgt_ani_dir = "../pretrained/ASMNet/v3/animation/action2style"

if not os.path.isdir(tgt_ani_dir):
    os.makedirs(tgt_ani_dir)

npy_files = os.listdir(src_dir)
npy_files = sorted(npy_files)

for npy_file in tqdm(npy_files):
    data = np.load(pjoin(src_dir, npy_file))
    save_path = pjoin(tgt_ani_dir, npy_file[:-4] + '.gif')
    if os.path.exists(save_path):
        continue

    data = data/20
    params ={}
    params["pose_rep"] = "xyz"
    data = rearrange(data, 't j c -> j c t')
    data[:, :, :] = data[:, :, :] - data[[0], :, :]
    plot_3d_motion(data, 60, save_path, params, title=npy_file[:-4], interval=50)
    # plot_3d_motion(data, 60, save_path, params, interval=50)