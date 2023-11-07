import logging
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from rich.progress import track
from utils.tensors import collate
logger = logging.getLogger(__name__)
import random

def get_split_keyids(path: str, split: str):
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")

def load_label(keyid, datapath):
    metapath = datapath / 'action_label' / (keyid + ".txt")
    with open(metapath, 'r') as f:
        label = f.readline().strip()
    return label

style_idx_to_name = {0: 'angry', 1: 'childlike', 2: 'depressed', 3: 'neutral', 4: 'old', 5: 'proud', 6: 'sexy', 7: 'strutting'}
class Xia(Dataset):
    dataname = "xia"

    def __init__(self,
                 datapath: str,
                 split: str,
                 min_motion_len: int = 24,
                 progress_bar: bool = True):
        super().__init__()
        self.num_frames = 60
        self.sampling = "conseq"
        self.sampling_step = 1
        self.split = split
        self.ids = []
        self.same_style = []

        self.labels = []
        self._joints = []
        self._actions = []
        self._num_frames_in_video = []

        keyids = get_split_keyids(path=datapath, split=split + '.txt')

        if progress_bar:
            enumerator = enumerate(track(keyids, f"Loading Xia {split}"))
        else:
            enumerator = enumerate(keyids)

        datapath = Path(datapath)
        for i, keyid in enumerator:
            motionpath = datapath / 'new_joints' / (keyid + '.npy')
            data = np.load(motionpath)
            style_id = keyid.split('_')[1]

            if (len(data)) < min_motion_len or (len(data) >= 400):
                continue
            label = load_label(keyid, datapath)
            label = int(label) - 1

            # get same_style
            next_keyid = keyids[i + 1] if i + 1 < len(keyids) else None
            while next_keyid:
                if next_keyid.split('_')[1] == str(style_id):
                    same_style_keyid = next_keyid
                    same_style = np.load(datapath / 'new_joints' / (same_style_keyid + '.npy'))
                    pose_style = same_style
                    break
                elif keyids.index(next_keyid) != len(keyids) - 1:
                    next_keyid = keyids[keyids.index(next_keyid) + 1]
                else:
                    pose_style = data
                    break
            pose_style = pose_style.reshape((-1, 21 * 3))

            self.ids.append(keyid)
            self.same_style.append(pose_style)
            if label not in self.labels:
                self.labels.append(label)

            self._joints.append(data)
            self._actions.append(label)
            self._num_frames_in_video.append(data.shape[0])

        total_num_actions = 6
        keep_actions = np.arange(0, total_num_actions)
        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = mxia_coarse_action_enumerator
        self._train = list(range(len(self._joints)))
        self.labels.sort()

    def __getitem__(self, index):
        same_style = self.same_style[index]
        ids = self.ids[index]

        actions = self._actions[index]
        joints = self._joints[index]
        joints = joints.reshape(joints.shape[0], -1)
        motion_length = 60
        motion_len = joints.shape[0]

        # random sample
        if motion_len >= motion_length:
            gap = motion_len - motion_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            end = start + motion_length
            r_motion = joints[start:end]
            # offset deduction
            r_motion = r_motion - np.tile(r_motion[0, :3], (1, int(r_motion.shape[-1] / 3)))
        # padding
        else:
            gap = motion_length - motion_len
            last_pose = np.expand_dims(joints[-1], axis=0)
            pad_poses = np.repeat(last_pose, gap, axis=0)
            r_motion = np.concatenate([joints, pad_poses], axis=0)

        if same_style.shape[0] >= motion_length:
            gap_style = same_style.shape[0] - motion_length
            start_style = 0 if gap_style == 0 else np.random.randint(0, gap_style, 1)[0]
            end_style = start_style + motion_length
            same = same_style[start_style:end_style]
            same = same - np.tile(same[0, :3], (1, int(same.shape[-1]/3)))
        # padding
        else:
            gap_style = motion_length - same_style.shape[0]
            last_pose_style = np.expand_dims(same_style[-1], axis=0)
            pad_poses_style = np.repeat(last_pose_style, gap_style, axis=0)
            same = np.concatenate([same_style, pad_poses_style], axis=0)

        r_motion = r_motion.reshape(-1, 21, 3)
        r_motion = r_motion.transpose(1, 2, 0)
        r_motion = torch.from_numpy(r_motion).float()

        same = same.reshape(-1, 21, 3)
        same = same.transpose(1, 2, 0)
        same = torch.from_numpy(same).float()

        return r_motion, actions, ids, same

    def __len__(self):
        return len(self._joints)

    def _get_item_data_index(self, data_index):
        nframes = self._num_frames_in_video[data_index]
        max_len = -1
        min_len = -1

        if self.num_frames == -1 and (max_len == -1 or nframes <= max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if max_len != -1:
                    max_frame = min(nframes, max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(min_len, max(max_frame, min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else max_len
            if num_frames > nframes:
                fair = False
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes), padding))

            elif self.sampling in ["conseq", "random_conseq"]:  # conseq
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

            inp, target = self.get_pose_data(data_index, frame_ix)
            return inp, target

    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            return self._label_to_action[label]
        else:  # if it is one hot vector
            label = np.argmax(label)
            return self._label_to_action[label]

    def get_label_sample(self, label, n=1, return_labels=False, return_index=False):
        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(np.array(self._actions)[index] == action).squeeze(1)

        if n == 1:
            data_index = index[np.random.choice(choices)]
            x, y, id, same = self.__getitem__(data_index)
            assert (label == y)
            y = label
        else:
            data_index = np.random.choice(choices, n)
            x = np.stack([self._get_item_data_index(index[di])[0] for di in data_index])
            y = label * np.ones(n, dtype=int)

        if return_labels:
            if return_index:
                return x, y, id, same, data_index
            return x, y, id, same
        else:
            if return_index:
                return x, data_index
            return x

    def get_label_sample_batch(self, labels):
        samples = [self.get_label_sample(label, n=1, return_labels=True, return_index=False) for label in labels]
        batch = collate(samples)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        same = batch["same_style"]
        ids = batch["ids"]
        return x, mask, lengths, same, ids

    def action_to_action_name(self, action):
        return self._action_classes[action]

    def label_to_action_name(self, label):
        action = self.label_to_action(label)
        return self.action_to_action_name(action)

    def ids_to_style_name(self, ids):
        style_id = ids.split('_')[-1]
        style_name = style_idx_to_name.get(int(style_id), None)
        return style_name

mxia_coarse_action_enumerator = {
    0: "walk",
    1: "run",
    2: "jump",
    3: "punch",
    4: "kick",
    5: "transition"
}