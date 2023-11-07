import torch
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_or_test(model, optimizer, content_iterator, style_iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    dict_loss = {loss: 0 for loss in model.losses}

    with grad_env():
        for i, (content_batch, style_batch) in tqdm(enumerate(zip(content_iterator, style_iterator)), desc="Computing batch"):
            content_batch = {key: (val.to(device) if key != 'ids' else val) for key, val in content_batch.items()}
            style_batch = {key: (val.to(device) if key != 'ids' else val) for key, val in style_batch.items()}

            if mode == "train":
                optimizer.zero_grad()

            content_batch, style_batch = model(content_batch, style_batch)
            mixed_loss, losses = model.compute_loss(content_batch, style_batch)

            for key in dict_loss.keys():
                dict_loss[key] += losses[key]

            if mode == "train":
                mixed_loss.backward()
                optimizer.step()
    return dict_loss

def train_or_test_motion(model, optimizer,  iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    dict_loss = {loss: 0 for loss in model.losses}

    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            batch = {key: (val.to(device) if key != 'ids' else val) for key, val in batch.items()}

            if mode == "train":
                optimizer.zero_grad()

            batch = model(batch)
            mixed_loss, losses = model.compute_loss(batch)
            
            for key in dict_loss.keys():
                dict_loss[key] += losses[key]

            if mode == "train":
                mixed_loss.backward()
                optimizer.step()
    return dict_loss

def train(model, optimizer, content_iterator, style_iterator, device):
    return train_or_test(model, optimizer, content_iterator, style_iterator, device, mode="train")

def train_motion(model, optimizer, iterator, device):
    return train_or_test_motion(model, optimizer, iterator, device, mode="train")

def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
