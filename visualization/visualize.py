import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from visualization.anim import plot_3d_motion_dico, load_anim
def stack_images(real, real_gens, gen):
    nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    allframes = np.concatenate((real[:, None, ...], *[x[:, None, ...] for x in real_gens], gen), 1)
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((*columns[0:nleft_cols], blackborder, *columns[nleft_cols:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)


def generate_by_video(visualization, reconstructions, generation,
                      label_to_action_name, ids_to_style_name, params, nats, nspa, tmp_path):
    fps = params["fps"]

    params = params.copy()

    outputkey = "output_xyz"
    params["pose_rep"] = "xyz"

    keep = [outputkey, "lengths", "y"]
    kep = [outputkey, "lengths", "y", "rxr_target", "style_ids"]

    visu = {key: visualization[key].data.cpu().numpy() for key in keep}

    recons = {key: reconstructions[key] for key in kep}
    gener = {key: generation[key].data.cpu().numpy() for key in keep}
    gener["rxr_target"] = generation["rxr_target"].cpu().numpy()
    gener["style_ids"] = generation["style_ids"]

    lenmax = max(gener["lengths"].max(),        # 80
                 visu["lengths"].max())

    timesize = lenmax + 5
    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format, isij):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        if isij:    # True
            array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                               for j in range(nats)]
                              for i in tqdm(range(nspa), desc=desc.format("Load"))])
            return array.transpose(2, 0, 1, 3, 4, 5)
        else:
            array = np.stack([load_anim(save_path_format.format(i), timesize)
                              for i in tqdm(range(nats), desc=desc.format("Load"))])
            return array.transpose(1, 0, 2, 3, 4)


    with multiprocessing.Pool() as pool:
        # Generated samples
        gen_path = os.path.join(tmp_path, "gen")
        if not os.path.isdir(gen_path):
            os.mkdir(gen_path)
        save_path_content = os.path.join(gen_path, "actor2motion_{}_{}.gif")
        save_path_style = os.path.join(gen_path, "actor2style_{}_{}.gif")
        iterator = ((gener[outputkey][i, j],
                     gener["lengths"][i, j],
                     gener["rxr_target"][i, j],
                     save_path_content.format(i, j),
                     save_path_style.format(i, j),
                     params,
                     {"title": f"gen: {label_to_action_name(gener['y'][i, j])}"+"_"+f"{ids_to_style_name(gener['style_ids'][i, j])}", "interval": 1000/fps})
                    for j in range(nats) for i in range(nspa))

        gener["frames"] = pool_job_with_desc(pool, iterator,
                                             "{} the generated samples",
                                             nats*nspa,
                                             save_path_content,
                                             True)

        # Real samples
        real_path = os.path.join(tmp_path, "real")
        if not os.path.isdir(real_path):
            os.mkdir(real_path)
        save_path_format = os.path.join(real_path, "real_{}.gif")
        iterator = ((visu[outputkey][i],
                     visu["lengths"][i],
                     save_path_format.format(i),
                     params, {"title": f"real: {label_to_action_name(visu['y'][i])}", "interval": 1000/fps})
                    for i in range(nats))
        visu["frames"] = pool_job_with_desc(pool, iterator,
                                            "{} the real samples",
                                            nats,
                                            save_path_format,
                                            False)

        recon_path = os.path.join(tmp_path, "recon")
        if not os.path.isdir(recon_path):
            os.mkdir(recon_path)

        save_path_content = os.path.join(recon_path, "actor2motion_{}.gif")
        save_path_style = os.path.join(recon_path, "actor2style_{}.gif")
        iterator = ((recons[outputkey][i],
                     recons["lengths"][i],
                     recons["rxr_target"][i],
                     save_path_content.format(i),
                     save_path_style.format(i),
                     params, {
                         "title": f"recons: {label_to_action_name(recons['y'][i])}" + "_" + f"{ids_to_style_name(recons['style_ids'][i])}",
                         "interval": 1000 / fps})
                    for i in range(nats))
        recons["frames"] = pool_job_with_desc(pool, iterator,
                                             "{} the reconstructed samples",
                                             nats,
                                             save_path_format,
                                             False)

    frames = stack_images(visu["frames"], [recons["frames"]], gener["frames"])
    return frames

def viz_epoch(model, dataset, epoch, params, folder, writer=None):
    """ Generate & viz samples """

    # visualization with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")

    noise_same_action = params["noise_same_action"]
    noise_diff_action = params["noise_diff_action"]

    fact = params["fact_latent"]
    figname = params["figname"].format(epoch)

    nspa = params["num_samples_per_action"]
    nats = params["num_actions_to_sample"]

    num_classes = params["num_classes"]

    # define some classes
    classes = torch.randperm(num_classes)[:nats]
    style_classes = torch.randperm(num_classes)[:nats]

    # extract the real samples
    content_samples, mask_content, content_lengths, content_same_style, content_ids = dataset.get_label_sample_batch(classes.numpy())
    style_samples, mask_style, style_lengths, style_same_style, style_ids = dataset.get_label_sample_batch(style_classes.numpy())

    # to visualization directly
    content_batch = {"x": content_samples.to(model.device),
                     "y": classes.to(model.device),
                     "lengths": content_lengths.to(model.device),
                     "mask": mask_content.to(model.device),
                     "ids": content_ids,
                     "same_style": content_same_style.to(model.device)}

    style_batch = {"x": style_samples.to(model.device),
                   "y": style_classes.numpy(),
                   "lengths": style_lengths.to(model.device),
                   "mask": mask_style.to(model.device),
                   "ids": style_ids,
                   "same_style": style_same_style.to(model.device)}

    # Visualizaion of real samples
    visualization = {"x": content_samples.to(model.device),
                     "y": classes.to(model.device),
                     "mask": mask_content.to(model.device),
                     "lengths": content_lengths.to(model.device),
                     "output": content_samples.to(model.device)}

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        # Reconstruction of the real data
        reconstructions_batch = {}
        reconstructions_content_batch, reconstructions_style_batch = model(content_batch, style_batch)
        reconstructions_batch["y"] = reconstructions_content_batch["y"].cpu().numpy()
        reconstructions_batch["lengths"] = reconstructions_content_batch["lengths"].cpu().numpy()
        reconstructions_batch["output_xyz"] = reconstructions_content_batch["output_xyz"].cpu().numpy()           # action2motion
        reconstructions_batch["style_ids"] = reconstructions_style_batch["ids"]
        reconstructions_batch["rxr_target"] = reconstructions_style_batch["rxr_target"].cpu().numpy()      # action2style

        batch = model.evaluate(content_batch, style_batch, nspa)  # generation
        for key, val in batch.items():
            if len(batch[key].shape) == 1:
                batch[key] = val.reshape(nspa, nats)
            else:
                batch[key] = val.reshape(nspa, nats, *val.shape[1:])
        batch["style_ids"] = np.array(style_batch['ids']*nspa).reshape(nspa, -1)

        # Get xyz for the real ones
        visualization["output_xyz"] = visualization["output"]

    finalpath = os.path.join(folder, figname + ".gif")
    tmp_path = os.path.join(folder, f"subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video(visualization, reconstructions_batch, batch,
                               dataset.label_to_action_name, dataset.ids_to_style_name, params, nats, nspa, tmp_path)


    print(f"Writing video {finalpath}..")
    imageio.mimsave(finalpath, frames, fps=params["fps"])

    if writer is not None:
        writer.add_video(f"Video/Epoch {epoch}", frames.transpose(0, 3, 1, 2)[None], epoch, fps=params["fps"])


def viz_dataset(dataset, params, folder):
    """ Generate & viz samples """
    print("Visualization of the dataset")

    nspa = params["num_samples_per_action"]
    nats = params["num_actions_to_sample"]

    num_classes = params["num_classes"]

    figname = "{}_{}_numframes_{}_sampling_{}_step_{}".format(params["dataset"],
                                                              params["pose_rep"],
                                                              params["num_frames"],
                                                              params["sampling"],
                                                              params["sampling_step"])

    # define some classes
    classes = torch.randperm(num_classes)[:nats]

    allclasses = classes.repeat(nspa, 1).reshape(nspa*nats)
    # extract the real samples
    real_samples, mask_real, real_lengths = dataset.get_label_sample_batch(allclasses.numpy())
    # to visualization directly

    # Visualizaion of real samples
    visualization = {"x": real_samples,
                     "y": allclasses,
                     "mask": mask_real,
                     "lengths": real_lengths,
                     "output": real_samples}

    from models.rotation2xyz import Rotation2xyz

    device = params["device"]
    rot2xyz = Rotation2xyz(device=device)

    rot2xyz_params = {"pose_rep": params["pose_rep"],
                      "glob_rot": params["glob_rot"],
                      "glob": params["glob"],
                      "jointstype": params["jointstype"],
                      "translation": params["translation"]}

    output = visualization["output"]
    visualization["output_xyz"] = rot2xyz(output.to(device),
                                          visualization["mask"].to(device), **rot2xyz_params)

    for key, val in visualization.items():
        if len(visualization[key].shape) == 1:
            visualization[key] = val.reshape(nspa, nats)
        else:
            visualization[key] = val.reshape(nspa, nats, *val.shape[1:])

    finalpath = os.path.join(folder, figname + ".gif")
    tmp_path = os.path.join(folder, f"subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video_sequences(visualization, dataset.label_to_action_name, params, nats, nspa, tmp_path)

    print(f"Writing video {finalpath}..")
    imageio.mimsave(finalpath, frames, fps=params["fps"])


def generate_by_video_sequences(visualization, label_to_action_name, params, nats, nspa, tmp_path):
    fps = params["fps"]

    if "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, "lengths", "y"]
    visu = {key: visualization[key].data.cpu().numpy() for key in keep}
    lenmax = visu["lengths"].max()

    timesize = lenmax + 5
    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                           for j in range(nats)]
                          for i in tqdm(range(nspa), desc=desc.format("Load"))])
        return array.transpose(2, 0, 1, 3, 4, 5)

    with multiprocessing.Pool() as pool:
        # Real samples
        save_path_format = os.path.join(tmp_path, "real_{}_{}.gif")
        iterator = ((visu[outputkey][i, j],
                     visu["lengths"][i, j],
                     save_path_format.format(i, j),
                     params, {"title": f"real: {label_to_action_name(visu['y'][i, j])}", "interval": 1000/fps})
                    for j in range(nats) for i in range(nspa))
        visu["frames"] = pool_job_with_desc(pool, iterator,
                                            "{} the real samples",
                                            nats,
                                            save_path_format)
    frames = stack_images_sequence(visu["frames"])
    return frames


def stack_images_sequence(visu):
    print("Stacking frames..")
    allframes = visu
    nframes, nspa, nats, h, w, pix = allframes.shape
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate(columns).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)
