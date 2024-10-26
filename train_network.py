# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16046duwDnLi8jjMoXzwqHHJGZN-MYtBH
"""

import glob
import hydra
import os
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader

from lightning.fabric import Fabric

from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss
import lpips as lpips_lib

from eval import evaluate_dataset
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset

import matplotlib.pyplot as plt

def export_point_cloud_as_ply(point_cloud, output_path):
    """
    Export a point cloud to a .ply file for visualization without using Open3D.

    Parameters:
        point_cloud (torch.Tensor): A tensor of shape [N, 3] representing 3D points.
        output_path (str): The path where the .ply file will be saved.
    """
    # Convert the tensor to numpy
    point_cloud_np = point_cloud.cpu().numpy()

    # Create the header for the .ply file
    header = f"""ply
format ascii 1.0
element vertex {len(point_cloud_np)}
property float x
property float y
property float z
end_header
"""

    # Write the header and point data to the file
    with open(output_path, 'w') as file:
        file.write(header)
        for point in point_cloud_np:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Point cloud saved to {output_path}")


#--------------------------------------------------------- HELPER FUNCS ---------------------------------------------------------------------------------

def splat_front_and_back(gaussians, camera_position):
    front_indices = []
    back_indices = []

    # Loop over batches
    for i in range(gaussians['rotation'].shape[0]):
        # Loop over individual Gaussians in the batch
        for j in range(gaussians['rotation'].shape[1]):
            # Check if the individual Gaussian is visible
            if is_visible(gaussians, camera_position, i, j):
                front_indices.append((i, j))  # Append to front layer
            else:
                back_indices.append((i, j))  # Append to back layer

    # Print the count of Gaussians in the front and back
    print(f"Number of Gaussians in front: {len(front_indices)}")
    print(f"Number of Gaussians in back: {len(back_indices)}")

    return front_indices, back_indices

def create_gaussian_splats_subset(gaussian_splats, indices):
    subset = {}
    expected_keys = ['xyz', 'scaling', 'rotation', 'opacity', 'features_dc']  # Add any other expected keys

    for key in expected_keys:
        if key not in gaussian_splats:
            print(f"Warning: Key '{key}' missing in gaussian_splats.")
            continue

        tensor = gaussian_splats[key]

        # For 2D tensors
        if tensor.dim() == 2:
            # Select [i, j] for each index pair
            subset[key] = torch.stack([tensor[i, j] for i, j in indices])

        # For 3D tensors
        elif tensor.dim() == 3:
            # Select [i, j, :] for each index pair
            subset[key] = torch.stack([tensor[i, j, :] for i, j in indices])

        # For 4D tensors (use as-is without reshaping unless necessary)
        elif tensor.dim() == 4:
            # Keep the 4D structure intact for each selected [i, j, :, :]
            subset[key] = torch.stack([tensor[i, j, :, :] for i, j in indices])

        # Handle unexpected dimensions with a warning
        else:
            print(f"Warning: Unexpected tensor dimension for key '{key}' with dim {tensor.dim()}. Skipping.")
            continue

    return subset





def is_visible(gaussian, camera_position, i, j):
    # Select the normal and position for the specific Gaussian (j)
    normal = gaussian['rotation'][i, j, :3]  # Shape [3]
    position = gaussian['xyz'][i, j]         # Shape [3]

    # Select one camera (e.g., the first camera)
    camera_position_per_batch = camera_position[i, 0, :]  # Shape [3]

    # Compute the direction vector
    direction = camera_position_per_batch - position  # Shape [3]

    # Normalize vectors
    normal = torch.nn.functional.normalize(normal, dim=-1)
    direction = torch.nn.functional.normalize(direction, dim=-1)

    # Perform dot product to check visibility
    dot_product = torch.sum(normal * direction)

    # Return whether this specific Gaussian is in front
    return dot_product > 0


# Function to plot losses
def plot_losses(losses, iteration):
    print("LOSSES PLOT SAVING IN PROCCES")
    epochs = range(1, len(losses['front']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses['front'], label='Front Loss')
    plt.plot(epochs, losses['back'], label='Back Loss')
    plt.plot(epochs, losses['combined'], label='Combined Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'loss_plot_iter_{iteration}.png')  # Save the plot to a file
    plt.close()  # Close the plot to free memory

@hydra.main(version_base=None, config_path='configs', config_name="default_config")

#--------------------------------------------------------- MAIN ---------------------------------------------------------------------------------
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    if cfg.general.mixed_precision:
        fabric = Fabric(accelerator="cuda", devices=cfg.general.num_devices, strategy="ddp",
                        precision="16-mixed")
    else:
        fabric = Fabric(accelerator="cuda", devices=cfg.general.num_devices, strategy="ddp")
    fabric.launch()

    if fabric.is_global_zero:
        vis_dir = os.getcwd()

        dict_cfg = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

        if os.path.isdir(os.path.join(vis_dir, "wandb")):
            run_name_path = glob.glob(os.path.join(vis_dir, "wandb", "latest-run", "run-*"))[0]
            print("Got run name path {}".format(run_name_path))
            run_id = os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
            print("Resuming run with id {}".format(run_id))
            wandb_run = wandb.init(project=cfg.wandb.project, resume=True,
                                   id=run_id, config=dict_cfg)

        else:
            wandb_run = wandb.init(project=cfg.wandb.project, reinit=True,
                                   config=dict_cfg)

    first_iter = 0
    device = safe_state(cfg)

    gaussian_predictor = GaussianSplatPredictor(cfg)
    gaussian_predictor = gaussian_predictor.to(memory_format=torch.channels_last)

    l = []
    if cfg.model.network_with_offset:
        l.append({'params': gaussian_predictor.network_with_offset.parameters(),
                  'lr': cfg.opt.base_lr})
    if cfg.model.network_without_offset:
        l.append({'params': gaussian_predictor.network_wo_offset.parameters(),
                  'lr': cfg.opt.base_lr})
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15,
                                 betas=cfg.opt.betas)

    # Resuming training
    if fabric.is_global_zero:
        if os.path.isfile(os.path.join(vis_dir, "model_latest.pth")):
            print('Loading an existing model from ', os.path.join(vis_dir, "model_latest.pth"))
            checkpoint = torch.load(os.path.join(vis_dir, "model_latest.pth"),
                                    map_location=device)
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                                   strict=False)
                print("Warning, model mismatch - was this expected?")
            first_iter = checkpoint["iteration"]
            best_PSNR = checkpoint["best_PSNR"]
            print('Loaded model')
        # Resuming from checkpoint
        elif cfg.opt.pretrained_ckpt is not None:
            pretrained_ckpt_dir = os.path.join(cfg.opt.pretrained_ckpt, "model_latest.pth")
            checkpoint = torch.load(pretrained_ckpt_dir,
                                    map_location=device)
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                                   strict=False)
            best_PSNR = checkpoint["best_PSNR"]
            print('Loaded model from a pretrained checkpoint')
        else:
            best_PSNR = 0.0

    if cfg.opt.ema.use and fabric.is_global_zero:
        ema = EMA(gaussian_predictor,
                  beta=cfg.opt.ema.beta,
                  update_every=cfg.opt.ema.update_every,
                  update_after_step=cfg.opt.ema.update_after_step)
        ema = fabric.to_device(ema)

    if cfg.opt.loss == "l2":
        loss_fn = l2_loss
    elif cfg.opt.loss == "l1":
        loss_fn = l1_loss

    if cfg.opt.lambda_lpips != 0:
        lpips_fn = fabric.to_device(lpips_lib.LPIPS(net='vgg'))
    lambda_lpips = cfg.opt.lambda_lpips
    lambda_l12 = 1.0 - lambda_lpips

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32)
    background = fabric.to_device(background)

    if cfg.data.category in ["nmr", "objaverse"]:
        num_workers = 12
        persistent_workers = True
    else:
        num_workers = 0
        persistent_workers = False

    dataset = get_dataset(cfg, "train")
    dataloader = DataLoader(dataset,
                            batch_size=cfg.opt.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            persistent_workers=persistent_workers)

    val_dataset = get_dataset(cfg, "val")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                persistent_workers=True,
                                pin_memory=True)

    test_dataset = get_dataset(cfg, "vis")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=True)

    # distribute model and training dataset
    gaussian_predictor, optimizer = fabric.setup(
        gaussian_predictor, optimizer
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    gaussian_predictor.train()

    # Initialize a dictionary to track front, back, and combined losses
    losses = {'front': [], 'back': [], 'combined': []}

    print("Beginning training")
    first_iter += 1
    iteration = first_iter

    for num_epoch in range((cfg.opt.iterations + 1 - first_iter) // len(dataloader) + 1):
        dataloader.sampler.set_epoch(num_epoch)
        print("EPOCH NUM %d" % num_epoch)
        num_batches = len(dataloader) // 10
        print(num_batches)

        for batch_idx, data in enumerate(dataloader):
            # Break the loop after processing 1/4 of the batches
            print(batch_idx)
            if batch_idx >= num_batches:
                break

        #for data in dataloader:
            iteration += 1

            print("Starting iteration {} on process {}".format(iteration, fabric.global_rank))

            # =============== Prepare input ================
            rot_transform_quats = data["source_cv2wT_quat"][:, :cfg.data.input_images]

            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                focals_pixels_pred = data["focals_pixels"][:, :cfg.data.input_images, ...]
                input_images = torch.cat([data["gt_images"][:, :cfg.data.input_images, ...],
                                          data["origin_distances"][:, :cfg.data.input_images, ...]],
                                         dim=2)
            else:
                focals_pixels_pred = None
                input_images = data["gt_images"][:, :cfg.data.input_images, ...]

            gaussian_splats = gaussian_predictor(input_images,
                                     data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                                     rot_transform_quats,
                                     focals_pixels_pred)

            # Classify Gaussians into front and back layers
            front_indices, back_indices = splat_front_and_back(gaussian_splats, data['camera_centers'])

            # Create subsets for front and back layers
            front_gaussian_splats = create_gaussian_splats_subset(gaussian_splats, front_indices)
            back_gaussian_splats = create_gaussian_splats_subset(gaussian_splats, back_indices)

            # Initialize lists to store rendered images
            front_rendered_images = []
            back_rendered_images = []
            gt_images_front = []
            gt_images_back = []

            for b_idx in range(data["gt_images"].shape[0]):
                for r_idx in range(cfg.data.input_images, data["gt_images"].shape[1]):

                    # Proceed with rendering
                    front_image = render_predicted(
                        front_gaussian_splats,
                        data["world_view_transforms"][b_idx, r_idx],
                        data["full_proj_transforms"][b_idx, r_idx],
                        data["camera_centers"][b_idx, r_idx],
                        background,
                        cfg
                    )["render"]

                    front_rendered_images.append(front_image)

                    # Render back layer
                    back_image = render_predicted(
                        back_gaussian_splats,
                        data["world_view_transforms"][b_idx, r_idx],
                        data["full_proj_transforms"][b_idx, r_idx],
                        data["camera_centers"][b_idx, r_idx],
                        background,
                        cfg
                    )["render"]
                    back_rendered_images.append(back_image)

                    # Get ground truth images for front and back layers if they exist
                    if "target_front" in data:
                        gt_image_front = data["target_front"][b_idx, r_idx]
                    else:
                       # print("Warning: 'target_front' not found, using 'gt_images' as fallback.")
                        gt_image_front = data["gt_images"][b_idx, r_idx]

                    if "target_back" in data:
                        gt_image_back = data["target_back"][b_idx, r_idx]
                    else:
                        #print("Warning: 'target_back' not found, using 'gt_images' as fallback.")
                        gt_image_back = data["gt_images"][b_idx, r_idx]

                    gt_images_front.append(gt_image_front)
                    gt_images_back.append(gt_image_back)



            # Stack rendered images and ground truth images
            front_rendered_images = torch.stack(front_rendered_images, dim=0)
            back_rendered_images = torch.stack(back_rendered_images, dim=0)
            gt_images_front = torch.stack(gt_images_front, dim=0)
            gt_images_back = torch.stack(gt_images_back, dim=0)

            # Add these print statements here to check min/max values
            print("Front rendered images min, max:", front_rendered_images.min().item(), front_rendered_images.max().item())
            print("Back rendered images min, max:", back_rendered_images.min().item(), back_rendered_images.max().item())
            print("GT images front min, max:", gt_images_front.min().item(), gt_images_front.max().item())
            print("GT images back min, max:", gt_images_back.min().item(), gt_images_back.max().item())
            # Clamp the rendered images to the range [0, 1]
            front_rendered_images = torch.clamp(front_rendered_images, 0.0, 1.0)
            back_rendered_images = torch.clamp(back_rendered_images, 0.0, 1.0)

            # Compute losses
            front_loss = loss_fn(front_rendered_images, gt_images_front)
            back_loss = loss_fn(back_rendered_images, gt_images_back)

            combined_loss = front_loss + back_loss

            # Compute total loss
            total_loss = combined_loss * lambda_l12
            # Record the losses
            losses['front'].append(front_loss.item())
            losses['back'].append(back_loss.item())
            losses['combined'].append(combined_loss.item())

            # Plot and save losses every 'plot_interval' iterations
            plot_losses(losses, iteration)

            if cfg.opt.lambda_lpips != 0:
                lpips_loss_front = lpips_fn(front_rendered_images * 2 - 1, gt_images_front * 2 - 1)
                lpips_loss_back = lpips_fn(back_rendered_images * 2 - 1, gt_images_back * 2 - 1)
                lpips_loss_sum = lpips_loss_front + lpips_loss_back
                total_loss += lpips_loss_sum * lambda_lpips
            # Plot losses every 'n' iterations (e.g., every 100 iterations)
            plot_interval = 100  # Define how often to plot

            if iteration % plot_interval == 0 and fabric.is_global_zero:
                plot_losses(losses, iteration)

            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                # regularize very big gaussians
                if len(torch.where(gaussian_splats["scaling"] > 20)[0]) > 0:
                    big_gaussian_reg_loss = torch.mean(
                        gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] > 20)] * 0.1)
                    print('Regularising {} big Gaussians on iteration {}'.format(
                        len(torch.where(gaussian_splats["scaling"] > 20)[0]), iteration))
                else:
                    big_gaussian_reg_loss = 0.0
                # regularize very small Gaussians
                if len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]) > 0:
                    small_gaussian_reg_loss = torch.mean(
                        -torch.log(gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] < 1e-5)]) * 0.1)
                    print('Regularising {} small Gaussians on iteration {}'.format(
                        len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]), iteration))
                else:
                    small_gaussian_reg_loss = 0.0
            # Render
            l12_loss_sum = 0.0
            lpips_loss_sum = 0.0
            rendered_images = []
            gt_images = []
            for b_idx in range(data["gt_images"].shape[0]):
                # image at index 0 is training, remaining images are targets
                # Rendering is done sequentially because gaussian rasterization code
                # does not support batching
                gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in gaussian_splats.items()}
                for r_idx in range(cfg.data.input_images, data["gt_images"].shape[1]):
                    if "focals_pixels" in data.keys():
                        focals_pixels_render = data["focals_pixels"][b_idx, r_idx].cpu()
                    else:
                        focals_pixels_render = None
                    image = render_predicted(gaussian_splat_batch,
                                             data["world_view_transforms"][b_idx, r_idx],
                                             data["full_proj_transforms"][b_idx, r_idx],
                                             data["camera_centers"][b_idx, r_idx],
                                             background,
                                             cfg,
                                             focals_pixels=focals_pixels_render)["render"]
                    # Put in a list for a later loss computation
                    rendered_images.append(image)
                    gt_image = data["gt_images"][b_idx, r_idx]
                    gt_images.append(gt_image)
            rendered_images = torch.stack(rendered_images, dim=0)
            gt_images = torch.stack(gt_images, dim=0)
            # Loss computation
            l12_loss_sum = loss_fn(rendered_images, gt_images)
            if cfg.opt.lambda_lpips != 0:
                lpips_loss_sum = torch.mean(
                    lpips_fn(rendered_images * 2 - 1, gt_images * 2 - 1),
                )

            total_loss = l12_loss_sum * lambda_l12 + lpips_loss_sum * lambda_lpips
            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                total_loss = total_loss + big_gaussian_reg_loss + small_gaussian_reg_loss

            # Backpropagation and optimization
            assert not total_loss.isnan(), "Found NaN loss!"
            fabric.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()


            print("finished opt {} on process {}".format(iteration, fabric.global_rank))

            if cfg.opt.ema.use and fabric.is_global_zero:
                ema.update()

            print("finished iteration {} on process {}".format(iteration, fabric.global_rank))

            gaussian_predictor.eval()

            # ========= Logging =============
            with torch.no_grad():
                if iteration % cfg.logging.loss_log == 0 and fabric.is_global_zero:
                    wandb.log({
                        "front_loss": front_loss.item(),
                        "back_loss": back_loss.item(),
                        "combined_loss": combined_loss.item(),
                        "total_loss": total_loss.item()
                    }, step=iteration)

                    if cfg.opt.lambda_lpips != 0:
                        wandb.log({
                            "lpips_loss_front": lpips_loss_front.item(),
                            "lpips_loss_back": lpips_loss_back.item(),
                            "lpips_loss_sum": lpips_loss_sum.item()
                        }, step=iteration)
                    if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                        if type(big_gaussian_reg_loss) == float:
                            brl_for_log = big_gaussian_reg_loss
                        else:
                            brl_for_log = big_gaussian_reg_loss.item()
                        if type(small_gaussian_reg_loss) == float:
                            srl_for_log = small_gaussian_reg_loss
                        else:
                            srl_for_log = small_gaussian_reg_loss.item()
                        wandb.log({"reg_loss_big": np.log10(brl_for_log + 1e-8)}, step=iteration)
                        wandb.log({"reg_loss_small": np.log10(srl_for_log + 1e-8)}, step=iteration)

                if (iteration % cfg.logging.render_log == 0 or iteration == 1) and fabric.is_global_zero:
                    wandb.log({"render": wandb.Image(image.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy())},
                              step=iteration)
                    wandb.log({"gt": wandb.Image(gt_image.permute(1, 2, 0).detach().cpu().numpy())}, step=iteration)
                if (iteration % cfg.logging.loop_log == 0 or iteration == 1) and fabric.is_global_zero:
                    # torch.cuda.empty_cache()
                    try:
                        vis_data = next(test_iterator)
                    except UnboundLocalError:
                        test_iterator = iter(test_dataloader)
                        vis_data = next(test_iterator)
                    except StopIteration or UnboundLocalError:
                        test_iterator = iter(test_dataloader)
                        vis_data = next(test_iterator)

                    vis_data = {k: fabric.to_device(v) for k, v in vis_data.items()}

                    rot_transform_quats = vis_data["source_cv2wT_quat"][:, :cfg.data.input_images]

                    if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                        focals_pixels_pred = vis_data["focals_pixels"][:, :cfg.data.input_images, ...]
                        input_images = torch.cat([vis_data["gt_images"][:, :cfg.data.input_images, ...],
                                                  vis_data["origin_distances"][:, :cfg.data.input_images, ...]],
                                                 dim=2)
                    else:
                        focals_pixels_pred = None
                        input_images = vis_data["gt_images"][:, :cfg.data.input_images, ...]

                    gaussian_splats_vis = gaussian_predictor(input_images,
                                                             vis_data["view_to_world_transforms"][:,
                                                             :cfg.data.input_images, ...],
                                                             rot_transform_quats,
                                                             focals_pixels_pred)

                    test_loop = []
                    test_loop_gt = []
                    for r_idx in range(vis_data["gt_images"].shape[1]):
                        # We don't change the input or output of the network, just the rendering cameras
                        if "focals_pixels" in vis_data.keys():
                            focals_pixels_render = vis_data["focals_pixels"][0, r_idx]
                        else:
                            focals_pixels_render = None
                        test_image = render_predicted({k: v[0].contiguous() for k, v in gaussian_splats_vis.items()},
                                                      vis_data["world_view_transforms"][0, r_idx],
                                                      vis_data["full_proj_transforms"][0, r_idx],
                                                      vis_data["camera_centers"][0, r_idx],
                                                      background,
                                                      cfg,
                                                      focals_pixels=focals_pixels_render)["render"]
                        test_loop_gt.append(
                            (np.clip(vis_data["gt_images"][0, r_idx].detach().cpu().numpy(), 0, 1) * 255).astype(
                                np.uint8))
                        test_loop.append((np.clip(test_image.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8))

                    wandb.log({"rot": wandb.Video(np.asarray(test_loop), fps=20, format="mp4")},
                              step=iteration)
                    wandb.log({"rot_gt": wandb.Video(np.asarray(test_loop_gt), fps=20, format="mp4")},
                              step=iteration)

            fnames_to_save = []
            # Find out which models to save
            if (iteration + 1) % cfg.logging.ckpt_iterations == 0 and fabric.is_global_zero:
                # Always save the latest model
                fnames_to_save.append("model_latest.pth")

            # Remove validation and best model saving
            # The code below can be commented out or removed if not needed
            # if (iteration + 1) % cfg.logging.val_log == 0 and fabric.is_global_zero:
            #     torch.cuda.empty_cache()
            #     print("\n[ITER {}] Validating".format(iteration + 1))
            #     if cfg.opt.ema.use:
            #         scores = evaluate_dataset(
            #             ema,
            #             val_dataloader,
            #             device=device,
            #             model_cfg=cfg)
            #     else:
            #         scores = evaluate_dataset(
            #             gaussian_predictor,
            #             val_dataloader,
            #             device=device,
            #             model_cfg=cfg)
            #     wandb.log(scores, step=iteration + 1)

            #     # Save models - skip the best model, always overwrite the latest model
            #     # if scores["PSNR_novel"] > best_PSNR:
            #     #     fnames_to_save.append("model_best.pth")
            #     #     best_PSNR = scores["PSNR_novel"]
            #     #     print("\n[ITER {}] Saving new best checkpoint PSNR:{:.2f}".format(
            #     #         iteration + 1, best_PSNR))

            # Model saving process
            print("MODEL SAVING IN PROGRESS")
            for fname_to_save in fnames_to_save:
                ckpt_save_dict = {
                    "iteration": iteration,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss.item(),
                    "best_PSNR": best_PSNR
                }
                if cfg.opt.ema.use:
                    ckpt_save_dict["model_state_dict"] = ema.ema_model.state_dict()
                else:
                    ckpt_save_dict["model_state_dict"] = gaussian_predictor.state_dict()
                torch.save(ckpt_save_dict, os.path.join(vis_dir, fname_to_save))


                torch.cuda.empty_cache()


            # ============ Model saving =================
            print("MODEL SAVAING IN PROCCES")
            for fname_to_save in fnames_to_save:
                ckpt_save_dict = {
                    "iteration": iteration,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss.item(),
                    "best_PSNR": best_PSNR
                }
                if cfg.opt.ema.use:
                    ckpt_save_dict["model_state_dict"] = ema.ema_model.state_dict()
                else:
                    ckpt_save_dict["model_state_dict"] = gaussian_predictor.state_dict()
                torch.save(ckpt_save_dict, os.path.join(vis_dir, fname_to_save))

            gaussian_predictor.train()

    wandb_run.finish()


if __name__ == "__main__":
    main()