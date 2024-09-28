import os
import sys
from tkinter import Image
from torchvision import transforms
import gradio as gr
import rembg
import torch
from PIL import Image
import imageio
import numpy as np
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from scene.gaussian_predictor import GaussianSplatPredictor
from utils.app_utils import remove_background, resize_foreground, set_white_background, resize_to_128, \
    get_source_camera_v2w_rmo_and_quats, to_tensor


def create_3d_rotation_video_with_opacity(xyz, scale, opacity, output_dir,
                                          video_name="point_cloud_opacity_rotation.mp4", num_frames=360,
                                          opacity_threshold=0.01):
    valid_points_mask = opacity.squeeze() > opacity_threshold  # Filter points by opacity
    delta_x = xyz[:, 0][valid_points_mask]
    delta_y = xyz[:, 1][valid_points_mask]
    delta_z = xyz[:, 2][valid_points_mask]
    scale = scale[:, 0][valid_points_mask]  # First scaling value
    opacity = opacity.squeeze()[valid_points_mask]  # Filtered opacity values

    # Map opacity to a heatmap
    colors = plt.cm.hot(opacity.detach().cpu().numpy())  # Use a heatmap (e.g., 'hot') to map opacity

    scale = scale.detach().cpu().numpy() * 100  # Multiply by 100 to get appropriate sizes

    fig = plt.figure(figsize=(10, 8))  # Enlarged figure
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Delta X')
    ax.set_ylabel('Delta Y')
    ax.set_zlabel('Delta Z')
    ax.set_title('3D Point Cloud with Opacity Heatmap')  # Add title

    frames = []
    for angle in np.linspace(0, 360, num_frames):
        ax.cla()
        ax.scatter(delta_x.detach().cpu().numpy(),
                   delta_y.detach().cpu().numpy(),
                   delta_z.detach().cpu().numpy(),
                   s=scale,
                   c=colors,  # Use colors based on opacity
                   marker='o')
        ax.view_init(30, angle)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    video_path = os.path.join(output_dir, video_name)
    imageio.mimsave(video_path, frames, fps=30)
    plt.close(fig)
    return video_path


def create_3d_rotation_video(xyz, scale, output_dir, video_name="point_cloud_rotation.mp4", num_frames=360):
    delta_x, delta_y, delta_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    scale = scale[:, 0]  # Assuming you're using the first scaling value to control size
    fig = plt.figure(figsize=(10, 8))  # Enlarged figure
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Delta X')
    ax.set_ylabel('Delta Y')
    ax.set_zlabel('Delta Z')
    ax.set_title('3D Point Cloud Rotation')  # Add title

    scale = scale.detach().cpu().numpy() * 100  # Multiply by 100 to get appropriate sizes

    frames = []
    for angle in np.linspace(0, 360, num_frames):
        ax.cla()
        ax.scatter(delta_x.detach().cpu().numpy(),
                   delta_y.detach().cpu().numpy(),
                   delta_z.detach().cpu().numpy(),
                   s=scale,
                   c='r', marker='o')
        ax.view_init(30, angle)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    video_path = os.path.join(output_dir, video_name)
    imageio.mimsave(video_path, frames, fps=30)
    plt.close(fig)
    return video_path


def create_3d_rotation_video_with_scale(xyz, scale, output_dir, num_frames=360):
    delta_x, delta_y, delta_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    scale_x, scale_y, scale_z = scale[:, 0], scale[:, 1], scale[:, 2]

    scale_video_paths = []
    scale_tensors = [scale_x, scale_y, scale_z]
    scale_names = ['scale_x', 'scale_y', 'scale_z']
    colormaps = ['Blues', 'Greens', 'Oranges']

    for i, (scale_dim, scale_name, cmap) in enumerate(zip(scale_tensors, scale_names, colormaps)):
        scale_dim = scale_dim.detach().cpu().numpy() * 100
        fig = plt.figure(figsize=(10, 8))  # Enlarged figure
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Delta X')
        ax.set_ylabel('Delta Y')
        ax.set_zlabel('Delta Z')
        ax.set_title(f'3D Point Cloud with {scale_name.capitalize()} Scaling')  # Add title

        frames = []
        for angle in np.linspace(0, 360, num_frames):
            ax.cla()
            ax.scatter(delta_x.detach().cpu().numpy(),
                       delta_y.detach().cpu().numpy(),
                       delta_z.detach().cpu().numpy(),
                       s=scale_dim,
                       c=scale_dim, cmap=cmap, marker='o')
            ax.view_init(30, angle)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        video_path = os.path.join(output_dir, f"{scale_name}_rotation.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        plt.close(fig)
        scale_video_paths.append(video_path)

    return scale_video_paths


def create_3d_rotation_video_with_quaternion(xyz, quaternions, output_dir, num_frames=360):
    delta_x, delta_y, delta_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    quaternion_w, quaternion_x, quaternion_y, quaternion_z = quaternions[:, 0], quaternions[:, 1], quaternions[:,
                                                                                                   2], quaternions[:, 3]

    quaternion_video_paths = []
    quaternion_tensors = [quaternion_w, quaternion_x, quaternion_y, quaternion_z]
    quaternion_names = ['quaternion_w', 'quaternion_x', 'quaternion_y', 'quaternion_z']
    colormaps = ['Purples', 'Reds', 'Greens', 'Blues']

    for i, (quat_dim, quat_name, cmap) in enumerate(zip(quaternion_tensors, quaternion_names, colormaps)):
        quat_dim = quat_dim.detach().cpu().numpy() * 100
        fig = plt.figure(figsize=(10, 8))  # Enlarged figure
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Delta X')
        ax.set_ylabel('Delta Y')
        ax.set_zlabel('Delta Z')
        ax.set_title(f'3D Point Cloud with {quat_name.capitalize()} Quaternion')  # Add title

        frames = []
        for angle in np.linspace(0, 360, num_frames):
            ax.cla()
            ax.scatter(delta_x.detach().cpu().numpy(),
                       delta_y.detach().cpu().numpy(),
                       delta_z.detach().cpu().numpy(),
                       s=quat_dim,
                       c=quat_dim, cmap=cmap, marker='o')
            ax.view_init(30, angle)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        video_path = os.path.join(output_dir, f"{quat_name}_rotation.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        plt.close(fig)
        quaternion_video_paths.append(video_path)

    return quaternion_video_paths


def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)



def check_input_image(input_image):
        if input_image is None:
            raise gr.Error("No image uploaded!")

def preprocess(input_image, rembg_session, preprocess_background=True, foreground_ratio=0.65):
        if preprocess_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = set_white_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = set_white_background(image)
        image = resize_to_128(image)
        return image


def get_reconstruction(image,model, device, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)

    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = to_tensor(image).to(device)
    view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
    view_to_world_source = view_to_world_source.to(device)
    rot_transform_quats = rot_transform_quats.to(device)

    reconstruction_unactivated = model(
        image.unsqueeze(0).unsqueeze(0),
        view_to_world_source,
        rot_transform_quats,
        None,
        activate_output=False
    )

    reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
    reconstruction["scaling"] = model.scaling_activation(reconstruction["scaling"])
    reconstruction["opacity"] = model.opacity_activation(reconstruction["opacity"])

    return reconstruction



def main() -> None:
    # Example usage
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model_cfg = OmegaConf.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "gradio_config.yaml"
        ))

    model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1",
                                 filename="model_latest.pth")

    model = GaussianSplatPredictor(model_cfg)

    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to(device)

    # ============= image preprocessing =============
    rembg_session = rembg.new_session()

    args = sys.argv
    image = Image.open(args[1])

    reconstruction = get_reconstruction(image, model, device, output_dir)
    xyz = reconstruction["xyz"]
    scale = reconstruction["scaling"]
    opacity = reconstruction["opacity"]


    # Create a 3D rotation video with opacity
    create_3d_rotation_video_with_opacity(xyz, scale, opacity, output_dir)

    # Create a 3D rotation video with scale
    create_3d_rotation_video_with_scale(xyz, scale, output_dir)

    # Create a 3D rotation video with quaternion
    quaternions = reconstruction["quaternions"]
    create_3d_rotation_video_with_quaternion(xyz, quaternions, output_dir)


if __name__ == "__main__":
    main()
