import torch
import torchvision
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from matplotlib.patches import Ellipse

from utils.app_utils import (
    remove_background,
    resize_foreground,
    set_white_background,
    resize_to_128,
    to_tensor,
    get_source_camera_v2w_rmo_and_quats,
    get_target_cameras,
    export_to_obj)

from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted
import gradio as gr
import rembg
from huggingface_hub import hf_hub_download


@torch.no_grad()
def main():

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model_cfg = OmegaConf.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
                    "gradio_config.yaml"
                    ))

     # Define the path to your local model
    local_model_path = "/content/splatter-image/experiments_out/2024-10-31/11-43-06/model_latest.pth"
    # Try loading the local model if it exists, otherwise fall back to Hugging Face Hub
    if os.path.exists(local_model_path):
        print(f"Loading model from local path: {local_model_path}")
        model_path = local_model_path
    else:
        print("Local model not found, downloading from Hugging Face Hub...")
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1", filename="model_latest.pth")

    model = GaussianSplatPredictor(model_cfg)

    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to(device)

    # ============= image preprocessing =============
    rembg_session = rembg.new_session()

    def check_input_image(input_image):
        if input_image is None:
            raise gr.Error("No image uploaded!")

    def preprocess(input_image, preprocess_background=True, foreground_ratio=0.65):
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

    def save_plot(fig, filename):
        fig.savefig(filename)
        plt.close(fig)

    def create_3d_rotation_video_with_opacity(xyz, scale, opacity, output_dir, video_name="point_cloud_opacity_rotation.mp4", num_frames=360, opacity_threshold=0.01):
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
        quaternion_w, quaternion_x, quaternion_y, quaternion_z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

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


    def reconstruct_and_export(image, output_dir="./output"):
        os.makedirs(output_dir, exist_ok=True)

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

        # Generate regular XYZ rotation video (without opacity heatmap)
        xyz_video_path = create_3d_rotation_video(
            reconstruction["xyz"], reconstruction["scaling"], output_dir
        )

        # Generate XYZ rotation video with opacity heatmap
        opacity_video_path = create_3d_rotation_video_with_opacity(
            reconstruction["xyz"], reconstruction["scaling"], reconstruction["opacity"], output_dir
        )

        # Generate scale videos for each dimension (x, y, z)
        scale_video_paths = create_3d_rotation_video_with_scale(
            reconstruction["xyz"], reconstruction["scaling"], output_dir
        )

        # Generate quaternion videos for each dimension (w, x, y, z)
        quaternion_video_paths = create_3d_rotation_video_with_quaternion(
            reconstruction["xyz"], reconstruction["rotation"], output_dir
        )


        # Define ply_out_path locally
        ply_out_path = os.path.join(output_dir, 'mesh.ply')

        # Render images in a loop
        world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        loop_renders = []
        t_to_512 = torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for r_idx in range(world_view_transforms.shape[0]):
            image = render_predicted(
                reconstruction,
                world_view_transforms[r_idx].to(device),
                full_proj_transforms[r_idx].to(device),
                camera_centers[r_idx].to(device),
                background,
                model_cfg,
                focals_pixels=None
            )["render"]
            image = t_to_512(image)
            loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))

        # Use the newly defined ply_out_path
        loop_out_path = os.path.join(os.path.dirname(ply_out_path), "loop.mp4")
        imageio.mimsave(loop_out_path, loop_renders, fps=25)

        # Export reconstruction to ply
        export_to_obj(reconstruction_unactivated, ply_out_path)

        # Return the paths to all videos
        return xyz_video_path, opacity_video_path, scale_video_paths, quaternion_video_paths, loop_out_path


    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """

    def run_example(image):
        output_dir = "./output"
        preprocessed = preprocess(image)
        ply_out_path, loop_out_path = reconstruct_and_export(np.array(preprocessed), output_dir)
        return preprocessed, ply_out_path, loop_out_path


    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # Splatter Image

            **Splatter Image (CVPR 2024)** [[code](https://github.com/szymanowiczs/splatter-image), [project page](https://szymanowiczs.github.io/splatter-image)] is a fast, super cheap-to-train method for object 3D reconstruction from a single image.
            The model used in the demo was trained on **Objaverse-LVIS on 2 A6000 GPUs for 3.5 days**.
            Locally, on an NVIDIA V100 GPU, reconstruction (forward pass of the network) can be done at 38FPS and rendering (with Gaussian Splatting) at 588FPS.
            Upload an image of an object or click on one of the provided examples to see how the Splatter Image does.
            The 3D viewer will render a .ply object exported from the 3D Gaussians, which is only an approximation.
            For best results run the demo locally and render locally with Gaussian Splatting - to do so, clone the [main repository](https://github.com/szymanowiczs/splatter-image).
            """
        )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
                with gr.Row():
                    with gr.Group():
                        preprocess_background = gr.Checkbox(
                            label="Remove Background", value=True
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(variant="panel"):
                    gr.Examples(
                        examples=[
                            './demo_examples/01_bigmac.png',
                            './demo_examples/02_hydrant.jpg',
                            './demo_examples/03_spyro.png',
                            './demo_examples/04_lysol.png',
                            './demo_examples/05_pinapple_bottle.png',
                            './demo_examples/06_unsplash_broccoli.png',
                            './demo_examples/07_objaverse_backpack.png',
                            './demo_examples/08_unsplash_chocolatecake.png',
                            './demo_examples/09_realfusion_cherry.png',
                            './demo_examples/10_triposr_teapot.png'
                        ],
                        inputs=[input_image],
                        cache_examples=False,
                        label="Examples",
                        examples_per_page=20,
                    )

            with gr.Column():
                with gr.Row():
                    with gr.Tab("Reconstruction"):
                        with gr.Column():
                            output_video = gr.Video(value=None, width=512, label="Rendered Video", autoplay=True)
                            output_model = gr.Model3D(
                                height=512,
                                label="Output Model",
                                interactive=False
                            )

        gr.Markdown(
            """
            ## Comments:
            1. If you run the demo online, the first example you upload should take about 4.5 seconds (with preprocessing, saving and overhead), the following take about 1.5s.
            2. The 3D viewer shows a .ply mesh extracted from a mix of 3D Gaussians. This is only an approximation and artefacts might show.
            3. Known limitations include:
            - a black dot appearing on the model from some viewpoints
            - see-through parts of objects, especially on the back: this is due to the model performing less well on more complicated shapes
            - back of objects are blurry: this is a model limitation due to it being deterministic.
            4. Our model is of comparable quality to state-of-the-art methods, and is **much** cheaper to train and run.

            ## How does it work?

            Splatter Image formulates 3D reconstruction as an image-to-image translation task. It maps the input image to another image,
            in which every pixel represents one 3D Gaussian and the channels of the output represent parameters of these Gaussians, including their shapes, colors, and locations.
            The resulting image thus represents a set of Gaussians (almost like a point cloud) which reconstruct the shape and color of the object.
            The method is very cheap: the reconstruction amounts to a single forward pass of a neural network with only 2D operators (2D convolutions and attention).
            The rendering is also very fast, due to using Gaussian Splatting.
            Combined, this results in very cheap training and high-quality results.
            For more results, see the [project page](https://szymanowiczs.github.io/splatter-image) and the [CVPR article](https://arxiv.org/abs/2312.13150).
            """
        )

        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, preprocess_background],
            outputs=[processed_image],
        ).success(
            fn=reconstruct_and_export,
            inputs=[processed_image],
            outputs=[output_model, output_video],
        )

    demo.queue(max_size=1)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
