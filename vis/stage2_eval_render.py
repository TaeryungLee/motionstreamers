from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMPLX_MODEL_DIR = Path("/home/taeryunglee/data/human_models")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stage2 import Stage2MotionDataset


def _drop_local_pytorch3d_from_path() -> list[str]:
    original = list(sys.path)
    root = PROJECT_ROOT.resolve()
    filtered = []
    for item in original:
        path = Path(item or os.getcwd()).resolve()
        if path == root:
            continue
        filtered.append(item)
    sys.path = filtered
    local = sys.modules.get("pytorch3d")
    if local is not None and str(getattr(local, "__file__", "")).endswith("/pytorch3d.py"):
        for key in list(sys.modules):
            if key == "pytorch3d" or key.startswith("pytorch3d."):
                sys.modules.pop(key, None)
    return original


def load_pytorch3d_modules() -> dict[str, Any]:
    original = _drop_local_pytorch3d_from_path()
    try:
        renderer = importlib.import_module("pytorch3d.renderer")
        structures = importlib.import_module("pytorch3d.structures")
        transforms = importlib.import_module("pytorch3d.transforms")
    finally:
        sys.path = original
    return {
        "Meshes": structures.Meshes,
        "MeshRenderer": renderer.MeshRenderer,
        "MeshRasterizer": renderer.MeshRasterizer,
        "SoftPhongShader": renderer.SoftPhongShader,
        "PointLights": renderer.PointLights,
        "RasterizationSettings": renderer.RasterizationSettings,
        "TexturesVertex": renderer.TexturesVertex,
        "FoVPerspectiveCameras": renderer.FoVPerspectiveCameras,
        "Materials": renderer.Materials,
        "look_at_view_transform": renderer.look_at_view_transform,
        "rotation_6d_to_matrix": transforms.rotation_6d_to_matrix,
        "matrix_to_axis_angle": transforms.matrix_to_axis_angle,
    }


def rgba_to_rgb_white_bg(image: np.ndarray) -> np.ndarray:
    rgba = np.asarray(image, dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4] if rgba.shape[-1] == 4 else np.ones_like(rgb[..., :1])
    out = rgb * alpha + (1.0 - alpha)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(255, 255, 255))
    draw.text((pad, pad), text, fill=(0, 0, 0), font=font)
    return np.asarray(image)


def look_at_camera_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32) if up is None else np.asarray(up, dtype=np.float32)
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    z_axis = eye - target
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-6)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def parse_motion_params(
    motion: torch.Tensor,
    dataset: str,
    modules: dict[str, Any],
) -> dict[str, torch.Tensor]:
    values = motion.float()
    T, D = values.shape
    if dataset == "trumans" and D != 315:
        raise ValueError(f"TRUMANS motion dim must be 315, got {D}")
    if dataset == "lingo" and D != 135:
        raise ValueError(f"LINGO motion dim must be 135, got {D}")

    rot6d_to_mat = modules["rotation_6d_to_matrix"]
    mat_to_aa = modules["matrix_to_axis_angle"]

    transl = values[:, 0:3]
    global_orient = mat_to_aa(rot6d_to_mat(values[:, 3:9].reshape(T, 1, 6))).reshape(T, 3)
    body_pose = mat_to_aa(rot6d_to_mat(values[:, 9:135].reshape(T, 21, 6))).reshape(T, 63)
    if dataset == "trumans":
        left_hand = mat_to_aa(rot6d_to_mat(values[:, 135:225].reshape(T, 15, 6))).reshape(T, 45)
        right_hand = mat_to_aa(rot6d_to_mat(values[:, 225:315].reshape(T, 15, 6))).reshape(T, 45)
    else:
        left_hand = values.new_zeros((T, 45))
        right_hand = values.new_zeros((T, 45))
    return {
        "transl": transl,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand,
        "right_hand_pose": right_hand,
    }


def motion_to_vertices(
    motion: torch.Tensor,
    dataset: str,
    smplx_model_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    import smplx

    modules = load_pytorch3d_modules()
    params = parse_motion_params(motion.to(device), dataset=dataset, modules=modules)
    T = int(motion.shape[0])
    smpl_model = smplx.create(
        str(smplx_model_dir),
        model_type="smplx",
        gender="male",
        ext="pkl",
        num_betas=10,
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=T,
    ).to(device)
    smpl_model.eval()
    with torch.no_grad():
        out = smpl_model(return_verts=True, **params)
    faces = torch.as_tensor(np.asarray(smpl_model.faces, dtype=np.int64), dtype=torch.long, device=device)
    return out.vertices.detach(), faces


def render_vertices_video(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    out_path: Path,
    label: str,
    image_size: int = 384,
    fps: int = 30,
    color: tuple[float, float, float] = (0.9, 0.25, 0.2),
) -> np.ndarray:
    modules = load_pytorch3d_modules()
    device = vertices.device
    Meshes = modules["Meshes"]
    MeshRenderer = modules["MeshRenderer"]
    MeshRasterizer = modules["MeshRasterizer"]
    SoftPhongShader = modules["SoftPhongShader"]
    PointLights = modules["PointLights"]
    RasterizationSettings = modules["RasterizationSettings"]
    TexturesVertex = modules["TexturesVertex"]
    FoVPerspectiveCameras = modules["FoVPerspectiveCameras"]
    Materials = modules["Materials"]
    look_at_view_transform = modules["look_at_view_transform"]

    mins = vertices.reshape(-1, 3).min(dim=0).values
    maxs = vertices.reshape(-1, 3).max(dim=0).values
    center = ((mins + maxs) * 0.5).detach()
    extent = float((maxs - mins).max().detach().cpu())
    dist = max(3.0, extent * 1.6)
    at = ((float(center[0].cpu()), float(center[1].cpu()), float(center[2].cpu())),)
    R, T = look_at_view_transform(dist=dist, elev=7.0, azim=0.0, at=at, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=55.0)
    raster_settings = RasterizationSettings(
        image_size=int(image_size),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        max_faces_per_bin=0,
    )
    lights = PointLights(device=device, location=[[float(center[0].cpu()), float(center[1].cpu()) + 2.5, float(center[2].cpu()) + 2.5]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    materials = Materials(device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)

    frames: list[np.ndarray] = []
    rgb = torch.tensor(color, dtype=vertices.dtype, device=device).view(1, 1, 3)
    chunk_size = 8
    progress = tqdm(range(0, vertices.shape[0], chunk_size), desc=f"render {label}", unit="chunk", leave=False)
    for start in progress:
        end = min(start + chunk_size, vertices.shape[0])
        verts = vertices[start:end]
        faces_b = faces.unsqueeze(0).expand(end - start, -1, -1)
        textures = TexturesVertex(verts_features=rgb.expand(end - start, verts.shape[1], 3))
        mesh = Meshes(verts=verts, faces=faces_b, textures=textures)
        with torch.no_grad():
            images = renderer(mesh, materials=materials).detach().cpu().numpy()
        for image in images:
            frame = rgba_to_rgb_white_bg((image * 255.0).astype(np.uint8))
            frames.append(add_label(frame, label))

    video = np.stack(frames, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, video, fps=int(fps))
    return video


def render_vertices_video_safe(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    out_path: Path,
    label: str,
    image_size: int,
    fps: int,
    color: tuple[float, float, float],
) -> np.ndarray:
    return render_vertices_video(vertices, faces, out_path, label, image_size=image_size, fps=fps, color=color)


def save_side_by_side(gt_video: np.ndarray, gen_video: np.ndarray, out_path: Path, fps: int = 30) -> None:
    length = min(gt_video.shape[0], gen_video.shape[0])
    video = np.concatenate([gt_video[:length], gen_video[:length]], axis=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, video, fps=int(fps))


def render_stage2_pair(
    gt_motion: torch.Tensor,
    gen_motion: torch.Tensor,
    dataset: str,
    out_dir: Path,
    smplx_model_dir: Path = DEFAULT_SMPLX_MODEL_DIR,
    render_device: str = "cuda",
    image_size: int = 384,
    fps: int = 30,
    frame_stride: int = 1,
    meta: dict[str, Any] | None = None,
) -> None:
    device = torch.device(render_device if torch.cuda.is_available() and str(render_device).startswith("cuda") else "cpu")
    stride = max(1, int(frame_stride))
    if stride > 1:
        gt_motion = gt_motion[::stride]
        gen_motion = gen_motion[::stride]
        fps = max(1, int(round(float(fps) / float(stride))))
    gt_vertices, faces = motion_to_vertices(gt_motion, dataset, smplx_model_dir, device)
    gen_vertices, _ = motion_to_vertices(gen_motion, dataset, smplx_model_dir, device)
    gt_video = render_vertices_video_safe(gt_vertices, faces, out_dir / "gt.mp4", "GT", image_size=image_size, fps=fps, color=(0.2, 0.45, 0.95))
    gen_video = render_vertices_video_safe(gen_vertices, faces, out_dir / "gen.mp4", "Gen", image_size=image_size, fps=fps, color=(0.9, 0.2, 0.2))
    save_side_by_side(gt_video, gen_video, out_dir / "side_by_side.mp4", fps=fps)
    if meta is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def render_stage2_gt(
    motion: torch.Tensor,
    dataset: str,
    out_dir: Path,
    smplx_model_dir: Path = DEFAULT_SMPLX_MODEL_DIR,
    render_device: str = "cuda",
    image_size: int = 384,
    fps: int = 30,
    frame_stride: int = 1,
    meta: dict[str, Any] | None = None,
) -> None:
    device = torch.device(render_device if torch.cuda.is_available() and str(render_device).startswith("cuda") else "cpu")
    stride = max(1, int(frame_stride))
    if stride > 1:
        motion = motion[::stride]
        fps = max(1, int(round(float(fps) / float(stride))))
    vertices, faces = motion_to_vertices(motion, dataset, smplx_model_dir, device)
    render_vertices_video_safe(vertices, faces, out_dir / "gt.mp4", "GT", image_size=image_size, fps=fps, color=(0.2, 0.45, 0.95))
    if meta is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage2 GT motion samples with PyTorch3D.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--task", choices=["move_wait", "action"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_eval_gt_vis"))
    parser.add_argument("--smplx-model-dir", type=Path, default=DEFAULT_SMPLX_MODEL_DIR)
    parser.add_argument("--render-device", default="cuda")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Stage2MotionDataset(dataset=args.dataset, task=args.task, split=args.split, seed=int(args.seed))
    count = min(int(args.num_samples), len(dataset))
    root = args.output_dir / args.dataset / args.task
    for idx in range(count):
        item = dataset[idx]
        meta = {
            "dataset": args.dataset,
            "task": args.task,
            "split": args.split,
            "index": idx,
            "scene_id": item["scene_id"],
            "sequence_id": item["sequence_id"],
            "segment_id": int(item["segment_id"]),
            "goal_type": item["goal_type"],
            "text": item["text"],
            "length": int(item["length"]),
        }
        render_stage2_gt(
            item["motion_raw"],
            dataset=args.dataset,
            out_dir=root / f"sample_{idx:03d}",
            smplx_model_dir=args.smplx_model_dir,
            render_device=args.render_device,
            image_size=int(args.image_size),
            fps=int(args.fps),
            frame_stride=int(args.frame_stride),
            meta=meta,
        )


if __name__ == "__main__":
    main()
