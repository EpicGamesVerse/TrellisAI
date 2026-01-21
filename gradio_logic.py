import os
import shutil
from dataclasses import dataclass
from typing import Any, List, Tuple, Literal

import torch  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import imageio  # pyright: ignore[reportMissingImports]
from easydict import EasyDict as edict  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]

from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


@dataclass(frozen=True)
class AppContext:
    pipeline: Any
    tmp_dir: str
    max_seed: int


def start_session(ctx: AppContext, req: Any) -> None:
    user_dir = os.path.join(ctx.tmp_dir, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(ctx: AppContext, req: Any) -> None:
    user_dir = os.path.join(ctx.tmp_dir, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image(ctx: AppContext, image: Image.Image) -> Image.Image:
    return ctx.pipeline.preprocess_image(image)


def preprocess_images(ctx: AppContext, images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    images_only = [image[0] for image in images]
    return [ctx.pipeline.preprocess_image(image) for image in images_only]


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    assert gs._xyz is not None
    assert gs._features_dc is not None
    assert gs._scaling is not None
    assert gs._rotation is not None
    assert gs._opacity is not None

    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh


def get_seed(ctx: AppContext, randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, ctx.max_seed) if randomize_seed else seed


def image_to_3d(
    ctx: AppContext,
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: Any,
    progress: Any = None,
) -> Tuple[dict, str]:
    user_dir = os.path.join(ctx.tmp_dir, str(req.session_hash))

    if progress is not None:
        progress(0.01, desc="Preparing inputs…")

    if not is_multiimage:
        if progress is not None:
            progress(0.05, desc="Sampling (single image)…")
        outputs = ctx.pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        if progress is not None:
            progress(0.05, desc="Sampling (multi image)…")
        outputs = ctx.pipeline.run_multi_image(
            [image[0] for image in multiimages],
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )

    if progress is not None:
        progress(0.75, desc="Rendering preview…")

    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]

    video_path = os.path.join(user_dir, 'sample.mp4')
    if progress is not None:
        progress(0.9, desc="Saving preview…")
    imageio.mimsave(video_path, video, fps=15)

    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    if progress is not None:
        progress(1.0, desc="Done")
    return state, video_path


def extract_glb(
    ctx: AppContext,
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: Any,
    progress: Any = None,
) -> Tuple[str, str]:
    user_dir = os.path.join(ctx.tmp_dir, str(req.session_hash))

    if progress is not None:
        progress(0.05, desc="Loading state…")

    gs, mesh = unpack_state(state)
    if progress is not None:
        progress(0.25, desc="Post-processing to GLB…")
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)

    glb_path = os.path.join(user_dir, 'sample.glb')
    if progress is not None:
        progress(0.9, desc="Writing GLB…")
    glb.export(glb_path)

    torch.cuda.empty_cache()
    if progress is not None:
        progress(1.0, desc="Done")
    return glb_path, glb_path


def extract_gaussian(ctx: AppContext, state: dict, req: Any, progress: Any = None) -> Tuple[str, str]:
    user_dir = os.path.join(ctx.tmp_dir, str(req.session_hash))

    if progress is not None:
        progress(0.1, desc="Loading state…")
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    if progress is not None:
        progress(0.8, desc="Writing PLY…")
    gs.save_ply(gaussian_path)

    torch.cuda.empty_cache()
    if progress is not None:
        progress(1.0, desc="Done")
    return gaussian_path, gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images: List[Image.Image] = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(ctx: AppContext, image: Image.Image) -> List[Image.Image]:
    image_arr = np.array(image)
    alpha = image_arr[..., 3]
    alpha = np.any(alpha > 0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()

    images: List[Image.Image] = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image_arr[:, s : e + 1]))

    return [preprocess_image(ctx, im) for im in images]
