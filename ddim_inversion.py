from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt
from torchvision.transforms.functional import pad
from pathlib import Path

def pad_to_divisible_by_8(img):
    print(f"image shape before padding = {img.shape}")
    _,_, height, width = img.shape
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8

    # Calculate padding for top/bottom and left/right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    padded_img = pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')
    print(f"image shape after padding = {padded_img.shape}")
    return padded_img

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    tensor_image =  tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension
    return pad_to_divisible_by_8(tensor_image)

def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.no_grad()
def ddim_inversion(imgname: str,prompt: str, num_steps: int = 50, verify: Optional[bool] = False) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base',
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae

    input_img = load_image(imgname,512).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt=prompt, negative_prompt="", guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)

    if verify:
        path_obj = Path(imgname)
        parent_path = path_obj.parent
        new_file_name = f"{path_obj.stem}_invert{path_obj.suffix}"
        new_path = parent_path / new_file_name
        pipe.scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
        image = pipe(prompt=prompt, negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        print(f"save image to {new_path}")
        plt.savefig(str(new_path))
    return inv_latents

