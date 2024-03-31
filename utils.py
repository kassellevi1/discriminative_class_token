from PIL import Image
import os
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
import torch

import kornia


# From timm.data.constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
import os
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join

def plot_images_with_epochs(image_folder, title):
    all_images = [f for f in os.listdir(image_folder) if (f.endswith('.jpg') and not f.startswith('plot_to_present'))]

    epochs = [int(f.split('_')[0]) for f in all_images]

    sorted_images = [x for _,x in sorted(zip(epochs,all_images))]

    max_epoch = max(epochs)
    indices = [round(i * (len(sorted_images) - 1) / 9) for i in range(10)]
    selected_images = [sorted_images[i] for i in indices]

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    fig.suptitle(title)

    for ax, img_name in zip(axes.flat, selected_images):
        epoch, class_name, loss = img_name.rsplit('_', 2)
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path)
        ax.imshow(img)
        shortened_class_name = class_name.split(',')[0]
        ax.set_xlabel(f"epoch: {epoch} | class: {shortened_class_name}", fontsize=7)  # Use set_xlabel for bottom placement
        ax.xaxis.labelpad = 4  # Adjust the padding to move closer or farther
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis ticks
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis ticks

    save_path = join(image_folder,"plot_to_present.jpg")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def transform_img_tensor(image, config):
    """
    Transforms an image based on the specified classifier input configurations.
    """
    if config.classifier == "inet":
        image = kornia.geometry.transform.resize(image, 256, interpolation="bicubic")
        image = kornia.geometry.transform.center_crop(image, (224, 224))
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    else:
        image = kornia.geometry.transform.resize(image, 224, interpolation="bicubic")
        image = kornia.geometry.transform.center_crop(image, (224, 224))
        image = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(image)
    return image


def prepare_classifier(config):
    if config.classifier == "inet":
        from transformers import ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224"
        ).cuda()
    elif config.classifier == "cub":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-cub"
        ).cuda()
    elif config.classifier == "inat":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-inat21"
        ).cuda()

    return model


def prepare_stable(config):
    # Generative model
    if config.sd_2_1:
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    else:
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(
        "cuda"
    )
    scheduler = pipe.scheduler
    del pipe
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    return unet, vae, text_encoder, scheduler, tokenizer


def save_progress(text_encoder, placeholder_token_id, accelerator, config, save_path):
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    learned_embeds_dict = {config.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)
