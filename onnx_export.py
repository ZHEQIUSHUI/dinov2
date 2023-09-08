import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F

from dinov2.eval.depth.models import build_depther


BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

# backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2:hubconf-depth-linear", model=backbone_name)
# backbone_model.eval()
# backbone_model.cuda()


dinov2_ld = torch.hub.load('facebookresearch/dinov2:hubconf-depth-linear', f'{backbone_name}_ld')
dinov2_ld.eval()
import urllib

from PIL import Image

def load_image_from_url(url: str) -> Image:
    if "http" in url:
        with urllib.request.urlopen(url) as f:
            return Image.open(f).convert("RGB")
    else:
        return Image.open(url).convert("RGB")


EXAMPLE_IMAGE_URL = "ed3c4d5b507efc0044fbbd8fa6bb833.jpg"


image = load_image_from_url(EXAMPLE_IMAGE_URL)

import matplotlib
from torchvision import transforms


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((480,640)),
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)


transform = make_depth_transform()

scale_factor = 1
rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
transformed_image = transform(rescaled_image)
batch = transformed_image.unsqueeze(0)


# feature = backbone_model(batch)

# print(feature.shape)
import numpy as np

# image = np.array(image)

# height,width,_ = image.shape
# print(image.shape)

# img_mate = dict()
# img_mate["ori_shape"] = (width,height)
# img_mate["img_shape"] = (630,630)
# img_mate["pad_shape"] = (0,0)

depth = dinov2_ld(batch)
print(depth.shape)

depth_img = render_depth(depth.detach().squeeze().numpy())
# print(depth_img.shape)
depth_img = np.array(depth_img)
import cv2

cv2.imwrite(f"depth_{backbone_name}_ld.png",depth_img[:,:,::-1])

torch.onnx.export(dinov2_ld,batch,f'{backbone_name}_ld.onnx',opset_version=12)