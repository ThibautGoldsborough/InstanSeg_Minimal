#!/usr/bin/python

import json
import os
import pdb

import fastremap
import numpy as np
import tifffile
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.morphology import label
from tqdm import tqdm
import warnings
from typing import Union


def moving_average(x, w):
    """Moving average of an array x with window size w"""
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_average(train, test, clip=99, window_size=10):
    fig = plt.figure(figsize=(10, 10))
    clip_val = np.percentile(test, [clip])
    test = np.clip(test, 0, clip_val[0])
    clip_val = np.percentile(train, [clip])
    train = np.clip(train, 0, clip_val[0])
    plt.plot(moving_average(test, window_size), label="test")
    plt.plot(moving_average(train, window_size), label="train")
    plt.legend()
    return fig


def _to_shape(a, shape):
    """Pad an array to a given shape."""
    a = _move_channel_axis(a)
    if len(np.shape(a)) == 2:
        a = a[None,]
    y_, x_ = shape
    y, x = a[0].shape
    y_pad = np.max([0, (y_ - y)])
    x_pad = np.max([0, (x_ - x)])
    return np.pad(a, ((0, 0), (y_pad // 2, y_pad // 2 + y_pad % 2),
                      (x_pad // 2, x_pad // 2 + x_pad % 2)),
                  mode='constant')


import rasterio.features
from rasterio.transform import Affine


def labels_to_features(lab: np.ndarray, object_type='annotation', connectivity: int = 4,
                       transform: Affine = None, downsample: float = 1.0, include_labels=False,
                       classification=None, offset=None):
    """
    Create a GeoJSON FeatureCollection from a labeled image.
    """
    features = []

    # Ensure types are valid
    if lab.dtype == bool:
        mask = lab
        lab = lab.astype(np.uint8)
    else:
        mask = lab != 0

    # Create transform from downsample if needed
    if transform is None:
        transform = Affine.scale(1.0)

    # Trace geometries
    for i, obj in enumerate(rasterio.features.shapes(lab, mask=mask,
                                                     connectivity=connectivity, transform=transform)):

        # Create properties
        props = dict(object_type=object_type)
        if include_labels:
            props['measurements'] = [{'name': 'Label', 'value': i}]
        #  pdb.set_trace()

        # Just to show how a classification can be added
        if classification is not None:
            props['classification'] = str(classification)

        if offset is not None:
            coordinates = obj[0]['coordinates']
            coordinates = [
                [(int(x[0] * downsample + offset[0]), int(x[1] * downsample + offset[1])) for x in coordinates[0]]]
            obj[0]['coordinates'] = coordinates

        # Wrap in a dict to effectively create a GeoJSON Feature
        po = dict(type="Feature", geometry=obj[0], properties=props)

        features.append(po)

    return features


def interp(image: np.ndarray, shape=None, scale=None):
    """Interpolate an image to a new shape or scale."""
    from scipy import interpolate
    x = np.array(range(image.shape[1]))
    y = np.array(range(image.shape[0]))
    interpolate_fn = interpolate.interp2d(x, y, image)
    if shape:
        x_new = np.linspace(0, image.shape[1] - 1, shape[1])
        y_new = np.linspace(0, image.shape[0] - 1, shape[0])
    elif scale:
        x_new = np.linspace(0, image.shape[1] - 1, int(np.floor(image.shape[1] * scale) + 1))
        y_new = np.linspace(0, image.shape[0] - 1, int(np.floor(image.shape[0] * scale) + 1))
    znew = interpolate_fn(x_new, y_new)
    return znew


from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc
import matplotlib as mpl


def show_images(*img_list, clip_pct=None, titles=None, save_str=False, n_cols=3, axes=False, cmap="viridis",
                labels=None,
                dpi=None, timer_flag=None, colorbar=True):
    """Designed to plot torch tensor and numpy arrays in windows robustly"""

    if labels is None:
        labels = []
    if titles is None:
        titles = []
    if dpi:
        mpl.rcParams['figure.dpi'] = dpi

    img_list = [img for img in img_list]
    if isinstance(img_list[0], list):
        img_list = img_list[0]

    rows = (len(img_list) - 1) // n_cols + 1
    columns = np.min([n_cols, len(img_list)])
    fig = plt.figure(figsize=(5 * (columns + 1), 5 * (rows + 1)))

    fig.tight_layout()
    grid = plt.GridSpec(rows, columns, figure=fig)
    grid.update(wspace=0.2, hspace=0, left=None, right=None, bottom=None, top=None)

    for i, img in enumerate(img_list):

        if torch.is_tensor(img):
            img = torch.squeeze(img).detach().cpu().numpy()
        img = np.squeeze(img)
        if len(img.shape) > 2:
            img = np.moveaxis(img, np.argmin(img.shape), -1)
            if img.shape[-1] > 4 or img.shape[-1] == 2:
                plt.close()
                show_images([img[..., i] for i in range(img.shape[-1])], clip_pct=clip_pct,
                            titles=["Channel:" + str(i) for i in range(img.shape[-1])], save_str=save_str,
                            n_cols=n_cols, axes=axes, cmap=cmap, colorbar=colorbar)
                continue
        ax1 = plt.subplot(grid[i])
        if not axes:
            plt.axis('off')
        if clip_pct is not None:
            print(np.percentile(img.ravel(), clip_pct), np.percentile(img.ravel(), 100 - clip_pct))
            im = ax1.imshow(img, vmin=np.percentile(img.ravel(), clip_pct),
                            vmax=np.percentile(img.ravel(), 100 - clip_pct))
        if i in labels:
            img = img.astype(int)
            img = fastremap.renumber(img)[0]
            n_instances = len(fastremap.unique(img))
            glasbey_cmap = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
            glasbey_cmap[0] = [0, 0, 0]  # Set bg to black
            cmap_lab = LinearSegmentedColormap.from_list('my_list', glasbey_cmap, N=n_instances)
            im = ax1.imshow(img, cmap=cmap_lab, interpolation='nearest')
        else:
            im = ax1.imshow(img, cmap=cmap)
        if colorbar:
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        if i < len(titles):
            ax1.set_title(titles[i])
    if not save_str:
        # plt.tight_layout()

        if timer_flag is not None:
            plt.show(block=False)
            plt.pause(timer_flag)
            plt.close()

        plt.show()

    if save_str:
        plt.savefig(str(save_str) + ".png", bbox_inches='tight')
        plt.close()
        return None


def _scale_length(size: float, pixel_size: float, do_round=True) -> float:
    """
    Convert length in calibrated units to a length in pixels
    """
    size_pixels = size / pixel_size
    return np.round(size_pixels) if do_round else size_pixels


def _scale_area(size: float, pixel_size: float, do_round=True) -> float:
    """
    Convert area in calibrated units to an area in pixels
    """
    size_pixels = size / (pixel_size * pixel_size)
    return np.round(size_pixels) if do_round else size_pixels


def _move_channel_axis(img: Union[np.ndarray, torch.Tensor], to_back: bool = False):
    if isinstance(img, np.ndarray):
        if img.ndim != 3:
            if img.ndim == 2:
                img = img[None,]
            if img.ndim != 3:
                raise ValueError("Input array should be 3D or 2D")
        ch = np.argmin(img.shape)
        if to_back:
            return np.rollaxis(img, ch, 3)

        return np.rollaxis(img, ch, 0)
    elif isinstance(img, torch.Tensor):
        if img.dim() != 3:
            if img.dim() == 2:
                img = img[None,]
            if img.dim() != 3:
                raise ValueError("Input array should be 3D or 2D")
        ch = np.argmin(img.shape)
        if to_back:
            return img.movedim(ch, -1)
        return img.movedim(ch, 0)


def percentile_normalize(img: Union[np.ndarray, torch.Tensor], percentile=0.1, subsampling_factor: int = 1,
                         epsilon: float = 1e-3):
    if isinstance(img, np.ndarray):
        assert img.ndim == 2 or img.ndim == 3, "Image must be 2D or 3D, got image of shape" + str(img.shape)
        img = np.atleast_3d(img)
        channel_axis = np.argmin(img.shape)
        img = _move_channel_axis(img, to_back=True)

        for c in range(img.shape[-1]):
            im_temp = img[::subsampling_factor, ::subsampling_factor, c]
            (p_min, p_max) = np.percentile(im_temp, [percentile, 100 - percentile])
            img[:, :, c] = (img[:, :, c] - p_min) / max(epsilon, p_max - p_min)
       # img = img / np.maximum(0.01, np.max(img))
        return np.moveaxis(img, 2, channel_axis)

    elif isinstance(img, torch.Tensor):
        assert img.ndim == 2 or img.ndim == 3, "Image must be 2D or 3D"
        img = torch.atleast_3d(img)
        channel_axis = np.argmin(img.shape)
        img = _move_channel_axis(img, to_back=True)
        for c in range(img.shape[-1]):
            im_temp = img[::subsampling_factor, ::subsampling_factor, c]
            (p_min, p_max) = torch.quantile(im_temp, torch.tensor([percentile / 100, (100 - percentile) / 100]))
            img[:, :, c] = (img[:, :, c] - p_min) / max(epsilon, p_max - p_min)
       # img = img / np.maximum(0.01, torch.max(img))
        return img.movedim(2, channel_axis)


def generate_colors(num_colors: int):
    import colorsys
    # Calculate the equally spaced hue values
    hues = [i / float(num_colors) for i in range(num_colors)]

    # Generate RGB colors
    rgb_colors = [list(colorsys.hsv_to_rgb(hue, 1.0, 1.0)) for hue in hues]

    return rgb_colors




def tensor_or_np_copy(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    else:
        return x.copy()


from pathlib import Path


def export_annotations_and_images(output_dir, original_image, lab, base_name=None):
    from pathlib import Path
    import os
    if os.path.isfile(output_dir):
        base_name = Path(output_dir).stem
        output_dir = Path(output_dir).parent
    else:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            if base_name is None:
                base_name = "image"

    image = np.atleast_3d(np.squeeze(np.array(original_image)))
    image = _move_channel_axis(image)

    if image.shape[0] == 3:

        features = labels_to_features(lab.astype(np.int32), object_type='annotation', include_labels=True,
                                      classification=None)
        geojson = json.dumps(features)
        with open(os.path.join(output_dir, str(base_name) + '_labels.geojson'), "w") as outfile:
            outfile.write(geojson)

        save_image_with_label_overlay(image, lab, output_dir=output_dir, label_boundary_mode='thick', alpha=0.8,
                                      base_name=base_name)

    else:
        warnings.warn("Did not attempt to save image of shape:")
        print(image.shape)


#
# def qupose(image: np.ndarray, path="../models", folder='1509189', preprocessing_flag: bool = True, debug_level: int = 0,
#            num_channels: int = None,
#            to_qupath: bool = False, neven_cleanup: bool = False,
#            device=None,
#            overlap=64,
#            output_dir=Path("../results"),
#            dim_output=4):
#     from utils.model_loader import load_model
#     original_image = tensor_or_np_copy(image)
#     model,_ = load_model(path=path, folder=folder)
#
#     device = _choose_device(device)
#
#     model = model.to(torch.device(device))
#
#     image = np.atleast_3d(np.squeeze(image))
#     assert image.ndim == 3, "Image has too many dimensions" + str(np.squeeze(image).shape)
#
#     num_channels_found = np.min(image.shape)
#     image = _move_channel_axis(image)
#
#     if num_channels != num_channels_found and num_channels is not None:
#         print("Warning: num_channels does not match image shape and model is rigid for the number of channels")
#         if num_channels == 1:
#             print("Warning: Taking a simple mean along the channel axis")
#             image = np.mean(image, axis=0, keepdims=True)
#             num_channels_found = 1
#
#     if preprocessing_flag:
#         image = clip_intensity(image.astype(float))
#         image = _move_channel_axis(image)
#         image = torch.Tensor(image)
#     if debug_level:
#         show_images(image)
#
#     model.eval()
#
#     pred = process_windows(image, shape=(num_channels_found, 256, 256), overlap=overlap, dim_output=dim_output, model=model,
#                            batch_size=2,
#                            pad_mode='constant', merge_mode='triang', on_torch=True, device=device,
#                            neven_cleanup=neven_cleanup)
#
#     if debug_level:
#         show_images(pred)
#     lab = erfnet_postprocessing(pred, alpha=10, mean_filter=False, watershed=True, seed_threshold=-1, mask_threshold=0,
#                                 dist_threshold=0.5, debug=debug_level)
#
#     #  lab=filter_intensity(lab,imgs)
#
#     if to_qupath:
#         output_dir = output_dir / "example_outputs"
#         export_annotations_and_images(output_dir, original_image, lab)
#
#     return pred, lab


def save_image_with_label_overlay(im: np.ndarray,
                                  lab: np.ndarray,
                                  output_dir: str = "./",
                                  base_name: str = 'image',
                                  clip_percentile: float = 1.0,
                                  scale_per_channel: bool = None,
                                  label_boundary_mode=None,
                                  label_colors=None,
                                  alpha=1.0,
                                  thickness=3,
                                  return_image=False):
    """
    Save an image as RGB alongside a corresponding label overlay.
    This can be used to quickly visualize the results of a segmentation, generally using the
    default image viewer of the operating system.

    :param im: Image to save. If this is an 8-bit RGB image (channels-last) it will be used as-is,
               otherwise it will be converted to RGB
    :param lab: Label image to overlay
    :param output_dir: Directory to save to
    :param base_name: Base name for the image files to save
    :param clip_percentile: Percentile to clip the image at. Used during RGB conversion, if needed.
    :param scale_per_channel: Whether to scale each channel independently during RGB conversion.
    :param label_boundary_mode: A boundary mode compatible with scikit-image find_boundaries;
                                one of 'thick', 'inner', 'outer', 'subpixel'. If None, the lab is used directly.
    :param alpha: Alpha value for the underlying image when using the overlay.
                  Setting this less than 1 can help the overlay stand out more prominently.
    """

    import imageio
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries
    from skimage import morphology

    # Check if we have an RGB, channels-last image already
    if im.dtype == np.uint8 and im.ndim == 3 and im.shape[2] == 3:
        im_rgb = im.copy()
    else:
        im_rgb = _to_rgb_channels_last(im, clip_percentile=clip_percentile, scale_per_channel=scale_per_channel)

    # Convert labels to boundaries, if required
    if label_boundary_mode is not None:
        bw_boundaries = find_boundaries(lab, mode=label_boundary_mode)
        lab = lab.copy()
        # Need to expand labels for outer boundaries, but have to avoid overwriting known labels
        if label_boundary_mode in ['thick', 'outer']:
            lab2 = morphology.dilation(lab, footprint=np.ones((thickness, thickness)))
            mask_dilated = bw_boundaries & (lab == 0)

            lab[mask_dilated] = lab2[mask_dilated]
        lab[~bw_boundaries] = 0
        mask_temp = bw_boundaries
    else:
        mask_temp = lab != 0

    # Create a labels displayed
    if label_colors is None:
        lab_overlay = label2rgb(lab).astype(np.float32)
    elif label_colors == "red":
        lab_overlay = np.zeros((lab.shape[0], lab.shape[1], 3))
        lab_overlay[lab > 0, 0] = 1
    elif label_colors == "green":
        lab_overlay = np.zeros((lab.shape[0], lab.shape[1], 3))
        lab_overlay[lab > 0, 1] = 1
    elif label_colors == "blue":
        lab_overlay = np.zeros((lab.shape[0], lab.shape[1], 3))
        lab_overlay[lab > 0, 2] = 1
    else:
        raise Exception("label_colors must be red, green, blue or None")

    im_overlay = im_rgb.copy()
    for c in range(im_overlay.shape[-1]):
        im_temp = im_overlay[..., c]
        lab_temp = lab_overlay[..., c]
        im_temp[mask_temp] = (lab_temp[mask_temp] * 255).astype(np.uint8)

    # Make main image translucent, if needed
    if alpha != 1.0:
        alpha_channel = (mask_temp * 255.0).astype(np.uint8)
        alpha_channel[~mask_temp] = alpha * 255
        im_overlay = np.dstack((im_overlay, alpha_channel))
    if return_image:
        return im_overlay

    if base_name is None:
        base_name = "image"

    print(f"Exporting image to {os.path.join(output_dir, f'{base_name}.png')}")
    imageio.imwrite(os.path.join(output_dir, f'{base_name}.png'), im_rgb)
    imageio.imwrite(os.path.join(output_dir, f'{base_name}_overlay.png'), im_overlay)


def _to_rgb_channels_last(im: np.ndarray,
                          clip_percentile: float = 1.0,
                          scale_per_channel: bool = True,
                          input_channels_first: bool = True) -> np.ndarray:
    """
    Convert an image to RGB, ensuring the output has channels-last ordering.
    """

    if im.ndim < 2 or im.ndim > 3:
        raise ValueError(f"Number of dimensions should be 2 or 3! Image has shape {im.shape}")
    if im.ndim == 3:
        if input_channels_first:
            im = np.moveaxis(im, source=0, destination=-1)
        if im.shape[-1] != 3:
            im = im.mean(axis=-1)
    if im.ndim > 2 and scale_per_channel:
        im_scaled = np.dstack(
            [_to_scaled_uint8(im[..., ii], clip_percentile=clip_percentile) for ii in range(3)]
        )
    else:
        im_scaled = _to_scaled_uint8(im, clip_percentile=clip_percentile)
    if im.ndim == 2:
        im_scaled = np.repeat(im_scaled, repeats=3, axis=-1)
    return im_scaled


def _to_scaled_uint8(im: np.ndarray, clip_percentile=1.0) -> np.ndarray:
    """
    Convert an image to uint8, scaling according to the given percentile.
    """
    im_float = im.astype(np.float32, copy=True)
    min_val = np.percentile(im_float.ravel(), clip_percentile)
    max_val = np.percentile(im_float.ravel(), 100.0 - clip_percentile)
    im_float -= min_val
    im_float /= (max_val - min_val)
    im_float *= 255
    return np.clip(im_float, a_min=0, a_max=255).astype(np.uint8)


def _choose_device(device: str = None, verbose=True) -> str:
    """
    Choose a device to use with PyTorch, given the desired device name.
    If a requested device is not specified or not available, then a default is chosen.
    """
    if device is not None:
        if device == 'cuda' and not torch.has_cuda:
            device = None
            print('CUDA device requested but not available!')
        if device == 'mps' and not torch.has_mps:
            device = None
            print('MPS device requested but not available!')

    if device is None:
        if torch.has_cuda:
            device = 'cuda'
        elif torch.has_mps:
            device = 'mps'
        else:
            device = 'cpu'
        if verbose:
            print(f'Requesting default device: {device}')

    return device


def count_instances(labels: Union[np.ndarray, torch.Tensor]):
    import fastremap
    if isinstance(labels, torch.Tensor):
        num_labels = len(torch.unique(labels[labels > 0]))
    elif isinstance(labels, np.ndarray):
        num_labels = len(fastremap.unique(labels[labels > 0]))
    else:
        raise Exception("Labels must be numpy array or torch tensor")
    return num_labels


def _estimate_image_modality(img, mask):
    """
    This function estimates the modality of an image (i.e. brightfield, chromogenic or fluorescence) based on the
    mean intensity of the pixels inside and outside the mask.
    """

    if isinstance(img, np.ndarray):
        img = np.atleast_3d(_move_channel_axis(img))
        mask = np.squeeze(mask)

        assert mask.ndim == 2, print("Mask must be 2D, but got shape", mask.shape)
        if count_instances(mask) < 1:
            return "Fluorescence"  # The images don't contain any cells, but they look like fluorescence images.
        elif np.mean(img[:, mask > 0]) > np.mean(img[:, mask == 0]):
            return "Fluorescence"
        else:
            if img.shape[0] == 1:
                return "Brightfield"
            else:
                return "Brightfield"  # "Chromogenic"


    elif isinstance(img, torch.Tensor):
        img = torch.at_least_3d(_move_channel_axis(img))
        mask = torch.squeeze(mask)
        assert mask.ndim == 2, "Mask must be 2D"

        if count_instances(mask) < 1:
            return "Fluorescence"  # The images don't contain any cells, but they look like fluorescence images.
        elif torch.mean(img[:, mask > 0]) > torch.mean(img[:, mask == 0]):
            return "Fluorescence"
        else:
            if img.shape[0] == 1:
                return "Brightfield"
            else:
                return "Brightfield"  # "Chromogenic"


def timer(func):
    import functools
    from line_profiler import LineProfiler
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        lp_wrapper(*args, **kwargs)
        lp.print_stats()
        value = func(*args, **kwargs)
        return value

    return wrapper


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from skimage import io

from tqdm import tqdm
import sys

if not '../' in sys.path:
    sys.path.append('../')

from utils.utils import show_images
import torch
import torch.nn as nn
import tifffile


def export_to_torchscript(model_str: str, show_example: bool = False, output_dir: str = None,
                          model_path: str = "../models", torchscript_name: str = None):
    device = 'cpu'
    from utils.model_loader import load_model
    model, model_dict = load_model(folder=model_str, path=model_path)
    model.eval()
    model.to(device)

    cells_and_nuclei = model_dict['cells_and_nuclei']
    pixel_size = model_dict['pixel_size']


    input_data = tifffile.imread("../examples/HE_example.tif")
    #input_data = tifffile.imread("../examples/LuCa1.tif")
    from utils.augmentations import Augmentations
    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(input_data, normalize=False)
    input_tensor, _ = Augmenter.normalize(input_tensor)
    input_tensor, _ = Augmenter.torch_rescale(input_tensor, current_pixel_size=0.5, requested_pixel_size=pixel_size, crop =True)
    input_tensor = input_tensor.to(device)



    from utils.loss.instanseg_loss import InstanSeg_Torchscript
    super_model = InstanSeg_Torchscript(model, cells_and_nuclei=cells_and_nuclei, pixel_size = pixel_size).to(device)
    out = super_model(input_tensor[None,])

    if show_example:
        show_images([input_tensor] + [i for i in out.squeeze(0)], labels=[i + 1 for i in range(len(out.squeeze(0)))])

    with torch.jit.optimized_execution(should_optimize=True):
        traced_cpu = torch.jit.script(super_model, input_tensor[None,])

    if torchscript_name is None:
        torchscript_name = model_str

    if output_dir is None:
        torch.jit.save(traced_cpu, "../examples/torchscript_models/instanseg_" + torchscript_name + ".pt")
        print("Saved torchscript model to ../examples/torchscript_models/instanseg_" + torchscript_name + ".pt")

    else:
        torch.jit.save(traced_cpu, os.path.join(output_dir, "instanseg_" + torchscript_name + ".pt"))
        print("Saved torchscript model to", os.path.join(output_dir, "instanseg_" + torchscript_name + ".pt"))


def drag_and_drop_file():
    """
    This opens a window where a user can drop a file and returns the path to the file
    """

    import tkinter as tk
    from tkinterdnd2 import TkinterDnD, DND_FILES

    def drop(event):
        file_path = event.data
        entry_var.set(file_path)

    def save_and_close():
        entry_var.get()
        root.destroy()  # Close the window

    root = TkinterDnD.Tk()
    entry_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=entry_var, width=40)
    entry.pack(pady=20)

    entry.drop_target_register(DND_FILES)
    entry.dnd_bind('<<Drop>>', drop)

    save_button = tk.Button(root, text="Save and Close", command=save_and_close)
    save_button.pack(pady=10)
    root.mainloop()
    return entry_var.get()
