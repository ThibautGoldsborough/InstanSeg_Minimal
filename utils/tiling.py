import torch
import numpy as np
import torch.nn.functional as F

import pdb

global gt

import sys

# if not '../' in sys.path:
sys.path[0] = '../'

from utils.utils import show_images
from utils.pytorch_utils import torch_sparse_onehot, fast_sparse_dual_iou, torch_onehot, fast_dual_iou, torch_fastremap, \
    remap_values


def edge_mask(labels, ignore=[None]):
    labels = labels.squeeze()
    first_row = labels[0, :]
    last_row = labels[-1, :]
    first_column = labels[:, 0]
    last_column = labels[:, -1]

    edges = []
    if 'top' not in ignore:
        edges.append(first_row)
    if 'bottom' not in ignore:
        edges.append(last_row)
    if 'left' not in ignore:
        edges.append(first_column)
    if 'right' not in ignore:
        edges.append(last_column)

    edges = torch.cat(edges, dim=0)
    return torch.isin(labels, edges[edges > 0])


def remove_edge_labels(labels, ignore=[None]):
    return labels * ~edge_mask(labels, ignore=ignore)


def _to_shape(a, shape):
    """Pad a tensor to a given shape."""
    if len(a.shape) == 2:
        a = a.unsqueeze(0)
    y_, x_ = shape
    y, x = a[0].shape[-2:]
    y_pad = max(0, y_ - y)
    x_pad = max(0, x_ - x)
    return torch.nn.functional.pad(a, (x_pad // 2, x_pad // 2 + x_pad % 2, y_pad // 2, y_pad // 2 + y_pad % 2))


def _to_shape_bottom_left(a, shape):
    """Pad a tensor to a given shape."""
    if len(a.shape) == 2:
        a = a.unsqueeze(0)
    y_, x_ = shape
    y, x = a[0].shape[-2:]
    y_pad = max(0, y_ - y)
    x_pad = max(0, x_ - x)
    return torch.nn.functional.pad(a, (0, x_pad, 0, y_pad))


def chops(img: torch.Tensor, shape: tuple, overlap: float = 0.5) -> tuple:
    """This function splits an image into desired windows and returns the indices of the windows"""

    if len(img.shape) != 2:
        img = img[0]
    if (torch.tensor(img.shape) < torch.tensor(shape)).any():
        return [0], [0]
    h, v = img.shape[-2:]
    stride_h = shape[0] - int(overlap * shape[0])
    stride_v = shape[1] - int(overlap * shape[1])

    #  print([i * stride_v for i in range(v // stride_v)] + [v - stride_v])
    v_index = np.unique([i * stride_v for i in range(v // stride_v)] + [v - shape[1]])
    h_index = np.unique([i * stride_h for i in range(h // stride_h)] + [h - shape[0]])

    #  pdb.set_trace()

    #   print(h_index,v_index)
    return h_index, v_index


def tiles_from_chops(image: torch.Tensor, shape: tuple, tuple_index: tuple) -> list:
    """This function takes an image, a shape, and a tuple of window indices (e.g., outputed by the function chops)
    and returns a list of windows"""
    h_index, v_index = tuple_index

    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    # if (torch.tensor(image.shape[-2:]) < torch.tensor(shape)).any():
    #  image = _to_shape(image, shape)
    stride_h = shape[0]
    stride_v = shape[1]
    tile_list = []
    for i, window_i in enumerate(h_index):
        for j, window_j in enumerate(v_index):
            current_window = image[..., window_i:window_i + stride_h, window_j:window_j + stride_v]
            tile_list.append(current_window)
    return tile_list


def stitch(tiles: list, shape: tuple, chop_list: list, final_shape: tuple, line_thickness: int = 1):
    """This function takes a list of tiles, a shape, and a tuple of window indices (e.g., outputed by the function chops)
    and returns a stitched image"""

    canvas = torch.zeros(final_shape, dtype=torch.int32)
    visited = torch.zeros(final_shape, dtype=torch.uint8)
    problematic = torch.zeros(final_shape, dtype=torch.uint8)

    for i, window_i in enumerate(chop_list[0]):
        for j, window_j in enumerate(chop_list[1]):
            new_tile = tiles[i * len(chop_list[1]) + j].squeeze()

            max_count = canvas.max()

            new_tile = (new_tile + max_count) * (new_tile > 0).int()

            new_problematic_ids = torch.unique(new_tile[problematic[..., window_i:window_i + shape[0],
                                                        window_j:window_j + shape[1]].squeeze(0) >= 1].int())
            new_problematic_ids = new_problematic_ids[new_problematic_ids > 0]

            old_problematic_ids = torch.unique(canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]][
                                                   problematic[..., window_i:window_i + shape[0],
                                                   window_j:window_j + shape[1]] >= 1])
            old_problematic_ids = old_problematic_ids[old_problematic_ids > 0]

            old_lab = canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]]
            new_lab = new_tile

            old_lab[torch.isin(old_lab, old_problematic_ids)] = 0

            mask_new_lab = visited[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] == 0  # unvisited
            mask_new_lab = torch.logical_or(torch.isin(new_lab, new_problematic_ids), mask_new_lab)
            new_lab = new_lab * mask_new_lab.int()

            canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]][
                canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] == 0] += new_lab[
                canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] == 0].int()

            problematic[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] = 0

            problematic[..., window_i + shape[0] - line_thickness:window_i + shape[0],
            window_j + 0:window_j + shape[1]] += 1
            problematic[..., window_i + 10:window_i + shape[0],
            window_j + shape[1] - line_thickness:window_j + shape[1]] += 1

            visited[..., window_i:window_i + shape[0] - line_thickness,
            window_j:window_j + shape[1] - line_thickness] += 1

        #  show_images(old_lab,new_lab)

    return canvas


def stitch_iou(tiles: list, shape: tuple, chop_list: list, final_shape: tuple):
    """This function takes a list of tiles, a shape, and a tuple of window indices (e.g., output of the function chops)
    and returns a stitched image"""

    canvas = torch.zeros(final_shape, dtype=torch.int32)
    visited = torch.zeros(final_shape, dtype=torch.uint8)

    overlap_x = None
    overlap_y = None

    for i, window_i in enumerate(chop_list[0]):
        for j, window_j in enumerate(chop_list[1]):

            old_lab = canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]].squeeze()
            new_lab = tiles[i * len(chop_list[1]) + j].squeeze()
            max_count = canvas.max()
            new_lab = (new_lab + max_count) * (new_lab > 0).int()
            gt_lab = gt[..., window_i:window_i + shape[0], window_j:window_j + shape[1]]  ##

            #   print(i,j)

            if j >= 1:

                overlap_y = shape[1] - (window_j - chop_list[1][j - 1])
                overlap_y = 128
                #  print(overlap_y,shape[1],window_j,chop_list[1][j-1])
                #     if j != len(chop_list[1])-1:
                cut_labels_r = torch.unique(old_lab[..., :, overlap_y - 1][old_lab[..., :, overlap_y - 1] > 0])
                cut_labels_l = torch.unique(old_lab[..., :, 0][old_lab[..., :, 0] > 0])
                cut_labels_overlap = torch.unique(
                    old_lab[..., :, 0:overlap_y - 1][old_lab[..., :, 0:overlap_y - 1] > 0])
                cut_labels_overlap_strict = cut_labels_overlap[~torch.isin(cut_labels_overlap, cut_labels_l)]

                if len(cut_labels_r) >= 1 and len(cut_labels_l) >= 1:
                    objects_to_remove = cut_labels_r[~torch.isin(cut_labels_r, cut_labels_l)]

                    old_lab[torch.isin(old_lab, objects_to_remove)] = 0

                new_cut_labels_l = torch.unique(new_lab[..., :, 0][new_lab[..., :, 0] > 0])
                if len(new_cut_labels_l) >= 1:
                    new_lab[torch.isin(new_lab, new_cut_labels_l)] = 0

                    new_lab[torch.isin(old_lab, cut_labels_overlap_strict)] = old_lab[
                        torch.isin(old_lab, cut_labels_overlap_strict)].float()

                    show_images(old_lab, new_lab)

            if i >= 1:
                overlap_x = shape[0] - (window_i - chop_list[0][i - 1])
                overlap_x = 128
                # print(overlap_x,shape[0],window_i,chop_list[0][i-1])
                #    if i != len(chop_list[0])-1:
                cut_labels_b = torch.unique(old_lab[..., overlap_x - 1, :][old_lab[..., overlap_x - 1, :] > 0])
                cut_labels_t = torch.unique(old_lab[..., 0, :][old_lab[..., 0, :] > 0])
                cut_labels_overlap = torch.unique(
                    old_lab[..., 0:overlap_x - 1, :][old_lab[..., 0:overlap_x - 1, :] > 0])
                cut_labels_overlap_strict = cut_labels_overlap[~torch.isin(cut_labels_overlap, cut_labels_b)]

                if len(cut_labels_b) >= 1 and len(cut_labels_t) >= 1:
                    objects_to_remove = cut_labels_b[~torch.isin(cut_labels_b, cut_labels_t)]
                    old_lab[torch.isin(old_lab, objects_to_remove)] = 0
                new_cut_labels_t = torch.unique(new_lab[..., 0, :][new_lab[..., 0, :] > 0])

                # print(new_cut_labels_t)
                if len(new_cut_labels_t) >= 1:
                    #  show_images(new_lab,old_lab)
                    new_lab[torch.isin(new_lab, new_cut_labels_t)] = 0
                    new_lab[torch.isin(old_lab, cut_labels_overlap_strict)] = old_lab[
                        torch.isin(old_lab, cut_labels_overlap_strict)].float()
                # show_images(new_lab,old_lab)

            #  show_images(canvas,gt,(canvas>0).float()-(gt>0).float())

            #  old_lab = canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]]

            #  new_lab = new_tile
            #  show_images(old_lab,new_lab,gt_lab, new_lab - gt_lab)
            #  pdb.set_trace()

            previously_visited = visited[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] > 0

            old_problematic = old_lab * previously_visited
            new_problematic = new_lab * previously_visited

            #   x1=torch_fastremap(old_lab * previously_visited)
            #    x2=torch_fastremap(new_lab * previously_visited)

            old_problematic_onehot, old_unique_values = torch_sparse_onehot(old_problematic, flatten=True)
            new_problematic_onehot, new_unique_values = torch_sparse_onehot(new_problematic, flatten=True)

            iou = fast_sparse_dual_iou(old_problematic_onehot, new_problematic_onehot)

            onehot_remapping = torch.nonzero(iou > 0.5).T + 1

            #
            #   pdb.set_trace()

            if onehot_remapping.shape[1] > 0:
                final_remapping = torch.stack(
                    (new_unique_values[onehot_remapping[1]], old_unique_values[onehot_remapping[0]]))
                mask = torch.isin(new_lab, final_remapping[0])

                try:
                    new_lab[mask] = remap_values(final_remapping, new_lab[mask])
                except:
                    pdb.set_trace()
            #  show_images(old_lab,new_lab,canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]])

            #    cut_labels = edge_mask(old_lab,ignore = ["bottom","top","right"])

            #  new_lab = old_problematic[0].float()

            #       pdb.set_trace()

            canvas[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] = new_lab.int()

            # show_images(canvas)

            #   problematic[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] = 0

            #  problematic[..., window_i + shape[0] - line_thickness :window_i + shape[0] , window_j + 0 :window_j + shape[1]  ] += 1
            #   problematic[...,  window_i +10 :window_i + shape[0] , window_j + shape[1] -line_thickness :window_j + shape[1] ] += 1

            visited[..., window_i:window_i + shape[0], window_j:window_j + shape[1]] += 1

            show_images(old_lab, new_lab, gt_lab, new_lab - gt_lab)

            show_images(canvas, gt, (canvas > 0).float() - (gt > 0).float())

    return canvas


def instanseg_padding(img: torch.Tensor, extra_pad: int = 96):
    min_dim = 16

    padx = min_dim * torch.ceil(torch.tensor((img.shape[-2] / min_dim))).int() - img.shape[-2] + extra_pad * 2
    pady = min_dim * torch.ceil(torch.tensor((img.shape[-1] / min_dim))).int() - img.shape[-1] + extra_pad * 2

    if padx > img.shape[-2]:
        padx = padx - extra_pad
    if pady > img.shape[-1]:
        pady = pady - extra_pad

    img = torch.functional.F.pad(img[None], [0, int(pady), 0, int(padx)], mode='reflect')[
        0]  # The order of these is a mystery to me.

    return img, torch.stack((padx, pady))


def recover_padding(x: torch.Tensor, pad: torch.Tensor):
    # x must be 1,C,H,W or C,H,W
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True

    if pad[0] == 0:
        pad[0] = -x.shape[2]
    if pad[1] == 0:
        pad[1] = -x.shape[3]

    if squeeze:
        return x[:, :, :-pad[0], :-pad[1]].squeeze(0)
    else:
        return x[:, :, :-pad[0], :-pad[1]]


from utils.utils import timer
from tqdm import tqdm


# @timer
def sliding_window_inference(input_tensor, predictor, window_size=(512, 512), overlap_size=0.1, sw_device='cuda',
                             device='cpu', output_channels=1):
    input_tensor = input_tensor.to(device)
    predictor = predictor.to(sw_device)

    tuple_index = chops(input_tensor, shape=window_size, overlap=overlap_size)
    tile_list = tiles_from_chops(input_tensor, shape=window_size, tuple_index=tuple_index)
    label_list = [predictor(tile[None].to(sw_device)).squeeze(0).to(device) for tile in tqdm(tile_list)]

    lab = torch.cat([stitch([lab[i] for lab in label_list], shape=window_size, chop_list=tuple_index,
                            final_shape=(1, input_tensor.shape[1], input_tensor.shape[2])) for i in
                     range(output_channels)], dim=0)

    return lab[None]  # 1,C,H,W


if __name__ == "__main__":
    from tqdm import tqdm
    import sys

    from utils.utils import show_images, _choose_device
    import torch

    from skimage import io

    import time

    instanseg = torch.jit.load("../examples/torchscript_models/instanseg_1735176.pt")
    device = 'cpu'
    instanseg.to(device)

    input_data=io.imread("../examples/LuCa1.tif")
  #  input_data = io.imread("../examples/HE_example.tif")
    from utils.augmentations import Augmentations

    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(input_data, normalize=True)
    # out=instanseg(input_tensor[None,].to(device)[:,:,:500,:1000])

    input_tensor = input_tensor[:, 0:512, 0:512]

    rgb = Augmenter.colourize(input_tensor, c_nuclei=6, metadata={"image_modality": "Fluorescence"})[0]

    device = "cpu"
    instanseg.to(device)
    gt = instanseg(input_tensor[None].to(device))[0, 0].cpu()

    device = _choose_device()
    instanseg.to(device)

    start = time.time()

    lab = sliding_window_inference(input_tensor, instanseg, window_size=(256, 256), overlap_size=32 / 256,
                                   sw_device=device, device='cpu', output_channels=1)
    # la2 = sliding_window_inference(input_tensor,instanseg, window_size = (256,256),overlap_size = 128/1024,sw_device = device,device = 'cpu', output_channels = 1)

    show_images(lab, gt, (lab > 0).int() - (gt > 0).int(), la2, gt, (la2 > 0).int() - (gt > 0).int(),
                labels=[0, 1, 3, 4], titles=["Tiled", "One Pass", "Difference", "Tiled", "One Pass", "Difference"],
                n_cols=3)

    end = time.time()

    print("Time for dual channel output sliding window: ", end - start)

    start = time.time()

    tuple_index = chops(input_tensor, shape=(256, 256), overlap=0.2)
    tile_list = tiles_from_chops(input_tensor, shape=(256, 256), tuple_index=tuple_index)
    label_list = [instanseg(tile[None].to(device))[0, 0].cpu() for tile in tile_list]
    lab = stitch(label_list, shape=(256, 256), chop_list=tuple_index,
                 final_shape=(1, input_tensor.shape[1], input_tensor.shape[2]))

    end = time.time()

    print("Time for 256x256 tiling: ", end - start)

    start = time.time()

    tuple_index = chops(input_tensor, shape=(512, 512), overlap=0.2)
    tile_list = tiles_from_chops(input_tensor, shape=(512, 512), tuple_index=tuple_index)
    label_list = [instanseg(tile[None].to(device))[0, 0].cpu() for tile in tile_list]
    lab2 = stitch(label_list, shape=(512, 512), chop_list=tuple_index,
                  final_shape=(1, input_tensor.shape[1], input_tensor.shape[2]))

    end = time.time()
    print("Time for 512x512 tiling: ", end - start)

    show_images(lab, gt, (lab > 0).int() - (gt > 0).int(), labels=[0, 1], titles=["Tiled", "One Pass", "Difference"])
    show_images(lab, lab2, (lab > 0).int() - (lab2 > 0).int(), rgb, labels=[0, 1],
                titles=["Tile 256", "Tile 512", "Difference", "original"])

    start = time.time()

    lab = sliding_window_inference(input_tensor, instanseg, window_size=(512, 512), overlap_size=0.2, sw_device='cpu',
                                   device='cpu', output_channels=1)

    end = time.time()

    print("Time for dual channel output sliding window: ", end - start)

# show_images([i for i in lab]+[i for i in lab],labels=[0,1])


# pdb.set_trace()
