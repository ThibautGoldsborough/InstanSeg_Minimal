from typing import Tuple, Any

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

sys.path[0] = '../'
import pdb
from utils.pytorch_utils import torch_sparse_onehot, torch_fastremap, remap_values
from utils.utils import show_images


def get_intersection_over_nucleus_area(label: torch.Tensor) -> tuple[Any, Any]:
    """
    Returns the intersection over nucleus area in a 2 channel labeled image
     
    label must be a 1,2,H,W tensor where the first channel is nuclei and the second is whole cell
    """
    nuclei_onehot = torch_sparse_onehot(label[0, 0], flatten=True)[0]
    cell_onehot = torch_sparse_onehot(label[0, 1], flatten=True)[0]
    intersection = torch.sparse.mm(nuclei_onehot, cell_onehot.T).to_dense()
    sparse_sum1 = torch.sparse.sum(nuclei_onehot, dim=(1,))[None].to_dense()
    nuclei_area = sparse_sum1.T

    return (intersection / nuclei_area), nuclei_area


def get_nonnucleated_cell_ids(iou: torch.Tensor, lab: torch.Tensor, threshold: float = 0.5, return_lab: bool = True):
    iou = iou > threshold
    lab_ids = torch.unique(lab[lab > 0])
    nonnucleated = (iou.sum(0)) == 0
    if return_lab:
        return lab_ids[nonnucleated], lab * torch.isin(lab, lab_ids[nonnucleated]), lab * torch.isin(lab, lab_ids[
            ~nonnucleated])

    return lab_ids[nonnucleated]


def get_nucleated_cell_ids(iou: torch.Tensor, lab: torch.Tensor, threshold: float = 0.5, return_lab: bool = True):
    iou = iou > threshold
    lab_ids = torch.unique(lab[lab > 0])
    nucleated = (iou.sum(0)) >= 1

    if return_lab:
        return lab_ids[nucleated], lab * torch.isin(lab, lab_ids[nucleated]), lab * torch.isin(lab, lab_ids[~nucleated])
    return lab_ids[nucleated]


def get_multinucleated_cell_ids(iou: torch.Tensor, lab: torch.Tensor, threshold: float = 0.5, return_lab: bool = True):
    iou = iou > threshold
    lab_ids = torch.unique(lab[lab > 0])
    multinucleated = (iou.sum(0)) > 1

    if return_lab:
        return lab_ids[multinucleated], lab * torch.isin(lab, lab_ids[multinucleated]), lab * torch.isin(lab, lab_ids[
            ~multinucleated])
    return lab_ids[multinucleated]


def keep_only_largest_nucleus_per_cell(labels: torch.Tensor, return_lab: bool = True):
    """
    labels: tensor of shape 1,2,H,W containing nucleus and cell labels respectively
    return_lab: if True, returns the labels with only the largest nucleus per cell, and only cells that have a nucleus.
    """
    labels = torch_fastremap(labels)
    iou, nuclei_area = get_intersection_over_nucleus_area(labels)
    iou_biggest_area = ((iou > 0.5).float() * nuclei_area) == (((iou > 0.5).float() * nuclei_area).max(0)[0])
    iou_biggest_area = ((iou_biggest_area.float() * iou) > 0.5)
    nuclei_ids = torch.unique(labels[0, 0][labels[0, 0] > 0])
    cell_ids = torch.unique(labels[0, 1][labels[0, 1] > 0])
    largest_nucleus = (iou_biggest_area.sum(1)) == 1
    nucleated_cells = ((iou > 0.5).float().sum(0)) >= 1
    if return_lab:
        return nuclei_ids[largest_nucleus], torch.stack(
            (labels[0, 0] * torch.isin(labels[0, 0], nuclei_ids[largest_nucleus]),
             labels[0, 1] * torch.isin(labels[0, 1], cell_ids[nucleated_cells]))).unsqueeze(0)
    return nuclei_ids[largest_nucleus]


def resolve_cell_and_nucleus_boundaries(lab: torch.Tensor) -> torch.Tensor:
    """
    lab: tensor of shape 1,2,H,W containing nucleus and cell labels respectively

    returns: tensor of the same shape as lab

    This function will resolve the boundaries between cells and nuclei. 
    It will first match the labels of the largest nucleus and its cell.
    It will then erase from the cell masks all the nuclei pixels. This resolves nuclei "just" overlapping adjacent cell.
    It will then recover the nuclei pixels that were erased by adding them back to the cell masks.

    Currently, this will remove all cells that don't have a nucleus.

    """

    original_nuclei_labels = lab[0, 0].clone()
    # original_cell_labels = lab[0,0].clone()


    lab = torch.stack((torch_fastremap(lab[0, 0]), torch_fastremap(lab[0, 1]))).unsqueeze(
        0)  # just relabel the nuclei and cells from 1 to N

    original_nuclei_labels = lab[0, 0].clone()

    iou, areas = get_intersection_over_nucleus_area(lab)

    ids, lab = keep_only_largest_nucleus_per_cell(lab, return_lab=True)

    lab = torch.stack((torch_fastremap(lab[0, 0]), torch_fastremap(lab[0, 1]))).unsqueeze(0)

    clean_lab = lab  # This lab will have a one-to-one mapping between nuclei and cells.



    iou, areas = get_intersection_over_nucleus_area(clean_lab)

    onehot_remapping = (torch.nonzero(iou > 0.5).T + 1).flip(0)

    remapping = torch.cat((torch.zeros(2, 1, device=onehot_remapping.device), onehot_remapping), dim=1)

    clean_lab[0, 1] = remap_values(remapping,
                                   clean_lab[0, 1]).int()  # Every matching cell and nucleus now have the same label.

    nuclei_labels = clean_lab[0, 0]
    cell_labels = clean_lab[0, 1]

    original_nuclei_labels[nuclei_labels > 0] = 0
    original_nuclei_labels = torch_fastremap(original_nuclei_labels)
    original_nuclei_labels[original_nuclei_labels > 0] = original_nuclei_labels[
                                                             original_nuclei_labels > 0] + nuclei_labels.max() + 1
    nuclei_labels += original_nuclei_labels

    cell_labels[nuclei_labels > 0] = 0
    cell_labels += nuclei_labels

    lab = torch.stack((nuclei_labels, cell_labels)).unsqueeze(0)


    return lab


def get_mean_object_features(image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # image is C,H,W
    # label is H,W
    # returns a tensor of size N,C for N objects and C channels
    label = label.squeeze()
    from utils.pytorch_utils import torch_sparse_onehot
    sparse_onehot = torch_sparse_onehot(label, flatten=True)[0]
    out = torch.mm(sparse_onehot, image.flatten(1).T)  # object features
    sums = torch.sparse.sum(sparse_onehot, dim=1).to_dense()  # object areas
    out = out / sums[None].T  # mean object features
    return out


def get_features_by_location(input_tensor: torch.Tensor, lab: torch.Tensor) -> tuple:
    # input tensor is C,H,W
    # lab is 1,2,H,W where the first channel is nuclei and the second is whole cell

    cell_features = get_mean_object_features(input_tensor, lab[0, 1])
    nuclei_features = get_mean_object_features(input_tensor, lab[0, 0])

    cytoplasm_lab = (lab[0, 0] == 0).float() * lab[0, 1]

    nuclei_features = get_mean_object_features(input_tensor, lab[0, 0])

    cytoplasm_features = get_mean_object_features(input_tensor, cytoplasm_lab)

    X_cell = cell_features.numpy()
    X_nuclei = nuclei_features.numpy()
    X_cytoplasm = cytoplasm_features.numpy()

    return X_cell, X_nuclei, X_cytoplasm


def violin_plot_feature_location(X_nuclei: np.ndarray, X_cytoplasm: np.ndarray):
    df = pd.DataFrame(columns=['Location', 'Channel', 'Value'])

    for i in range(X_nuclei.shape[1]):
        # Assuming you have three arrays: array1, array2, and array3
        # Replace these with your actual data
        array1 = X_nuclei[:, i]
        array2 = X_cytoplasm[:, i]

        # Create a DataFrame with the arrays
        df = pd.concat([df, pd.DataFrame({
            'Value': np.concatenate([array1.flatten(), array2.flatten()]),
            'Location': np.repeat(['Nuclei', 'Cytoplasm'], [len(array1), len(array2)]),
            'Channel': np.repeat([str(i), str(i)], [len(array1), len(array2)])
        })], ignore_index=True)

        # Print the DataFrame

    plt.figure(figsize=(20, 10))
    sns.violinplot(data=df, x="Channel", y="Value", hue="Location", split=True, inner="quart", cut=0)
    plt.show()


def show_umap_and_cluster(X_features):
    import scanpy as sc
    # Create a Scanpy AnnData object
    adata = sc.AnnData(X_features)

    # Z-normalise the data
    sc.pp.scale(adata)

    # Perform PCA and UMAP
    # sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.umap(adata)

    # Perform Louvain clustering
    sc.tl.louvain(adata, resolution=0.2)

    # Plot the UMAP result with Louvain clusters
    sc.pl.umap(adata, color='louvain', palette='viridis', legend_loc='on data', title='UMAP with Louvain Clustering')

    # Plot the UMAP result with Louvain clusters
    plt.show()


if __name__ == "__main__":
    from skimage import io
    from utils.tiling import sliding_window_inference
    from utils.pytorch_utils import torch_sparse_onehot, fast_sparse_dual_iou
    from utils.utils import _choose_device
    from utils.utils import drag_and_drop_file
    from pathlib import Path

    instanseg = torch.jit.load("../examples/torchscript_models/instanseg_1735176.pt")
    device = _choose_device()
    instanseg.to(device)
    input_data = io.imread("../examples/LuCa1.tif")
   # input_data = io.imread(Path(drag_and_drop_file()))
    from utils.augmentations import Augmentations

    Augmenter = Augmentations()
  #  input_tensor, _ = Augmenter.to_tensor(input_data, normalize=True)
    input_tensor,_ =Augmenter.to_tensor(input_data,normalize=False) #this converts the input data to a tensor and does percentile normalization (no clipping)
    input_tensor,_ = Augmenter.normalize(input_tensor)

    print("Running InstanSeg ...")

    lab = sliding_window_inference(input_tensor, instanseg, window_size=(512, 512), overlap_size=32 / 256,
                                   sw_device=device, device='cpu', output_channels=2)


    print("Calculating cellular features ...")
    X_cell, X_nuclei, X_cytoplasm = get_features_by_location(input_tensor, lab)

    print("Plotting violing plots ...")

    violin_plot_feature_location(X_nuclei, X_cytoplasm)

    print("Clustering and umap ...")

    show_umap_and_cluster(X_cell)

    print("Done !")
