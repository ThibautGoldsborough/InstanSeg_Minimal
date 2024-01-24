import torch
import torch.nn.functional as F
import monai


def remap_values(remapping: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    This remaps the values in x according to the pairs in the remapping tensor.
    remapping: 2,N      Make sure the remapping is 1 to 1, and there are no loops (i.e. 1->2, 2->3, 3->1). Loops can be removed using graph based connected components algorithms (see instanseg postprocessing for an example)
    x: any shape
    """
    sorted_remapping = remapping[:, remapping[0].argsort()]
    index = torch.bucketize(x.ravel(), sorted_remapping[0])
    return sorted_remapping[1][index].reshape(x.shape)


def torch_fastremap(x: torch.Tensor):
    unique_values = torch.unique(x, sorted=True)
    new_values = torch.arange(len(unique_values), dtype=x.dtype, device=x.device)
    remapping = torch.stack((unique_values, new_values))
    return remap_values(remapping, x)


def torch_onehot(x: torch.Tensor) -> torch.Tensor:
    # x is a labeled image of shape _,_,H,W returns a onehot encoding of shape 1,C,H,W
    H, W = x.shape[-2:]
    x = x.view(-1, 1, H, W)
    x = x.squeeze().view(1, 1, H, W)
    unique = torch.unique(x[x > 0])
    x = x.repeat(1, len(unique), 1, 1)
    return x == unique.unsqueeze(-1).unsqueeze(-1)


def fast_iou(onehot: torch.Tensor, threshold: float = 0.5):
    # onehot is C,H,W
    if onehot.ndim == 3:
        onehot = onehot.flatten(1)
    onehot = (onehot > threshold).float()
    intersection = onehot @ onehot.T
    union = onehot.sum(1)[None].T + onehot.sum(1)[None] - intersection
    return intersection / union


def fast_sparse_iou(sparse_onehot: torch.Tensor):
    intersection = torch.sparse.mm(sparse_onehot, sparse_onehot.T).to_dense()
    sparse_sum = torch.sparse.sum(sparse_onehot, dim=(1,))[None].to_dense()
    union = sparse_sum.T + sparse_sum - intersection
    return intersection / union

def instance_wise_edt(x):
    x = torch_onehot(x)
    x = monai.transforms.utils.distance_transform_edt(x[0])
    x = x / (x.flatten(1).max(1)[0]).view(-1,1,1)
    x = x.sum(0)
    return x


def fast_dual_iou(onehot1: torch.Tensor, onehot2: torch.Tensor) -> torch.Tensor:
    """
    Returns the intersection over union between two dense onehot encoded tensors
    """
    # onehot1 and onehot2 are C1,H,W and C2,H,W

    C1 = onehot1.shape[0]
    C2 = onehot2.shape[0]

    max_C = max(C1, C2)

    onehot1 = torch.cat((onehot1, torch.zeros((max_C - C1, *onehot1.shape[1:]))), dim=0)
    onehot2 = torch.cat((onehot2, torch.zeros((max_C - C2, *onehot2.shape[1:]))), dim=0)

    onehot1 = onehot1.flatten(1)
    onehot1 = (onehot1 > 0.5).float()  # onehot should be binary

    onehot2 = onehot2.flatten(1)
    onehot2 = (onehot2 > 0.5).float()

    intersection = onehot1 @ onehot2.T
    union = (onehot1).sum(1)[None].T + (onehot2).sum(1)[None] - intersection

    return (intersection / union)[:C1, :C2]


def torch_sparse_onehot(x: torch.Tensor, flatten: bool = False) -> torch.Tensor:
    # x is a labeled image of shape _,_,H,W returns a sparse tensor of shape C,H,W
    unique_values = torch.unique(x, sorted=True)
    x = torch_fastremap(x)

    H, W = x.shape[-2:]

    if flatten:
        x = x.view(H * W)
        xxyy = torch.nonzero(x > 0).squeeze()
        zz = x[xxyy] - 1
        C = x.max().int()
        sparse_onehot = torch.sparse_coo_tensor(torch.stack((zz, xxyy)), (torch.ones_like(xxyy).float()),
                                                size=(C, H * W), dtype=torch.float32)

    else:
        x = x.squeeze().view(H, W)
        xx, yy = torch.nonzero(x > 0).T
        zz = x[xx, yy] - 1
        C = x.max().int()
        sparse_onehot = torch.sparse_coo_tensor(torch.stack((zz, xx, yy)), (torch.ones_like(xx).float()),
                                                size=(C, H, W), dtype=torch.float32)

    return sparse_onehot, unique_values


import pdb


def fast_sparse_dual_iou(onehot1: torch.Tensor, onehot2: torch.Tensor) -> torch.Tensor:
    """
    Returns the intersection over union between two sparse onehot encoded tensors
    """
    # onehot1 and onehot2 are C1,H*W and C2,H*W

    intersection = torch.sparse.mm(onehot1, onehot2.T).to_dense()
    sparse_sum1 = torch.sparse.sum(onehot1, dim=(1,))[None].to_dense()
    sparse_sum2 = torch.sparse.sum(onehot2, dim=(1,))[None].to_dense()
    union = sparse_sum1.T + sparse_sum2 - intersection

    return (intersection / union)


def iou_test():
    """
    Unit test for the fast dual iou functions
    """
    out = torch.randint(0, 50, (1, 2, 124, 256), dtype=torch.float32)
    onehot1 = torch_onehot(out[0, 0])[0]
    onehot2 = torch_onehot(out[0, 1])[0]
    iou_dense = fast_dual_iou(onehot1, onehot2)

    onehot1 = torch_sparse_onehot(out[0, 0], flatten=True)[0]
    onehot2 = torch_sparse_onehot(out[0, 1], flatten=True)[0]
    iou_sparse = fast_sparse_dual_iou(onehot1, onehot2)

    assert torch.allclose(iou_dense, iou_sparse)


def remove_small_fragments(image: torch.Tensor, max_fragment_size: int = 100, num_repeats: int = 3,
                           return_fragments: bool = False):
    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W)  #

    mask = image_view > 0

    num_iterations = max_fragment_size + 1

    out_masks = []

    for i in range(num_repeats):

        out_c = torch.randperm(H * W, device=image.device, dtype=torch.float).view((-1, 1, H, W))
        out_c = torch.mul(out_c, mask)
        second_order_difference = torch.zeros_like(out_c)
        second_order_unique = torch.zeros_like(out_c[0])
        second_order_counts = torch.zeros_like(out_c[0])
        first_order_unique = torch.zeros_like(out_c[0])
        first_order_counts = torch.zeros_like(out_c[0])
        constant_ids = torch.zeros_like(out_c[0])

        for i in range(num_iterations):
            out_c = F.max_pool2d(out_c, kernel_size=3, stride=1, padding=1)
            out_c = torch.mul(out_c, mask)

            if i >= num_iterations - 3:
                candidate_mask = (((second_order_difference == out_c).sum(0) * mask) == 1)
                unique_list, counts = torch.unique(out_c.flatten(1)[:, candidate_mask.flatten() > 0],
                                                   return_counts=True, sorted=True)

                first_order_unique = unique_list[counts < max_fragment_size]
                first_order_counts = counts[counts < max_fragment_size]

                match_second_order_unique = second_order_unique[torch.isin(second_order_unique, first_order_unique)]
                match_second_order_counts = second_order_counts[torch.isin(second_order_unique, first_order_unique)]

                match_first_order_unique = first_order_unique[torch.isin(first_order_unique, second_order_unique)]

                assert match_first_order_unique.shape == match_second_order_unique.shape  # sanity check
                match_first_order_counts = first_order_counts[torch.isin(first_order_unique, second_order_unique)]

                constant_ids = match_second_order_unique[match_second_order_counts == match_first_order_counts]

            if i == num_iterations - 1:
                out_masks.append(torch.isin(out_c, constant_ids))
                break

            if i >= num_iterations - 3:
                second_order_difference = out_c
                second_order_unique = first_order_unique
                second_order_counts = first_order_counts

    out = torch.stack(out_masks).min(0)[0]

    if return_fragments:
        return out
    return torch.mul(image_view, ~out).view_as(image)


def fill_holes(image: torch.Tensor, max_hole_size: int = 10, num_repeats: int = 3, return_holes: bool = False):
    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W).float()
    mask = (image_view == 0).float()

    holes = remove_small_fragments(mask, max_fragment_size=max_hole_size, num_repeats=num_repeats,
                                   return_fragments=True)

    if return_holes:
        return holes

    num_iterations = int(max_hole_size ** 0.5)

    image_temp = image_view.clone()

    for i in range(num_iterations):
        image_temp = F.max_pool2d(image_temp, kernel_size=3, stride=1, padding=1)

    labeled_holes = image_temp * holes

    return (labeled_holes + image_view).view_as(image)


def centroids_from_lab(lab: torch.Tensor):
    mesh_grid = torch.stack(torch.meshgrid(torch.arange(lab.shape[-2]), torch.arange(lab.shape[-1]))).float()

    sparse_onehot, label_ids = torch_sparse_onehot(lab, flatten=True)

    sum_centroids = torch.sparse.mm(sparse_onehot, mesh_grid.flatten(1).T)

    centroids = sum_centroids / torch.sparse.sum(sparse_onehot, dim=(1,)).to_dense().unsqueeze(-1)

    return centroids, label_ids  # N,2  N


def get_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64, return_lab_ids: bool = False):
    # lab is 1,H,W with N objects
    # image is C,H,W

    # Returns N,C,patch_size,patch_size

    centroids, label_ids = centroids_from_lab(lab)
    N = centroids.shape[0]

    C, h, w = image.shape[-3:]

    window_size = patch_size // 2
    centroids = centroids.clone()  # N,2
    centroids[:, 0].clamp_(min=window_size, max=h - window_size)
    centroids[:, 1].clamp_(min=window_size, max=w - window_size)
    window_slices = centroids[:, None] + torch.tensor([[-1, -1], [1, 1]]).to(image.device) * window_size
    window_slices = window_slices.long()  # N,2,2

    slice_size = window_size * 2

    # Create grids of indices for slice windows
    grid_x, grid_y = torch.meshgrid(
        torch.arange(slice_size, device=image.device),
        torch.arange(slice_size, device=image.device), indexing="ij")
    mesh = torch.stack((grid_x, grid_y))

    mesh_grid = mesh.expand(N, 2, slice_size, slice_size)  # N,2,2*window_size,2*window_size
    mesh_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)  # 2,N,2*window_size*2*window_size
    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_flat = mesh_flat + idx
    mesh_flater = torch.flatten(mesh_flat, 1)  # 2,N*2*window_size*2*window_size

    out = image[:, mesh_flater[0], mesh_flater[1]].reshape(C, N, -1)
    out = out.reshape(C, N, patch_size, patch_size)
    out = out.permute(1, 0, 2, 3)

    if return_lab_ids:
        return out, label_ids

    return out  # N,C,patch_size,patch_size


def get_masked_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64):
    # lab is 1,H,W
    # image is C,H,W

    lab_patches, label_ids = get_patches(lab, lab[0], patch_size, return_lab_ids=True)
    mask_patches = lab_patches == label_ids[1:, None, None, None]

    image_patches = get_patches(lab, image, patch_size)

    canvas = torch.ones_like(image_patches) * (~mask_patches).float()

    image_patches = image_patches * mask_patches.float() + canvas

    return image_patches  # N,C,patch_size,patch_size


def feature_extractor():
    import torch
    from torchvision.models.resnet import ResNet
    from torchvision.models.resnet import ResNet18_Weights
    import torch.nn as nn
    from typing import Type, Union, List, Optional, Any
    from torchvision.models.resnet import BasicBlock, Bottleneck, WeightsEnum

    class ResNetNoInitialDownsize(ResNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)

    def _resnet_custom(
            resnet_constructor,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            weights: Optional[WeightsEnum],
            progress: bool,
            **kwargs: Any,
    ) -> ResNet:
        # if weights is not None:
        #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

        model = resnet_constructor(block, layers, **kwargs)

        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=progress))

        return model

    weights = ResNet18_Weights.verify(ResNet18_Weights.IMAGENET1K_V1)
    # weights = None
    model = _resnet_custom(ResNetNoInitialDownsize, BasicBlock, [2, 2, 2, 2], weights, progress=True)

    return model
