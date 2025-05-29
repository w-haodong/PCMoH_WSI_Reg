import sys
current_file = sys.modules[__name__]

import math
import numpy as np
import torch as tc
import torch.nn.functional as F

def ncc_local(sources: tc.Tensor, targets: tc.Tensor, win_size = 3, device: str=None):
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("不支持的维度.")

    window = (win_size, ) * ndim
    if device is None:
        sum_filt = tc.ones([1, 1, *window]).type_as(sources)
    else:
        sum_filt = tc.ones([1, 1, *window], device=device)

    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -tc.mean(ncc)

def sparse_ncc(sources: tc.Tensor, targets: tc.Tensor, keypoints, win_size, device: str=None):

    scores = tc.zeros(len(keypoints), device = device)
    _, _, y_size, x_size = sources.shape
    for i in range(len(keypoints)):
        try:
            keypoint = int(keypoints[i].pt[0]), int(keypoints[i].pt[1])
        except:
            keypoint = int(keypoints[i, 0]), int(keypoints[i, 1])
        b_y, e_y = max(min(keypoint[1] - int(win_size // 2), y_size), 0), max(
            min(keypoint[1] + int(win_size // 2) + 1, y_size), 0)
        b_x, e_x = max(min(keypoint[0] - int(win_size // 2), x_size), 0), max(
            min(keypoint[0] + int(win_size // 2) + 1, x_size), 0)
        cs = sources[:, :, b_y:e_y, b_x:e_x]
        ts = targets[:, :, b_y:e_y, b_x:e_x]
        scores[i] = ncc_global(cs, ts)
    scores = scores[scores != 1]
    return tc.mean(scores)

def ncc_global(sources: tc.Tensor, targets: tc.Tensor, device: str="cpu"):
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("两个张量的形状必须相同")
    size = sources.size()
    prod_size = tc.prod(tc.Tensor(list(size[1:])))
    sources_mean = tc.mean(sources, dim=list(range(1, len(size)))).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_mean = tc.mean(targets, dim=list(range(1, len(size)))).view((targets.size(0),) + (len(size)-1)*(1,))
    sources_std = tc.std(sources, dim=list(range(1, len(size))), unbiased=False).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_std = tc.std(targets, dim=list(range(1, len(size))), unbiased=False).view((targets.size(0),) + (len(size)-1)*(1,))
    ncc = (1 / prod_size) * tc.sum((sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std), dim=list(range(1, len(size))))
    ncc = tc.mean(ncc)
    if ncc != ncc:
        ncc = tc.autograd.Variable(tc.Tensor([-1]), requires_grad=True).to(device)
    return -ncc