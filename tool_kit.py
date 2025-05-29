import os
import pathlib
import math
import numpy as np
import scipy.ndimage as nd
import pandas as pd
import matplotlib.pyplot as plt
import torch as tc
import torch.nn.functional as F
import torchvision.transforms as tr
import pyvips
import cv2
import csv
from scipy.interpolate import griddata
from skimage import color as skcolor, draw, exposure, filters, morphology
from typing import Union, Iterable, Sequence, Tuple
import colour
import SimpleITK as sitk
import PIL
import os
openslide_path = "C:/Program Files/openslide/bin"  # TODO
os.add_dll_directory(openslide_path)
import openslide
from datetime import datetime

def get_level_dimensions(image_path):
    slide = openslide.OpenSlide(image_path)
    level_dimensions = slide.level_dimensions
    return level_dimensions

def load_image(image_path, level, load_slide=False):
    slide = openslide.OpenSlide(image_path)
    dimension = slide.level_dimensions[level]
    image = slide.read_region((0, 0), level, dimension)
    image = np.asarray(image)[:, :, 0:3].astype(np.float32)
    image = normalize(image)
    if load_slide:
        return image, slide
    else:
        return image
def normalize(image):
    # 根据图像形式（变量/张量）进行归一化
    if isinstance(image, np.ndarray):
        return normalize_np(image)
    if isinstance(image, tc.Tensor):
        return normalize_tc(image)
    else:
        raise ValueError("Unsupported type.")

# np归一化
def normalize_np(image : np.ndarray):
    if len(image.shape) == 2:
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    elif len(image.shape) == 3:
        normalized_image = np.zeros_like(image)
        for i in range(normalized_image.shape[2]):
            normalized_image[:, :, i] = normalize(image[:, :, i])
        return normalized_image
    else:
        raise ValueError("Unsupported number of channels.")

# tc归一化
def normalize_tc(tensor : tc.Tensor):
    if len(tensor.size()) - 2 == 2:
        num_channels = tensor.size(1)
        normalized_tensor = tc.zeros_like(tensor)
        for i in range(num_channels):
            mins, _ = tc.min(tc.min(tensor[:, i, :, :] , dim=1, keepdim=True)[0], dim=2, keepdim=True) # TODO - find better approach
            maxs, _ = tc.max(tc.max(tensor[:, i, :, :] , dim=1, keepdim=True)[0], dim=2, keepdim=True)
            normalized_tensor[:, i, :, :] = (tensor[:, i, :, :] - mins) / (maxs - mins)
        return normalized_tensor
    else:
        raise ValueError("Unsupported number of channels.")
def initial_resampling(

    source : Union[tc.Tensor, np.ndarray],
    target : Union[tc.Tensor, np.ndarray],
    resolution : int) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray]]:
    """
    TODO
    """
    source_y_size, source_x_size, target_y_size, target_x_size = get_combined_size(source, target)
    resample_ratio = calculate_resampling_ratio((source_x_size, target_x_size), (source_y_size, target_y_size), resolution)
    resampled_source = resample(gaussian_smoothing(source, min(max(resample_ratio -1, 0.1), 10)), resample_ratio)
    resampled_target = resample(gaussian_smoothing(target, min(max(resample_ratio -1, 0.1), 10)), resample_ratio)
    return resampled_source, resampled_target, resample_ratio

def resample(tensor : tc.Tensor, resample_ratio : float, mode: str="bilinear") -> tc.Tensor:
    """
    TODO
    """
    return F.interpolate(tensor, scale_factor = 1 / resample_ratio, mode=mode, recompute_scale_factor=False, align_corners=False)

def get_combined_size(tensor_1 : tc.Tensor, tensor_2 : tc.Tensor) -> Iterable[int]:
    """
    TODO
    """
    tensor_1_y_size, tensor_1_x_size = tensor_1.size(2), tensor_1.size(3)
    tensor_2_y_size, tensor_2_x_size = tensor_2.size(2), tensor_2.size(3)
    return tensor_1_y_size, tensor_1_x_size, tensor_2_y_size, tensor_2_x_size

def calculate_resampling_ratio(x_sizes : Iterable, y_sizes : Iterable, min_resolution : int) -> float:
    """
    TODO
    """
    x_size, y_size = max(x_sizes), max(y_sizes)
    min_size = min(x_size, y_size)
    if min_resolution > min_size:
        resampling_ratio = 1
    else:
        resampling_ratio = min_size / min_resolution
    return resampling_ratio

def load_img_with_level(data_dir, level, device):
    source_dir = data_dir + '/source_level_' + str(level) + '.jpg'
    target_dir = data_dir + '/target_level_' + str(level) + '.jpg'
    source = cv2.imread(source_dir, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_dir, cv2.IMREAD_GRAYSCALE)
    source = tc.from_numpy(source.astype(np.float32) / 255).to(device).unsqueeze(0).unsqueeze(0)
    target = tc.from_numpy(target.astype(np.float32) / 255).to(device).unsqueeze(0).unsqueeze(0)
    return source, target

def calculate_tre(source_landmarks, target_landmarks):
    tre = np.sqrt(np.square(source_landmarks[:, 0] - target_landmarks[:, 0]) + np.square(source_landmarks[:, 1] - target_landmarks[:, 1]))
    return tre

def calculate_rtre(source_landmarks, target_landmarks, image_diagonal):
    tre = calculate_tre(source_landmarks, target_landmarks)
    rtre = tre / image_diagonal
    return rtre

def tre(source_landmarks : Union[tc.Tensor, np.ndarray], target_landmarks : Union[tc.Tensor, np.ndarray]):
    # TODO - documentation
    if isinstance(source_landmarks, tc.Tensor) and isinstance(target_landmarks, tc.Tensor):
        return tc.sqrt(((source_landmarks - target_landmarks)**2).sum(axis=1))
    elif isinstance(source_landmarks, np.ndarray) and isinstance(target_landmarks, np.ndarray):
        return np.sqrt(((source_landmarks - target_landmarks)**2).sum(axis=1))
    else:
        raise ValueError("Unsupported type.")

def points_to_homogeneous_representation(points: np.ndarray):
    homogenous_points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    return homogenous_points

def generate_grid_tc(tensor : tc.Tensor=None, tensor_size: tc.Tensor=None, device: str=None):
    if tensor is not None:
        tensor_size = tensor.size()
    if device is None:
        identity_transform = tc.eye(len(tensor_size)-1)[:-1, :].unsqueeze(0).type_as(tensor)
    else:
        identity_transform = tc.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    # 按指定的方式重复张量中的元素(dim=0/1/2)
    identity_transform = tc.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    # 给定一批仿射矩阵:attr:'theta'，生成二维或三维流场(采样网格)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid


# 图像转换（变量/张量）
def image_to_tensor(image: np.ndarray, device: str = "cpu"):
    """
    将图像转换为张量，并将其移动到指定的设备（如CPU或GPU）。

    参数:
    image (np.ndarray): 输入图像，可以是二维（灰度图）或三维（彩色图）。
    device (str): 指定设备，默认是"cpu"。

    返回:
    torch.Tensor: 转换后的张量，形状为 (1, 3, H, W)。
    """
    if len(image.shape) == 3:  # 如果是彩色图像
        # 转换为PyTorch张量，并调整维度顺序，然后添加一个批次维度并移动到指定设备
        return tc.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    elif len(image.shape) == 2:  # 如果是灰度图像
        # 转换为PyTorch张量，添加通道维度并复制三次，然后添加一个批次维度并移动到指定设备
        return tc.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    else:
        raise ValueError("输入图像必须是二维或三维数组")

def tensor_to_image(tensor : tc.Tensor):
    if tensor.size(0) == 1:
        return tensor[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
    else:
        return tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

def load_landmarks_ANHIR(landmarks_path):
    landmarks = pd.read_csv(landmarks_path)
    landmarks = landmarks.to_numpy()[:, 1:3]
    return landmarks

def load_landmarks_ACRBOAT(dataset_csv, dataset_id):
    df = pd.read_csv(dataset_csv)
    filtered_df = df[df['anon_id'] == dataset_id]
    landmarks = filtered_df[['ihc_x', 'ihc_y', 'mpp_ihc_10X']].to_numpy()
    return landmarks

def resample_landmarks_ACRBOAT(source_landmarks, level_ini_dim, level_current_dim):
    new_source_landmarks = np.zeros((source_landmarks.shape[0], 5))
    new_source_landmarks[:, 0] = source_landmarks[:, 0] / source_landmarks[:, 2]
    new_source_landmarks[:, 1] = source_landmarks[:, 1] / source_landmarks[:, 2]
    x_p = level_current_dim[0] / level_ini_dim[0]
    y_p = level_current_dim[1] / level_ini_dim[1]
    new_source_landmarks[:, 0] = new_source_landmarks[:, 0] * (x_p)
    new_source_landmarks[:, 1] = new_source_landmarks[:, 1] * (y_p)
    new_source_landmarks[:, 2] = source_landmarks[:, 2]
    new_source_landmarks[:, 3] = x_p
    new_source_landmarks[:, 4] = y_p
    return new_source_landmarks

def save_landmarks(landmarks, landmarks_path):
    if landmarks.shape[1] > 2:
        df = pd.DataFrame(landmarks)
    else:
        df = pd.DataFrame([{'X': x, 'Y': y} for x, y in landmarks])
    df.to_csv(landmarks_path)

# np重采样
def resample_img(image, resample_ratio, cval=0, order=1):
    if len(image.shape) == 2:
        y_size, x_size = image.shape
        new_y_size, new_x_size = int(y_size / resample_ratio), int(x_size / resample_ratio)
        grid_x, grid_y = np.meshgrid(np.arange(new_x_size), np.arange(new_y_size))
        grid_x = grid_x * (x_size / new_x_size)
        grid_y = grid_y * (y_size / new_y_size)
        resampled_image = nd.map_coordinates(image, [grid_y, grid_x], cval=cval, order=3)
    elif len(image.shape) == 3:
        y_size, x_size, num_channels = image.shape
        new_y_size, new_x_size = int(y_size / resample_ratio), int(x_size / resample_ratio)
        grid_x, grid_y = np.meshgrid(np.arange(new_x_size), np.arange(new_y_size))
        grid_x = grid_x * (x_size / new_x_size)
        grid_y = grid_y * (y_size / new_y_size)
        resampled_image = np.zeros((grid_x.shape[0], grid_x.shape[1], num_channels))
        for i in range(num_channels):
            resampled_image[:, :, i] = nd.map_coordinates(image[:, :, i], [grid_y, grid_x], cval=cval, order=order)
    else:
        raise ValueError("不支持的通道数")
    return resampled_image

def pad_to_same_size(image_1 : np.ndarray, image_2 : np.ndarray, pad_value : float=1.0):
    y_size_1, x_size_1 = image_1.shape[0], image_1.shape[1]
    y_size_2, x_size_2 = image_2.shape[0], image_2.shape[1]
    pad_1, pad_2 = calculate_pad_value((y_size_1, x_size_1), (y_size_2, x_size_2))
    image_1 = np.pad(image_1, ((pad_1[0][0], pad_1[0][1]), (pad_1[1][0], pad_1[1][1]), (0, 0)), mode='constant', constant_values=pad_value)
    image_2 = np.pad(image_2, ((pad_2[0][0], pad_2[0][1]), (pad_2[1][0], pad_2[1][1]), (0, 0)), mode='constant', constant_values=pad_value)
    padding_params = dict()
    padding_params['pad_1'] = pad_1
    padding_params['pad_2'] = pad_2
    return image_1, image_2, padding_params

def calculate_pad_value(size_1 : Iterable[int], size_2 : Iterable[int]) -> Tuple[Iterable[tuple], Iterable[tuple]]:
    y_size_1, x_size_1 = size_1
    y_size_2, x_size_2 = size_2
    pad_1 = [(0, 0), (0, 0)]
    pad_2 = [(0, 0), (0, 0)]
    if y_size_1 > y_size_2:
        pad_size = y_size_1 - y_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[0] = pad
    elif y_size_1 < y_size_2:
        pad_size = y_size_2 - y_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[0] = pad
    else:
        pass
    if x_size_1 > x_size_2:
        pad_size = x_size_1 - x_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[1] = pad
    elif x_size_1 < x_size_2:
        pad_size = x_size_2 - x_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[1] = pad
    else:
        pass
    return pad_1, pad_2

def pad_landmarks(landmarks, padding_size):
    y_pad = padding_size[0]
    x_pad = padding_size[1]
    landmarks[:, 0] = landmarks[:, 0] + x_pad[0]
    landmarks[:, 1] = landmarks[:, 1] + y_pad[0]
    return landmarks

def pad_to_ori_landmarks(landmarks, padding_size):
    y_pad = float(padding_size[0][0])
    x_pad = float(padding_size[1][0])
    landmarks[:, 0] = landmarks[:, 0] - x_pad
    landmarks[:, 1] = landmarks[:, 1] - y_pad
    return landmarks


# 创建金字塔
def create_pyramid(tensor: tc.Tensor, num_levels: int, mode: str = 'bilinear'):
    """
    创建输入张量的分辨率金字塔(假设均匀重采样步长= 2)
    """
    pyramid = [None] * num_levels
    pyramid_sizes = []  # 用来存储每个层级的尺寸

    for i in range(num_levels - 1, -1, -1):
        if i == num_levels - 1:
            pyramid[i] = tensor
            pyramid_sizes.append([tensor.size(2),tensor.size(3)])  # 初始尺寸
        else:
            current_size = pyramid[i + 1].size()
            new_size = (int(current_size[j] / 2) if j > 1 else current_size[j] for j in range(len(current_size)))
            new_size = tc.Size(new_size)[2:]  # 假设我们只关心空间维度
            new_tensor = resample_tensor_to_size(gaussian_smoothing(pyramid[i + 1], 1), new_size, mode=mode)
            pyramid[i] = new_tensor
            pyramid_sizes.append([new_size[0], new_size[1]])  # 存储新的尺寸
    return pyramid, pyramid_sizes  # 返回金字塔和 NumPy 数组形式的尺寸

def resample_tensor_to_size(tensor: tc.Tensor, new_size: tc.Tensor, mode: str='bilinear'):
    return F.interpolate(tensor, size=new_size, mode=mode, align_corners=False)

# 高斯平滑
def gaussian_smoothing(image, sigma):
    if isinstance(image, np.ndarray):
        return gaussian_smoothing_np(image, sigma)
    elif isinstance(image, tc.Tensor):
        diagonal = calculate_diagonal(image)
        if diagonal > 10000:
            return gaussian_smoothing_patch_tc(image, sigma)
        else:
            return gaussian_smoothing_tc(image, sigma)
    else:
        raise TypeError("不支持的类型")

# 评估对角
def calculate_diagonal(tensor : tc.Tensor):
    return math.sqrt(tensor.size(2)**2 + tensor.size(3)**2)

# np平滑
def gaussian_smoothing_np(image, sigma):
    if len(image.shape) == 2:
        return nd.gaussian_filter(image, sigma)
    elif len(image.shape) == 3:
        _, _, num_channels = image.shape
        smoothed_image = np.zeros_like(image)
        for i in range(num_channels):
            smoothed_image[:, :, i] = nd.gaussian_filter(image[:, :, i], sigma)
        return smoothed_image
    else:
        raise ValueError("不支持的通道数")

# tc平滑
def gaussian_smoothing_tc(tensor, sigma):
    with tc.set_grad_enabled(False):
        kernel_size = int(sigma * 2.54) + 1 if int(sigma * 2.54) % 2 == 0 else int(sigma * 2.54)
        return tr.GaussianBlur(kernel_size, sigma)(tensor)

def gaussian_smoothing_patch_tc(tensor, sigma, patch_size=(2048, 2048), offset=(50, 50)):
    smoothed_tensor = tc.zeros_like(tensor)
    with tc.set_grad_enabled(False):
        y_size, x_size = tensor.size(2), tensor.size(3)
        rows, cols = int(np.ceil(y_size / patch_size[0])), int(np.ceil(x_size / patch_size[1]))
        for row in range(rows):
            for col in range(cols):
                b_x = max(0, min(x_size, col*patch_size[1]))
                b_y = max(0, min(y_size, row*patch_size[0]))
                e_x = max(0, min(x_size, (col+1)*patch_size[1]))
                e_y = max(0, min(y_size, (row+1)*patch_size[0]))
                ob_x = max(0, min(x_size, b_x - offset[1]))
                oe_x = max(0, min(x_size, e_x + offset[1]))
                ob_y = max(0, min(y_size, b_y - offset[0]))
                oe_y = max(0, min(y_size, e_y + offset[0]))
                diff_bx = b_x - ob_x
                diff_by = b_y - ob_y
                smoothed_tensor[:, :, b_y:e_y, b_x:e_x] = gaussian_smoothing(tensor[:, :, ob_y:oe_y, ob_x:oe_x], sigma)[:, :, diff_by:diff_by+patch_size[0], diff_bx:diff_bx+patch_size[1]]
    return smoothed_tensor


def create_forward_identity_displacement_field(tensor_bchw: tc.Tensor) -> tc.Tensor:
    """创建正向零位移场 (B, 2, H, W)。"""
    batch_size, _, height, width = tensor_bchw.shape
    identity_df = tc.zeros((batch_size, 2, height, width),
                           dtype=tensor_bchw.dtype,
                           device=tensor_bchw.device)
    return identity_df


def warp_forward(source_tensor_bchw: tc.Tensor,
                 forward_displacement_field_b2hw: tc.Tensor,  # 假设这是 P_new - P_old
                 mode: str = 'bilinear',
                 padding_mode: str = 'border',
                 align_corners: bool = True,
                 output_size: tuple = None  # 新增
                 ) -> tuple[tc.Tensor, tc.Tensor]:
    """使用“前向”定义的位移场进行反向查找变形图像。"""
    B, C, H_in, W_in = source_tensor_bchw.shape
    dev = source_tensor_bchw.device

    if output_size is None:
        H_out, W_out = H_in, W_in
    else:
        H_out, W_out = output_size
        if not (isinstance(H_out, int) and isinstance(W_out, int) and H_out > 0 and W_out > 0):
            raise ValueError(f"output_size 必须是正整数的元组 (H_out, W_out)，得到: {output_size}")

    # 1. 创建输出画布的坐标网格 (x_t_coords, y_t_coords)
    y_t_coords, x_t_coords = tc.meshgrid(
        tc.arange(H_out, device=dev, dtype=tc.float32),
        tc.arange(W_out, device=dev, dtype=tc.float32),
        indexing='ij'
    )  # Shape H_out, W_out

    # 2. 如果 forward_displacement_field_b2hw 的尺寸与输出尺寸不同，需要重采样
    #    这个位移场 DF(P_source) = P_target - P_source 是定义在源图像网格上的。
    #    我们需要的是对于输出画布上的每个点 P_target_canvas，它应该从源图像的哪个点 P_source_sample 采样。
    #    P_source_sample = P_target_canvas - DF_interpolated_at_P_target_canvas_but_representing_forward_from_P_source_that_maps_there
    #    这还是很复杂。

    # 让我们回到你提供的 warp_forward 实现，并只修改它以接受 output_size
    # 它的核心是 x_s_sample_pixels = x_t_coords - dx_fwd_bhw
    # 这意味着 dx_fwd_bhw 必须是 (x_t_coords - x_s_sample_pixels)
    # 即，位移场 DF[xt,yt] = (xt-xs, yt-ys)

    # 如果 matrix_to_dense_displacement_field 返回的是 (xs_transformed - xs_orig, ys_transformed - ys_orig)
    # 而 warp_forward 期望的是 (xt_canvas - xs_sample, yt_canvas - ys_sample)

    # *** 最直接的修改，假设你的 warp_forward 核心逻辑是对的，只是输出尺寸问题 ***
    # 它会输出一个与 forward_displacement_field_b2hw 相同 H,W 的图像。
    # 所以，如果想让输出是 H_out, W_out，那么 forward_displacement_field_b2hw
    # 也必须被重采样到 H_out, W_out。

    # 如果 H_out, W_out 与 forward_displacement_field_b2hw 的 H,W 不同，
    # 我们需要调整位移场以匹配输出尺寸。
    # 这假设位移场可以被平滑地缩放。

    df_H, df_W = forward_displacement_field_b2hw.shape[2], forward_displacement_field_b2hw.shape[3]

    resampled_df_b2hw = forward_displacement_field_b2hw
    if H_out != df_H or W_out != df_W:
        print(f"Warp_forward: Resampling displacement field from ({df_H},{df_W}) to ({H_out},{W_out})")
        resampled_df_b2hw = F.interpolate(forward_displacement_field_b2hw,
                                          size=(H_out, W_out),
                                          mode='bilinear',
                                          align_corners=align_corners)
        # 注意：直接缩放位移值可能不完全正确，因为位移是像素单位的。
        # 如果从大图缩到小图，位移值应该相应缩小。反之亦然。
        scale_h = H_out / df_H if df_H > 0 else 1.0
        scale_w = W_out / df_W if df_W > 0 else 1.0
        resampled_df_b2hw[:, 0, :, :] *= scale_w  # dx scales with width
        resampled_df_b2hw[:, 1, :, :] *= scale_h  # dy scales with height

    # 现在 resampled_df_b2hw 的 H,W 与 H_out, W_out 相同
    y_t_coords_out, x_t_coords_out = tc.meshgrid(
        tc.arange(H_out, device=dev, dtype=tc.float32),
        tc.arange(W_out, device=dev, dtype=tc.float32),
        indexing='ij'
    )
    dx_fwd_bhw_resampled = resampled_df_b2hw[:, 0, :, :]
    dy_fwd_bhw_resampled = resampled_df_b2hw[:, 1, :, :]

    # 你的原始采样逻辑，但现在 x_t_coords 和位移场尺寸一致
    x_s_sample_pixels = x_t_coords_out.unsqueeze(0).expand(B, -1, -1) - dx_fwd_bhw_resampled
    y_s_sample_pixels = y_t_coords_out.unsqueeze(0).expand(B, -1, -1) - dy_fwd_bhw_resampled

    # 归一化是相对于输入源图像 source_tensor_bchw 的尺寸 (H_in, W_in)
    W_in_eff = W_in - 1 if W_in > 1 else 1.0
    H_in_eff = H_in - 1 if H_in > 1 else 1.0

    if W_in == 1:
        x_s_sample_norm = tc.zeros_like(x_s_sample_pixels)
    else:
        x_s_sample_norm = (x_s_sample_pixels / W_in_eff) * 2 - 1 if align_corners else ((
                                                                                                    x_s_sample_pixels * 2 + 1) / W_in) - 1

    if H_in == 1:
        y_s_sample_norm = tc.zeros_like(y_s_sample_pixels)
    else:
        y_s_sample_norm = (y_s_sample_pixels / H_in_eff) * 2 - 1 if align_corners else ((
                                                                                                    y_s_sample_pixels * 2 + 1) / H_in) - 1

    sampling_grid_bhw2 = tc.stack((x_s_sample_norm, y_s_sample_norm), dim=-1)  # Shape B, H_out, W_out, 2

    warped_image = F.grid_sample(source_tensor_bchw,  # 输入是原始源图像
                                 sampling_grid_bhw2,  # 采样格点，其值是相对于源图像的归一化坐标
                                 mode=mode,
                                 padding_mode=padding_mode,
                                 align_corners=align_corners)
    return warped_image, sampling_grid_bhw2


def resample_displacement_field_b2hw_to_size(  # 重命名以区分
        displacement_field_b2hw: tc.Tensor,
        new_hw_tuple: tuple[int, int],
        mode: str = 'bilinear',
        align_corners: bool = True  # 与warp_forward和grid_sample保持一致
) -> tc.Tensor:
    """重采样 (B, 2, H, W) 位移场到新尺寸，并缩放位移值。"""
    B, _, H_in, W_in = displacement_field_b2hw.shape
    H_out, W_out = new_hw_tuple

    if H_in == 0 or W_in == 0 or H_out == 0 or W_out == 0:
        return tc.zeros((B, 2, H_out, W_out),
                        dtype=displacement_field_b2hw.dtype,
                        device=displacement_field_b2hw.device)

    resampled_df_spatial = F.interpolate(displacement_field_b2hw,
                                         size=new_hw_tuple,
                                         mode=mode,
                                         align_corners=align_corners)

    scale_w = W_out / W_in if W_in > 0 else 1.0
    scale_h = H_out / H_in if H_in > 0 else 1.0

    resampled_df_scaled = tc.zeros_like(resampled_df_spatial)
    resampled_df_scaled[:, 0, :, :] = resampled_df_spatial[:, 0, :, :] * scale_w
    resampled_df_scaled[:, 1, :, :] = resampled_df_spatial[:, 1, :, :] * scale_h

    return resampled_df_scaled


def compose_forward_displacement_fields(df1_fwd_b2hw: tc.Tensor,
                                        df2_fwd_b2hw: tc.Tensor,
                                        mode: str = 'bilinear',
                                        padding_mode: str = 'border',
                                        align_corners: bool = True
                                        ) -> tc.Tensor:
    """组合两个正向位移场。"""
    B, _, H, W = df1_fwd_b2hw.shape
    device = df1_fwd_b2hw.device
    y_orig_coords, x_orig_coords = tc.meshgrid(
        tc.arange(H, device=device, dtype=tc.float32),
        tc.arange(W, device=device, dtype=tc.float32),
        indexing='ij'
    )
    x_orig_coords_bhw = x_orig_coords.unsqueeze(0).expand(B, -1, -1)
    y_orig_coords_bhw = y_orig_coords.unsqueeze(0).expand(B, -1, -1)
    x_intermediate_pixels = x_orig_coords_bhw + df1_fwd_b2hw[:, 0, :, :]
    y_intermediate_pixels = y_orig_coords_bhw + df1_fwd_b2hw[:, 1, :, :]

    if W == 1:
        x_intermediate_norm = tc.zeros_like(x_intermediate_pixels)
    else:
        x_intermediate_norm = (x_intermediate_pixels / (W - 1)) * 2 - 1 if align_corners else ((
                                                                                                           x_intermediate_pixels * 2 + 1) / W) - 1
    if H == 1:
        y_intermediate_norm = tc.zeros_like(y_intermediate_pixels)
    else:
        y_intermediate_norm = (y_intermediate_pixels / (H - 1)) * 2 - 1 if align_corners else ((
                                                                                                           y_intermediate_pixels * 2 + 1) / H) - 1

    sampling_grid_for_df2 = tc.stack((x_intermediate_norm, y_intermediate_norm), dim=-1)
    df2_resampled_b2hw = F.grid_sample(df2_fwd_b2hw, sampling_grid_for_df2,
                                       mode=mode, padding_mode=padding_mode,
                                       align_corners=align_corners)
    total_displacement_fwd_b2hw = df1_fwd_b2hw + df2_resampled_b2hw
    return total_displacement_fwd_b2hw

def matrix_to_dense_displacement_field(affine_matrix_np: np.ndarray,
                                            target_height: int,
                                            target_width: int,
                                            device: str = 'cpu') -> tc.Tensor:
    """
    将一个3x3的仿射变换矩阵转换为一个稠密的B2HW位移场。
    位移场表示每个像素从原始位置到变换后位置的位移向量 (dx, dy)。
    输出的位移场可以直接用于 tc.nn.functional.grid_sample (如果配合单位网格)。
    或者，如果 tk.warp_forward 期望的是像素单位的绝对位移，那么这个函数应该返回这个。

    这里我们生成的是像素单位的绝对位移 (x_transformed - x_original, y_transformed - y_original)。

    Args:
        affine_matrix_np (np.ndarray): 3x3的仿射变换矩阵 (NumPy数组)。
        target_height (int): 目标位移场的高度。
        target_width (int): 目标位移场的宽度。
        device (str): 生成张量的设备 ('cpu' or 'cuda:x').

    Returns:
        tc.Tensor: 形状为 (1, 2, target_height, target_width) 的位移场张量。
                   通道0是x方向的位移 (dx)，通道1是y方向的位移 (dy)。
                   单位是像素。
    """
    if not isinstance(affine_matrix_np, np.ndarray) or affine_matrix_np.shape != (3, 3):
        raise ValueError("affine_matrix_np 必须是一个3x3的NumPy数组。")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_height 和 target_width 必须是正整数。")

    # 将NumPy仿射矩阵转换为PyTorch张量
    affine_matrix_tc = tc.from_numpy(affine_matrix_np).float().to(device)

    # 创建原始像素坐标网格 (x_coords, y_coords)
    # y_coords 范围 [0, target_height - 1]
    # x_coords 范围 [0, target_width - 1]
    y_coords_orig, x_coords_orig = tc.meshgrid(tc.arange(target_height, device=device, dtype=tc.float32),
                                               tc.arange(target_width, device=device, dtype=tc.float32),
                                               indexing='ij') # H, W

    # 将原始坐标转换为齐次坐标 (x, y, 1)
    # 堆叠成 (3, H*W) 的形式
    ones_hw = tc.ones_like(x_coords_orig) # H, W
    coords_homogeneous_hw3 = tc.stack((x_coords_orig, y_coords_orig, ones_hw), dim=0) # 3, H, W
    coords_homogeneous_3_n = coords_homogeneous_hw3.reshape(3, -1) # 3, H*W (N=H*W)

    # 应用仿射变换: transformed_coords_h = Affine @ original_coords_h
    # (3x3) @ (3xN) -> (3xN)
    transformed_coords_homogeneous_3_n = affine_matrix_tc @ coords_homogeneous_3_n

    # 转换回笛卡尔坐标 (除以w分量)
    # w分量 (transformed_coords_homogeneous_3_n[2, :]) 应该接近1，除非有非常奇异的变换
    w_component = transformed_coords_homogeneous_3_n[2, :]
    # 防止除以零或非常小的值
    w_component[tc.abs(w_component) < 1e-7] = 1e-7 if tc.all(w_component >= 0) else -1e-7 # 保持符号

    transformed_x_coords_n = transformed_coords_homogeneous_3_n[0, :] / w_component
    transformed_y_coords_n = transformed_coords_homogeneous_3_n[1, :] / w_component

    # 计算位移: dx = transformed_x - original_x, dy = transformed_y - original_y
    dx_n = transformed_x_coords_n - x_coords_orig.reshape(-1) # H*W
    dy_n = transformed_y_coords_n - y_coords_orig.reshape(-1) # H*W

    # 将位移场重塑为 (2, H, W)
    displacement_field_2hw = tc.stack((dx_n.reshape(target_height, target_width),
                                       dy_n.reshape(target_height, target_width)), dim=0) # 2, H, W

    # 添加批次维度 (B=1) -> B2HW
    displacement_field_b2hw = displacement_field_2hw.unsqueeze(0) # 1, 2, H, W

    return displacement_field_b2hw


def create_identity_displacement_field(tensor : tc.Tensor):
    return tc.zeros((tensor.size(0), tensor.size(2), tensor.size(3)) + (2,)).type_as(tensor)

def resample_displacement_field_to_size(displacement_field: tc.Tensor, new_size: tc.Tensor, mode: str='bilinear'):
    return F.interpolate(displacement_field.permute(0, 3, 1, 2), size=new_size, mode=mode, align_corners=False).permute(0, 2, 3, 1)

def compose_displacement_fields(displacement_field_1, displacement_field_2):
    sampling_grid = generate_grid(tensor_size=(displacement_field_1.size(0), 1, displacement_field_1.size(1), displacement_field_1.size(2)), device=displacement_field_1.device)
    composed_displacement_field = F.grid_sample((sampling_grid + displacement_field_1).permute(0, 3, 1, 2), sampling_grid + displacement_field_2, padding_mode='border', align_corners=False).permute(0, 2, 3, 1)
    composed_displacement_field = composed_displacement_field - sampling_grid
    return composed_displacement_field

def warp_tensor(tensor: tc.Tensor, displacement_field: tc.Tensor, grid: tc.Tensor = None, mode: str = 'bilinear',
                   padding_mode: str = 'zeros', device: str = None):
    if grid is None:
        grid = generate_grid(tensor=tensor, device=device)
    sampling_grid = grid + displacement_field

    # 进行变形采样
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return transformed_tensor, sampling_grid

def generate_grid(tensor : tc.Tensor=None, tensor_size: tc.Tensor=None, device: str=None):
    if tensor is not None:
        tensor_size = tensor.size()
    if device is None:
        identity_transform = tc.eye(len(tensor_size)-1)[:-1, :].unsqueeze(0).type_as(tensor)
    else:
        identity_transform = tc.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    # 按指定的方式重复张量中的元素(dim=0/1/2)
    identity_transform = tc.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    # 给定一批仿射矩阵:attr:'theta'，生成二维或三维流场(采样网格)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def warp_landmarks(df_grid, landmarks, affined_landmarks):
    temp_landmarks = landmarks.copy()
    df_grid_np = df_grid.detach().clone().cpu().numpy()[0]
    H, W, _ = df_grid_np.shape

    # 处理变形场的方向和缩放
    df_grid_np[:, :, 0] = (df_grid_np[:, :, 0] + 1) * (W / 2)
    df_grid_np[:, :, 1] = (df_grid_np[:, :, 1] + 1) * (H / 2)

    count_error = 0
    matched_coordinates = []
    fw_value = 0.5
    max_fw_add = 5  # 最大的fw_add值

    # 对每个 landmarks 坐标，查找 df_grid_np 中匹配的 X 和 Y 坐标
    for lm_i in range(len(landmarks)):
        x_val, y_val = landmarks[lm_i, :2]
        fw_add = 0.5
        found_match = False

        while fw_add <= max_fw_add:
            fw_max = fw_value + fw_add
            matches = np.where(((df_grid_np[..., 0] <= (x_val + fw_max)) &
                                (df_grid_np[..., 0] >= (x_val - fw_max)) &
                                (df_grid_np[..., 1] <= (y_val + fw_max)) &
                                (df_grid_np[..., 1] >= (y_val - fw_max))))

            if len(matches[0]) > 0:
                mat_lm_x = np.mean(matches[1])
                mat_lm_y = np.mean(matches[0])
                found_match = True
                break

            fw_add += 1

        if not found_match:
            mat_lm_x = affined_landmarks[lm_i, 0]
            mat_lm_y = affined_landmarks[lm_i, 1]
            count_error += 1

        matched_coordinates.append([mat_lm_x, mat_lm_y])

    matched_coordinates = np.array(matched_coordinates)
    temp_landmarks[:, :2] = matched_coordinates
    return temp_landmarks

def current_timestamp():
    """
    获取当前日期和时间，并格式化为 "YYYYMMDD_HHMMSS" 格式的字符串。
    例如: "20231027_153045"
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    return timestamp_str