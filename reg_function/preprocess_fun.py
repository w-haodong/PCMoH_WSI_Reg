import cv2
import math
import numpy as np
import scipy.ndimage as nd
import pandas as pd
import torch as tc
import torch.nn.functional as F
import torchvision.transforms as tr
from reg_function import tool_kit as tk

def pad_img(source, target, source_landmarks, target_landmarks):
    # 归一化
    source, target = tk.normalize(source), tk.normalize(target)
    # 使得两张图大小相同
    source, target, padding_params = tk.pad_to_same_size(source, target, 1)
    if source_landmarks is not None:
        source_landmarks = tk.pad_landmarks(source_landmarks, padding_params['pad_1'])

    if target_landmarks is not None:
        target_landmarks = tk.pad_landmarks(target_landmarks, padding_params['pad_2'])

    return source, target, source_landmarks, target_landmarks, padding_params

def gray_img(source, target, device):
    source = tk.image_to_tensor(source, device)
    target = tk.image_to_tensor(target, device)
    pre_source = 1 - tr.Grayscale()(source)
    pre_target = 1 - tr.Grayscale()(target)

    # 图像色彩均值化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    src = clahe.apply((pre_source[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    trg = clahe.apply((pre_target[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    source = src / 255
    target = trg / 255
    return source, target


# --- 灰度化和CLAHE函数 ---
def gray_and_clahe_preprocess(source_img_np, target_img_np, device_str):
    if not isinstance(source_img_np, np.ndarray) or not isinstance(target_img_np, np.ndarray):
        raise TypeError("gray_and_clahe_preprocess 的输入应为 NumPy 数组")
    source_tensor = tk.image_to_tensor(source_img_np, device_str)
    target_tensor = tk.image_to_tensor(target_img_np, device_str)
    grayscale_transform = tr.Grayscale()
    pre_source_tensor_gray_inv = 1 - grayscale_transform(source_tensor)
    pre_target_tensor_gray_inv = 1 - grayscale_transform(target_tensor)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    src_for_clahe = (pre_source_tensor_gray_inv[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    trg_for_clahe = (pre_target_tensor_gray_inv[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    source_clahe_uint8 = clahe.apply(src_for_clahe)
    target_clahe_uint8 = clahe.apply(trg_for_clahe)
    source_processed_float = source_clahe_uint8.astype(np.uint8)
    target_processed_float = target_clahe_uint8.astype(np.uint8)
    return source_processed_float, target_processed_float

def preprocess(source_cv_orig, target_cv_orig, source_landmarks_orig_np, target_landmarks_orig_np, device_str, resmaple_px):


    source_padded_cv, target_padded_cv, landmarks1_padded_np, landmarks2_padded_np, PADDING_INFO = pad_img(
        source_cv_orig, target_cv_orig,
        source_landmarks_orig_np.copy() if source_landmarks_orig_np is not None else None,
        target_landmarks_orig_np.copy() if target_landmarks_orig_np is not None else None
    )
    h_padded, w_padded = source_padded_cv.shape[:2]  # padded后源和目标同尺寸
    source = tk.image_to_tensor(source_padded_cv / 255, device_str)
    target = tk.image_to_tensor(target_padded_cv / 255, device_str)
    del source_cv_orig, target_cv_orig
    resampled_source, resampled_target, scale_common_applied = tk.initial_resampling(source, target, resmaple_px)

    img1_geom_processed = tk.tensor_to_image(resampled_source) * 255
    img2_geom_processed = tk.tensor_to_image(resampled_target) * 255
    del resampled_source, resampled_target

    print(
        f"几何预处理: 从 ({h_padded}, {w_padded}) 统一缩放至 ({img1_geom_processed.shape[0]}, {img1_geom_processed.shape[1]}), 因子: {scale_common_applied:.4f}")

    if landmarks1_padded_np is not None:
        landmarks1_geom_processed_np = landmarks1_padded_np / scale_common_applied
    else:
        landmarks1_geom_processed_np = None

    if landmarks2_padded_np is not None:
        landmarks2_geom_processed_np = landmarks2_padded_np / scale_common_applied
    else:
        landmarks2_geom_processed_np = None

    img1_for_process, img2_for_process = gray_and_clahe_preprocess(
        img1_geom_processed, img2_geom_processed, device_str)

    return img1_geom_processed, img2_geom_processed, img1_for_process, landmarks1_geom_processed_np, img2_for_process, landmarks2_geom_processed_np, PADDING_INFO, scale_common_applied
