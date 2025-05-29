import sys
import os
import cv2
import numpy as np
import traceback  # For detailed error logging
from datetime import datetime  # Added for current_timestamp if tk.current_timestamp is not available

# Set KMP_DUPLICATE_LIB_OK before PyTorch/OpenCV might be imported deeply
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch as tc
from scipy.spatial.distance import cdist

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QLineEdit, QTextEdit,
                             QGridLayout, QMessageBox, QSizePolicy, QDialog,  # Added QDialog
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)  # Added QGraphics...
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QPainter, QTransform  # Added QPainter, QTransform
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QPointF, QRectF  # Added QPointF, QRectF

# --- Ensure reg_function is importable ---
# Add current directory to sys.path to find reg_function if it's alongside the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from reg_function.kps_des_calc import kps_des_calculation, kps_match
    from reg_function import tool_kit as tk
    from reg_function import preprocess_fun as pf
    from reg_function import cost_functions as cf
    from reg_function import regularizers as rl
except ImportError as e:
    print(f"ERROR: Could not import 'reg_function' modules. Make sure 'reg_function' directory is accessible.")
    print(f"Details: {e}")
    app_temp = QApplication.instance()
    if not app_temp: app_temp = QApplication(sys.argv)
    QMessageBox.critical(None, "Import Error",
                         "Could not import 'reg_function' modules.\n"
                         "Ensure 'reg_function' directory is in the same folder as the script or in PYTHONPATH.\n"
                         f"Error details: {e}")
    sys.exit(1)


# --- Helper: current_timestamp (if not in tk or tk unavailable) ---
def get_current_timestamp_str():
    """
    Gets current timestamp as YYYYMMDD_HHMMSS string.
    Prioritizes tk.current_timestamp if available.
    """
    try:
        # Try to use the one from user's toolkit first
        return tk.current_timestamp()
    except (AttributeError, NameError):
        # Fallback if tk.current_timestamp doesn't exist or tk isn't fully imported yet
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")


# --- RBF Functions (from user script) ---
def rbf_kernel_function(distances, epsilon, kernel_type='gaussian'):
    if kernel_type == 'gaussian':
        return np.exp(-(epsilon * distances) ** 2)
    elif kernel_type == 'multiquadric':
        return np.sqrt(1 + (epsilon * distances) ** 2)
    elif kernel_type == 'inv_multiquadric':
        return 1.0 / np.sqrt(1 + (epsilon * distances) ** 2)
    elif kernel_type == 'thin_plate':
        d_sq = distances ** 2
        zero_mask = np.isclose(distances, 0.0, atol=1e-9)
        log_r = np.log(np.where(zero_mask, 1.0, distances))
        result = d_sq * log_r
        result[zero_mask] = 0
        return result
    else:
        raise ValueError(f"Unknown RBF kernel type: {kernel_type}")


def compute_adaptive_epsilon_and_weights(rbf_centers, pcmh_quality_scores=None,
                                         k_neighbors=5, epsilon_scale_factor=0.5):
    num_control_points = rbf_centers.shape[0]
    epsilons = np.ones(num_control_points)
    significance_weights = np.ones(num_control_points)
    if num_control_points == 0: return epsilons, significance_weights
    if num_control_points > 1:
        dist_matrix_centers = cdist(rbf_centers, rbf_centers)
        for i in range(num_control_points):
            actual_k_neighbors = min(k_neighbors, num_control_points - 1)
            if actual_k_neighbors > 0:
                sorted_dists_to_others = np.sort(dist_matrix_centers[i, :])[1:actual_k_neighbors + 1]
                avg_knn_dist = np.mean(sorted_dists_to_others)
                if avg_knn_dist > 1e-6:
                    epsilons[i] = 1.0 / (avg_knn_dist * epsilon_scale_factor)
                else:
                    epsilons[i] = 1.0 / (1e-5 * epsilon_scale_factor)
            elif num_control_points == 2:
                avg_dist_all = dist_matrix_centers[i, 1 - i]
                if avg_dist_all > 1e-6:
                    epsilons[i] = 1.0 / (avg_dist_all * epsilon_scale_factor)
                else:
                    epsilons[i] = 1.0 / (1e-5 * epsilon_scale_factor)
            else:
                epsilons[i] = 1.0
    else:
        epsilons[0] = 1.0

    if pcmh_quality_scores is not None and len(pcmh_quality_scores) == num_control_points:
        scores = np.array(pcmh_quality_scores).astype(float)
        min_score, max_score = np.min(scores), np.max(scores)
        if (max_score - min_score) > 1e-9:
            normalized_scores = (scores - min_score) / (max_score - min_score)
            significance_weights = normalized_scores * 0.9 + 0.1
    return epsilons, significance_weights


# --- Refactored Registration Logic (EXACTLY AS PROVIDED BY USER IN PREVIOUS MESSAGE) ---
def run_registration_process(source_image_path_ui, target_image_path_ui, base_output_folder, progress_callback):
    output_paths = {
        'preprocess_source': None, 'preprocess_target': None,
        'affined_source': None, 'viz_affine_matches': None,
        'warped_source': None,  # This will be the final color warped image
        'img2_color_nr': None
    }
    try:
        progress_callback("初始化配准流程...")
        device_str = "cuda:0" if tc.cuda.is_available() else "cpu"
        progress_callback(f"使用设备: {device_str}")

        # --- 配准参数 (from user script) ---
        AFFINE_PX = 512
        N_SCALES_PARAM_PHASECONG = 6
        N_ORIENTS_PARAM_PHASECONG = 12
        INITIAL_FAST_CANDIDATES_MAX = 10000
        TARGET_KEYPOINT_COUNT_AFTER_SELECTION = 2000
        HARRIS_BLOCK_SIZE = 3
        HARRIS_KSIZE = 3
        HARRIS_K_PARAM = 0.04
        AFFINE_FSC_ERROR_THRESHOLD = 5.0
        AFFINE_PIXEL_ERROR_THRESHOLD = 5.0
        RBF_K_NEIGHBORS_FOR_EPSILON = 5
        RBF_EPSILON_SCALE_FACTOR = 0.5
        RBF_KERNEL_TYPE = 'gaussian'
        RBF_REGULARIZATION = 1e-6
        NR_PX = 2048
        NR_Params = dict()
        NR_Params['n_levels'] = 7
        NR_Params['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.8]
        NR_Params['n_iterations'] = [100, 100, 100, 100, 100, 100, 200]
        NR_Params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015]

        # --- 输出目录 ---
        s_name_base = os.path.splitext(os.path.basename(source_image_path_ui))[0]
        t_name_base = os.path.splitext(os.path.basename(target_image_path_ui))[0]
        s_name_sanitized = "".join(c if c.isalnum() else "_" for c in s_name_base)
        t_name_sanitized = "".join(c if c.isalnum() else "_" for c in t_name_base)

        pair_identifier = f"reg_{s_name_sanitized}_vs_{t_name_sanitized}"
        timestamp = get_current_timestamp_str()  # Use helper
        specific_run_folder_name = f"{pair_identifier}_{timestamp}"
        pair_output_dir = os.path.join(base_output_folder, specific_run_folder_name)

        if not os.path.exists(base_output_folder): os.makedirs(base_output_folder)
        if not os.path.exists(pair_output_dir): os.makedirs(pair_output_dir)
        progress_callback(f"输出将保存到: {pair_output_dir}")

        if tc.cuda.is_available(): tc.cuda.empty_cache()

        progress_callback(
            f"\n--- 处理图像对: {os.path.basename(source_image_path_ui)} vs {os.path.basename(target_image_path_ui)} ---")

        # Load source image
        source_ext = os.path.splitext(source_image_path_ui)[1].lower()
        if source_ext in ['.tif', '.tiff']:
            try:
                progress_callback(f"以TIFF格式加载源图像: {source_image_path_ui} ")
                # Assuming tk.load_image returns BGR, float32, [0,1]
                print(source_image_path_ui)
                source_cv_orig = tk.load_image(source_image_path_ui, level=2, load_slide=False)
            except Exception as e:
                progress_callback(f"错误: 无法使用OpenSlide加载源TIFF图像: {source_image_path_ui}. 错误: {e}")
                return output_paths
        else:
            progress_callback(f"以标准格式加载源图像: {source_image_path_ui}")
            source_cv_orig = cv2.imread(source_image_path_ui)
            if source_cv_orig is not None:
                source_cv_orig = source_cv_orig.astype(np.float32) / 255.0

        if source_cv_orig is None:
            progress_callback(f"错误: 无法读取源图像: {source_image_path_ui}")
            return output_paths

        # Load target image
        target_ext = os.path.splitext(target_image_path_ui)[1].lower()
        if target_ext in ['.tif', '.tiff']:
            try:
                progress_callback(f"以TIFF格式加载目标图像: {target_image_path_ui}")
                target_cv_orig = tk.load_image(target_image_path_ui, level=2, load_slide=False)
            except Exception as e:
                progress_callback(f"错误: 无法使用OpenSlide加载目标TIFF图像: {target_image_path_ui}. 错误: {e}")
                return output_paths
        else:
            progress_callback(f"以标准格式加载目标图像: {target_image_path_ui}")
            target_cv_orig = cv2.imread(target_image_path_ui)
            if target_cv_orig is not None:
                target_cv_orig = target_cv_orig.astype(np.float32) / 255.0

        if target_cv_orig is None:
            progress_callback(f"错误: 无法读取目标图像: {target_image_path_ui}")
            return output_paths
        # --- END MODIFIED Image Loading ---

        # --- 预处理 ---
        source_landmarks_orig, target_landmarks_orig = None, None
        progress_callback("--- 开始预处理 (Padding、Resize、灰度化、反相和CLAHE) ---")

        # Affine preprocessing
        # pf.preprocess returns: source_image_color_processed, source_image_processed, source_mask,
        #                        target_image_processed, target_mask, source_padding_info, target_padding_info
        _, _, img1_for_affine, _, img2_for_affine, _, _, _ = pf.preprocess(
            source_cv_orig, target_cv_orig, source_landmarks_orig, target_landmarks_orig, device_str, AFFINE_PX)

        # Non-rigid preprocessing
        img1_color_nr, img2_color_nr, img1_for_nr, _, img2_for_nr, _, _, _ = pf.preprocess(
            source_cv_orig, target_cv_orig, source_landmarks_orig, target_landmarks_orig, device_str, NR_PX)

        # Save preprocessed images (assuming they are float [0,1] from pf.preprocess)
        output_paths['preprocess_source'] = os.path.join(pair_output_dir, "preprocess_source.png")
        cv2.imwrite(output_paths['preprocess_source'], (img1_for_affine).astype(np.uint8))
        progress_callback(f"INTERMEDIATE:preprocess_source:{output_paths['preprocess_source']}")

        output_paths['preprocess_target'] = os.path.join(pair_output_dir, "preprocess_target.png")
        cv2.imwrite(output_paths['preprocess_target'], (img2_for_affine).astype(np.uint8))
        progress_callback(f"INTERMEDIATE:preprocess_target:{output_paths['preprocess_target']}")
        progress_callback("--- 预处理处理完成 ---")

        if tc.cuda.is_available(): tc.cuda.empty_cache()

        # --- 仿射配准 ---
        progress_callback("\n--- 开始仿射配准 ---")
        progress_callback("》关键点和描述符计算《")

        # Convert to uint8 [0,255] for kps_des_calculation
        img1_affine_uint8 = (img1_for_affine * 255).astype(np.uint8)
        img2_affine_uint8 = (img2_for_affine * 255).astype(np.uint8)

        kps1_coords_from_pcmoh, des_m1_array, kps2_coords_from_pcmoh, des_m2_array = kps_des_calculation(
            img1_affine_uint8, img2_affine_uint8, N_SCALES_PARAM_PHASECONG,
            N_ORIENTS_PARAM_PHASECONG, INITIAL_FAST_CANDIDATES_MAX,
            TARGET_KEYPOINT_COUNT_AFTER_SELECTION,
            HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K_PARAM)

        if not (des_m1_array.size > 0 and des_m2_array.size > 0 and
                kps1_coords_from_pcmoh is not None and kps1_coords_from_pcmoh.shape[0] > 0 and
                kps2_coords_from_pcmoh is not None and kps2_coords_from_pcmoh.shape[0] > 0):
            progress_callback("错误: 描述子计算失败或未产生足够的关键点。跳过此图像对的配准。")
            return output_paths

        progress_callback("》计算仿射变换矩阵《")
        H_affine_best, affine_pts1_unique_best, affine_pts2_unique_best, affine_match_distances_best, affine_inliers_mask_best, max_inliers_affine = \
            kps_match(des_m1_array, des_m2_array, kps1_coords_from_pcmoh, kps2_coords_from_pcmoh,
                      AFFINE_FSC_ERROR_THRESHOLD, AFFINE_PIXEL_ERROR_THRESHOLD)

        cleaned_pts1_affine = np.empty((0, 2), dtype=np.float32)
        quality_scores_rbf = np.array([])
        img1_affined_for_rbf = None  # Low-res affined for RBF input
        img1_affined_hr = None  # High-res grayscale affined (for RBF warp target)
        img1_affined_color_hr = None  # High-res color affined (for saving and final ADAM warp target)
        img1_warped_rbf = None  # High-res grayscale RBF warped (ADAM input)
        img1_warped_color_rbf = None  # High-res color RBF warped (for ADAM final warp)

        if max_inliers_affine > 0 and H_affine_best is not None and H_affine_best.shape == (3, 3):
            cleaned_pts1_affine = affine_pts1_unique_best[affine_inliers_mask_best]
            cleaned_pts2_affine = affine_pts2_unique_best[affine_inliers_mask_best]
            distances_of_inliers = affine_match_distances_best[affine_inliers_mask_best]

            if len(distances_of_inliers) > 0:
                quality_scores_rbf = 1.0 / (distances_of_inliers + 1e-6)
                if np.max(quality_scores_rbf) - np.min(quality_scores_rbf) > 1e-9:
                    quality_scores_rbf = (quality_scores_rbf - np.min(quality_scores_rbf)) / \
                                         (np.max(quality_scores_rbf) - np.min(quality_scores_rbf))
                else:
                    quality_scores_rbf = np.ones_like(distances_of_inliers)
            else:
                quality_scores_rbf = np.array([])
            progress_callback(f"最终仿射清洁匹配点数: {cleaned_pts1_affine.shape[0]}")

            h_affine_src, w_affine_src = img1_affine_uint8.shape[:2]
            h_affine_tgt, w_affine_tgt = img2_affine_uint8.shape[:2]
            img1_affined_for_rbf = cv2.warpPerspective(img1_affine_uint8, H_affine_best, (w_affine_tgt, h_affine_tgt),
                                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=0)

            h_nr_src, w_nr_src = img1_for_nr.shape[:2]  # img1_for_nr is float [0,1]
            h_nr_tgt, w_nr_tgt = img2_for_nr.shape[:2]  # img2_for_nr is float [0,1]

            if not (
                    w_affine_src == 0 or h_affine_src == 0 or w_nr_src == 0 or h_nr_src == 0 or w_affine_tgt == 0 or h_affine_tgt == 0):
                scale_x_nr_to_affine_src = w_affine_src / w_nr_src
                scale_y_nr_to_affine_src = h_affine_src / h_nr_src
                S_down_src = np.array([[scale_x_nr_to_affine_src, 0, 0], [0, scale_y_nr_to_affine_src, 0], [0, 0, 1]],
                                      dtype=np.float64)
                scale_x_affine_tgt_to_nr_tgt = w_nr_tgt / w_affine_tgt
                scale_y_affine_tgt_to_nr_tgt = h_nr_tgt / h_affine_tgt
                S_up_tgt = np.array(
                    [[scale_x_affine_tgt_to_nr_tgt, 0, 0], [0, scale_y_affine_tgt_to_nr_tgt, 0], [0, 0, 1]],
                    dtype=np.float64)
                H_nr_adjusted = S_up_tgt @ H_affine_best @ S_down_src

                # Warp high-res images (inputs are float [0,1], convert to uint8 for warp, then result is uint8)
                img1_affined_hr = cv2.warpPerspective((img1_for_nr).astype(np.uint8), H_nr_adjusted,
                                                      (w_nr_tgt, h_nr_tgt),
                                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                      borderValue=0)
                img1_affined_color_hr = cv2.warpPerspective((img1_color_nr * 255).astype(np.uint8), H_nr_adjusted,
                                                            (w_nr_tgt, h_nr_tgt),
                                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                            borderValue=0)

                output_paths['affined_source'] = os.path.join(pair_output_dir, "affined_source.png")
                cv2.imwrite(output_paths['affined_source'], img1_affined_color_hr)
                progress_callback(f"INTERMEDIATE:affined_source:{output_paths['affined_source']}")
            else:
                progress_callback("错误：仿射阶段图像尺寸为零，无法计算缩放因子。跳过高分辨率仿射变换。")
                # Fallback: if affine warp for HR fails, RBF and ADAM might not proceed well.
                # For now, let RBF section handle img1_affined_hr being None.

            kp1_viz_affine = [cv2.KeyPoint(p[0], p[1], 3) for p in cleaned_pts1_affine]
            kp2_viz_affine = [cv2.KeyPoint(p[0], p[1], 3) for p in cleaned_pts2_affine]
            matches_viz_affine = [cv2.DMatch(i, i, 0) for i in range(len(kp1_viz_affine))]
            img_matches_viz = cv2.drawMatches(img1_affine_uint8, kp1_viz_affine, img2_affine_uint8,
                                              kp2_viz_affine, matches_viz_affine, None, matchColor=(0, 255, 0),
                                              singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
            output_paths['viz_affine_matches'] = os.path.join(pair_output_dir, "viz_affine_matches.png")
            cv2.imwrite(output_paths['viz_affine_matches'], img_matches_viz)
            progress_callback(f"INTERMEDIATE:viz_affine_matches:{output_paths['viz_affine_matches']}")
            progress_callback("--- 仿射配准结束 ---")

            # --- 自适应RBF非刚性配准 ---
            # RBF operates on img1_affined_for_rbf (low-res uint8 affined)
            # Then its displacement is applied to img1_affined_hr and img1_affined_color_hr (high-res uint8 affined)
            if img1_affined_for_rbf is not None and img1_affined_hr is not None and \
                    cleaned_pts1_affine.shape[0] >= max(4, RBF_K_NEIGHBORS_FOR_EPSILON):
                progress_callback(f"\n--- 开始自适应RBF非刚性配准 (控制点: {cleaned_pts1_affine.shape[0]}) ---")

                num_ctrl_pts_rbf = cleaned_pts1_affine.shape[0]
                pts1_affine_hom_rbf = np.hstack((cleaned_pts1_affine, np.ones((num_ctrl_pts_rbf, 1))))
                transformed_pts1_hom_rbf = (H_affine_best @ pts1_affine_hom_rbf.T).T
                w_rbf_h = transformed_pts1_hom_rbf[:, 2]
                w_rbf_h[np.abs(w_rbf_h) < 1e-9] = 1e-9
                rbf_centers = transformed_pts1_hom_rbf[:, :2] / w_rbf_h[:, np.newaxis]
                displacements_rbf = cleaned_pts2_affine - rbf_centers

                adaptive_eps, significance_w = compute_adaptive_epsilon_and_weights(
                    rbf_centers, pcmh_quality_scores=quality_scores_rbf,
                    k_neighbors=RBF_K_NEIGHBORS_FOR_EPSILON, epsilon_scale_factor=RBF_EPSILON_SCALE_FACTOR)
                epsilon_phi = np.mean(adaptive_eps) if len(adaptive_eps) > 0 else 1.0
                if RBF_KERNEL_TYPE == 'thin_plate': epsilon_phi = 1.0

                dist_centers_rbf = cdist(rbf_centers, rbf_centers)
                phi_mat = rbf_kernel_function(dist_centers_rbf, epsilon_phi, kernel_type=RBF_KERNEL_TYPE)
                weighted_phi = np.diag(np.sqrt(significance_w)) @ phi_mat @ np.diag(np.sqrt(significance_w))
                weighted_disp = np.diag(np.sqrt(significance_w)) @ displacements_rbf
                phi_mat_reg = weighted_phi + np.eye(num_ctrl_pts_rbf) * RBF_REGULARIZATION

                rbf_w_intermediate = None
                try:
                    rbf_w_intermediate = np.linalg.solve(phi_mat_reg, weighted_disp)
                except np.linalg.LinAlgError:
                    try:
                        rbf_w_intermediate = np.linalg.pinv(phi_mat_reg) @ weighted_disp
                    except Exception as e_p:
                        progress_callback(f"RBF伪逆错误: {e_p}")

                if rbf_w_intermediate is not None:
                    rbf_w = np.diag(np.sqrt(significance_w)) @ rbf_w_intermediate
                    h_rbf_input, w_rbf_input = img1_affined_for_rbf.shape[:2]  # Low-res dimensions
                    yy_g, xx_g = np.meshgrid(np.arange(h_rbf_input), np.arange(w_rbf_input), indexing='ij')
                    pix_coords_rbf = np.vstack((xx_g.ravel(), yy_g.ravel())).T
                    interp_disp_xy = np.zeros_like(pix_coords_rbf, dtype=float)
                    for c_i in range(num_ctrl_pts_rbf):
                        dist_to_c = cdist(pix_coords_rbf, rbf_centers[c_i, np.newaxis, :])
                        kern_val = rbf_kernel_function(dist_to_c, adaptive_eps[c_i], kernel_type=RBF_KERNEL_TYPE)
                        interp_disp_xy += kern_val * rbf_w[c_i, :]

                    # Low-res displacement field components
                    rbf_displacement_field_x = interp_disp_xy[:, 0].reshape(h_rbf_input, w_rbf_input)
                    rbf_displacement_field_y = interp_disp_xy[:, 1].reshape(h_rbf_input, w_rbf_input)

                    # Low-res map (absolute coordinates for remap)
                    map_x_rbf_lowres = (xx_g + rbf_displacement_field_x).astype(np.float32)
                    map_y_rbf_lowres = (yy_g + rbf_displacement_field_y).astype(np.float32)

                    h_hr, w_hr = img1_affined_hr.shape[:2]  # High-res dimensions

                    # Upscale the displacement map components, not absolute coordinates directly.
                    # Then scale the displacement values.
                    # An alternative, more robust way is to scale control points and re-evaluate RBF sums for high-res grid.
                    # User's script scales the absolute map values:
                    sf_x = w_hr / w_rbf_input
                    sf_y = h_hr / h_rbf_input
                    map_x_rbf_u = cv2.resize(map_x_rbf_lowres, (w_hr, h_hr), interpolation=cv2.INTER_LINEAR)
                    map_y_rbf_u = cv2.resize(map_y_rbf_lowres, (w_hr, h_hr), interpolation=cv2.INTER_LINEAR)
                    map_x_final_highres = map_x_rbf_u * sf_x
                    map_y_final_highres = map_y_rbf_u * sf_y

                    img1_warped_rbf = cv2.remap(img1_affined_hr, map_x_final_highres, map_y_final_highres,
                                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)
                    img1_warped_color_rbf = cv2.remap(img1_affined_color_hr, map_x_final_highres, map_y_final_highres,
                                                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                      borderValue=0)
                    progress_callback("--- 自适应RBF非刚性配准结束 ---")
                else:
                    progress_callback("RBF权重计算失败，跳过RBF非刚性变形。")
                    img1_warped_rbf = img1_affined_hr.copy() if img1_affined_hr is not None else None
                    img1_warped_color_rbf = img1_affined_color_hr.copy() if img1_affined_color_hr is not None else None
            elif img1_affined_for_rbf is None or img1_affined_hr is None:
                progress_callback("跳过RBF：仿射变换图像（低分辨率或高分辨率）不可用。")
                img1_warped_rbf = img1_affined_hr.copy() if img1_affined_hr is not None else None
                img1_warped_color_rbf = img1_affined_color_hr.copy() if img1_affined_color_hr is not None else None
            else:  # Not enough control points
                progress_callback(
                    f"跳过RBF：控制点数不足 ({cleaned_pts1_affine.shape[0]} < {max(4, RBF_K_NEIGHBORS_FOR_EPSILON)})。")
                img1_warped_rbf = img1_affined_hr.copy() if img1_affined_hr is not None else None
                img1_warped_color_rbf = img1_affined_color_hr.copy() if img1_affined_color_hr is not None else None

            # --- ADAM 非刚性配准 ---
            if img1_warped_rbf is not None and img1_warped_color_rbf is not None:  # Check if RBF stage produced valid inputs
                # adam_input_source_high_res is uint8 grayscale [0,255] from RBF
                adam_input_source_high_res = img1_warped_rbf
                # adam_target_high_res is uint8 grayscale [0,255]
                adam_target_high_res = (img2_for_nr * 255).astype(np.uint8)  # img2_for_nr is float [0,1]

                progress_callback("\n--- 开始ADAM非刚性配准 ---")
                source_nr_pr, _ = tk.create_pyramid(tk.image_to_tensor(adam_input_source_high_res / 255.0, device_str),
                                                    NR_Params['n_levels'])
                target_nr_pr, _ = tk.create_pyramid(tk.image_to_tensor(adam_target_high_res / 255.0, device_str),
                                                    NR_Params['n_levels'])
                opt_df_adam = None

                for i in range(NR_Params['n_levels']):
                    opt_source = source_nr_pr[i]
                    opt_target = target_nr_pr[i]
                    progress_callback(
                        f"  ADAM Lvl {i + 1}/{NR_Params['n_levels']}, Src: {opt_source.shape}, Tgt: {opt_target.shape}")
                    if i == 0:
                        current_df = tk.create_identity_displacement_field(opt_source).detach().clone()
                    else:
                        current_df = tk.resample_displacement_field_to_size(opt_df_adam, (
                        opt_source.size(2), opt_source.size(3))).detach().clone()
                    current_df.requires_grad = True
                    optimizer = tc.optim.Adam([current_df], NR_Params['learning_rates'][i])
                    for j in range(NR_Params['n_iterations'][i]):
                        optimizer.zero_grad()
                        warped_s_adam, _ = tk.warp_tensor(opt_source, current_df, device=device_str)
                        cost = cf.ncc_local(warped_s_adam, opt_target, win_size=7, device=device_str)
                        reg = rl.diffusion_relative(current_df)
                        loss = cost + NR_Params['alphas'][i] * reg
                        loss.backward()
                        optimizer.step()
                        if (j + 1) % 20 == 0 or j == NR_Params['n_iterations'][i] - 1:
                            progress_callback(
                                f"    Iter {j + 1}/{NR_Params['n_iterations'][i]}, Loss: {loss.item():.4f}, Cost: {cost.item():.4f}, Reg: {reg.item():.4f}")
                    opt_df_adam = current_df.detach().clone()

                # Apply final ADAM DF to img1_warped_color_rbf (uint8 color [0,255])
                final_s_tensor_for_adam = tk.image_to_tensor(img1_warped_color_rbf / 255,
                                                             device_str)  # To float [0,1] tensor
                warped_s_final_adam_tensor, _ = tk.warp_tensor(final_s_tensor_for_adam, opt_df_adam, device=device_str)

                # Convert tensor to uint8 numpy array (H, W, C)
                img1_final_warped_adam = (
                            warped_s_final_adam_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
                    np.uint8)

                output_paths['warped_source'] = os.path.join(pair_output_dir, "warped_source.png")
                cv2.imwrite(output_paths['warped_source'], img1_final_warped_adam)
                progress_callback(f"INTERMEDIATE:warped_source:{output_paths['warped_source']}")

                warped_st_gray, _ = tk.warp_tensor(tk.image_to_tensor(img1_warped_rbf / 255,
                                                             device_str), opt_df_adam, device=device_str)
                img1_st_gray = (
                        warped_st_gray.squeeze().detach().cpu().numpy() * 255).astype(
                    np.uint8)
                diff_final_adam = cv2.absdiff(img1_st_gray, img2_for_nr)
                #output_paths['diff_final_nr'] = os.path.join(pair_output_dir, "diff_final_nr.png")
                #cv2.imwrite(output_paths['diff_final_nr'], diff_final_adam)

                output_paths['img2_color_nr'] = os.path.join(pair_output_dir, "target.png")
                cv2.imwrite(output_paths['img2_color_nr'], (img2_color_nr*255).astype(
                    np.uint8))

                #progress_callback(f"INTERMEDIATE:diff_final_nr:{output_paths['diff_final_nr']}")
                progress_callback("--- 非刚性配准结束 ---")
            else:
                progress_callback("跳过ADAM：RBF阶段的输入图像无效。")
        else:
            progress_callback("仿射配准未找到足够内点或有效变换矩阵，跳过后续非刚性步骤。")

        progress_callback(f"--- 图像对处理完毕 ---")
        return output_paths

    except Exception as e:
        progress_callback(f"配准过程中发生严重错误: {e}\n{traceback.format_exc()}")
        return output_paths  # Return whatever paths were generated before error


# --- PyQt5 GUI Classes ---

# --- 新增：可缩放和拖动的图像查看器 ---
class ZoomableImageViewer(QDialog):
    def __init__(self, pixmap, title="放大图像", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)  # Initial size of the dialog

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.pixmap_item)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("放大 (+)")
        zoom_in_button.clicked.connect(lambda: self.zoom(1.2))
        zoom_out_button = QPushButton("缩小 (-)")
        zoom_out_button.clicked.connect(lambda: self.zoom(1 / 1.2))
        reset_button = QPushButton("重置视图")
        reset_button.clicked.connect(self.reset_view)

        zoom_layout.addWidget(zoom_out_button)
        zoom_layout.addWidget(zoom_in_button)
        zoom_layout.addWidget(reset_button)
        layout.addLayout(zoom_layout)

        self.view.setSceneRect(self.pixmap_item.boundingRect())

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom(zoom_factor)
        event.accept()

    def zoom(self, factor):
        self.view.scale(factor, factor)

    def reset_view(self):
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)


class RegistrationWorker(QObject):  # Unchanged
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    intermediate_image = pyqtSignal(str, str)

    def __init__(self, source_path, target_path, output_base):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.output_base = output_base
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.progress.emit("尝试停止配准...")

    def _progress_callback(self, message):
        if not self._is_running:
            raise InterruptedError("Registration stopped by user request.")
        if message.startswith("INTERMEDIATE:"):
            try:
                _, img_type, img_path = message.split(":", 2)
                if self._is_running: self.intermediate_image.emit(img_type, img_path)
            except ValueError:
                if self._is_running: self.progress.emit(message)
        else:
            if self._is_running: self.progress.emit(message)

    def run(self):
        try:
            output_paths = run_registration_process(
                self.source_path, self.target_path, self.output_base, self._progress_callback
            )
            if self._is_running: self.finished.emit(output_paths)
        except InterruptedError:
            self.progress.emit("配准已由用户停止。")
            self.finished.emit(None)
        except Exception as e:
            if self._is_running:
                detailed_error = f"Error in registration thread: {e}\n{traceback.format_exc()}"
                self.error.emit(detailed_error)


class ImageDisplayWidget(QWidget):  # Modified for zoom
    def __init__(self, title="Image"):
        super().__init__()
        self.path = None
        self.pixmap = None
        self.title_str = title
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel("点击加载图像\n(双击放大)")  # Modified
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 150)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.image_label.setAutoFillBackground(True)
        palette = self.image_label.palette()
        palette.setColor(QPalette.Window, QColor('lightgray'))
        self.image_label.setPalette(palette)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def set_image_from_path(self, image_path):
        self.path = image_path
        self.pixmap = None  # Reset pixmap

        if image_path and os.path.exists(image_path):
            file_ext = os.path.splitext(image_path)[1].lower()
            temp_pixmap = None

            if file_ext in ['.tif', '.tiff']:
                try:
                    # 使用 tk.load_image 加载 level 0 用于预览
                    # 确保 tk.load_image 返回的是适合显示的 NumPy 数组 (例如 uint8, [0,255])
                    # 并且是 RGB 顺序 (QImage.Format_RGB888) 或 BGR (然后用 cv2.cvtColor)

                    # 假设 tk.load_image 返回的是 float32 [0,1] BGR (基于你之前的 tk.load_image)
                    # 我们需要将其转换为 uint8 [0,255] RGB for QImage
                    img_data_bgr_float = tk.load_image(image_path, level=2, load_slide=False)  # level 0 for preview

                    if img_data_bgr_float is not None:
                        img_data_bgr_uint8 = (img_data_bgr_float * 255).astype(np.uint8)

                        # 如果 img_data_bgr_uint8 可能是空的或形状不对
                        if img_data_bgr_uint8.size == 0 or len(img_data_bgr_uint8.shape) != 3:
                            raise ValueError("Loaded TIFF data has incorrect shape or is empty.")

                        height, width, channel = img_data_bgr_uint8.shape
                        if channel == 3:  # BGR
                            # QImage 需要 RGB 数据
                            img_data_rgb_uint8 = cv2.cvtColor(img_data_bgr_uint8, cv2.COLOR_BGR2RGB)
                            q_image = QImage(img_data_rgb_uint8.data, width, height, width * channel,
                                             QImage.Format_RGB888)
                            temp_pixmap = QPixmap.fromImage(q_image)
                        elif channel == 1:  # Grayscale - 假设你tk.load_image可能返回灰度
                            q_image = QImage(img_data_bgr_uint8.data, width, height, width, QImage.Format_Grayscale8)
                            temp_pixmap = QPixmap.fromImage(q_image)
                        else:
                            self.image_label.setText(f"TIFF通道数不支持:\n{os.path.basename(image_path)}\n(双击放大)")
                            return
                    else:  # tk.load_image returned None
                        self.image_label.setText(f"tk.load_image无法加载:\n{os.path.basename(image_path)}\n(双击放大)")
                        return

                except Exception as e:
                    print(f"Error loading TIFF preview for {image_path} with tk.load_image: {e}")  # Log for debugging
                    traceback.print_exc()
                    self.image_label.setText(f"TIFF预览错误:\n{os.path.basename(image_path)}\n(双击放大)")
                    return  # Important to return here
            else:  # For non-TIFF images
                temp_pixmap = QPixmap(image_path)

            # Check temp_pixmap after attempting to load
            if temp_pixmap and not temp_pixmap.isNull():
                self.pixmap = temp_pixmap
                self._update_pixmap_display()
            else:
                self.image_label.setText(f"无法加载/显示:\n{os.path.basename(image_path)}\n(双击放大)")
                self.pixmap = None  # Ensure pixmap is None if loading failed
                # self._update_pixmap_display() # Not strictly needed if text is set above
        else:  # Path is None or does not exist
            self.image_label.setText(f"图像不存在\n或未提供\n(双击放大)")
            self.image_label.setPixmap(QPixmap())  # Clear existing pixmap
            self.pixmap = None

    def _update_pixmap_display(self):
        if self.pixmap and not self.pixmap.isNull():
            self.image_label.setPixmap(self.pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            base_text = f"图像未加载"
            if self.path and not (self.pixmap and not self.pixmap.isNull()):
                base_text = f"无法显示:\n{os.path.basename(self.path if self.path else 'N/A')}"
            elif not self.path:
                base_text = f"图像未加载"
            self.image_label.setText(f"{base_text}\n(双击放大)")
            self.image_label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap_display()

    def mouseDoubleClickEvent(self, event):  # Added for zoom
        if event.button() == Qt.LeftButton and self.pixmap and not self.pixmap.isNull():
            viewer = ZoomableImageViewer(self.pixmap, title=self.title_str, parent=self.window())
            viewer.exec_()
        super().mouseDoubleClickEvent(event)


class RegistrationApp(QMainWindow):  # Unchanged
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像配准工具 v1.2 (带缩放)")  # Updated version
        self.setGeometry(50, 50, 1600, 950)

        self.source_image_path = ""
        self.target_image_path = ""
        self.output_directory = os.path.join(os.getcwd(), "registration_gui_outputs")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        input_layout = QHBoxLayout()
        self.source_button = QPushButton("1. 选择源图像")
        self.source_button.clicked.connect(lambda: self.load_image_dialog('source'))
        self.source_path_label = QLineEdit("未选择")
        self.source_path_label.setReadOnly(True)
        input_layout.addWidget(self.source_button)
        input_layout.addWidget(self.source_path_label, 1)

        self.target_button = QPushButton("2. 选择目标图像")
        self.target_button.clicked.connect(lambda: self.load_image_dialog('target'))
        self.target_path_label = QLineEdit("未选择")
        self.target_path_label.setReadOnly(True)
        input_layout.addWidget(self.target_button)
        input_layout.addWidget(self.target_path_label, 1)
        main_layout.addLayout(input_layout)

        image_grid_layout = QGridLayout()
        self.displays = {
            'source_orig': ImageDisplayWidget("源图像 (原始)"),
            'target_orig': ImageDisplayWidget("目标图像 (原始)"),
            'preprocess_source': ImageDisplayWidget("源图像 (预处理)"),
            'preprocess_target': ImageDisplayWidget("目标图像 (预处理)"),
            'viz_affine_matches': ImageDisplayWidget("仿射匹配点"),
            'affined_source': ImageDisplayWidget("源图像 (仿射后彩色)"),
            'warped_source': ImageDisplayWidget("最终配准结果 (彩色)"),
            'img2_color_nr': ImageDisplayWidget("目标图像 (Pad)")
        }
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        keys = list(self.displays.keys())
        for i, key in enumerate(keys):
            image_grid_layout.addWidget(self.displays[key], positions[i][0], positions[i][1])

        for r in range(2): image_grid_layout.setRowStretch(r, 1)
        for c in range(4): image_grid_layout.setColumnStretch(c, 1)
        main_layout.addLayout(image_grid_layout, 1)

        control_layout = QHBoxLayout()
        self.register_button = QPushButton("3. 开始配准")
        self.register_button.clicked.connect(self.start_registration)
        self.register_button.setFixedHeight(40)
        self.stop_button = QPushButton("停止配准")
        self.stop_button.clicked.connect(self.stop_registration)
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedHeight(40)
        control_layout.addWidget(self.register_button)
        control_layout.addWidget(self.stop_button)
        main_layout.addLayout(control_layout)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setFixedHeight(150)
        main_layout.addWidget(self.status_log)

        self.thread = None
        self.worker = None

    def load_image_dialog(self, image_type):
        path, _ = QFileDialog.getOpenFileName(self, f"选择{image_type}图像", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            if image_type == 'source':
                self.source_image_path = path
                self.source_path_label.setText(path)
                self.displays['source_orig'].set_image_from_path(path)
            elif image_type == 'target':
                self.target_image_path = path
                self.target_path_label.setText(path)
                self.displays['target_orig'].set_image_from_path(path)
            self.log_message(f"{image_type.capitalize()} 图像已加载: {os.path.basename(path)}")

    def log_message(self, message):
        self.status_log.append(message)
        QApplication.processEvents()

    def start_registration(self):
        if not self.source_image_path or not self.target_image_path:
            QMessageBox.warning(self, "输入错误", "请先选择源图像和目标图像。")
            return

        for key, disp_widget in self.displays.items():
            if key not in ['source_orig', 'target_orig']:
                disp_widget.set_image_from_path(None)

        self.register_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_message("开始配准过程...")

        self.thread = QThread()
        self.worker = RegistrationWorker(self.source_image_path, self.target_image_path, self.output_directory)
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.log_message)
        self.worker.intermediate_image.connect(self.display_intermediate_image)
        self.worker.finished.connect(self.on_registration_finished)
        self.worker.error.connect(self.on_registration_error)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def stop_registration(self):
        if self.worker: self.worker.stop()
        self.stop_button.setEnabled(False)

    def display_intermediate_image(self, image_type, image_path):
        self.log_message(f"显示中间图像: {image_type} - {os.path.basename(image_path)}")
        if image_type in self.displays:
            self.displays[image_type].set_image_from_path(image_path)
        else:
            self.log_message(f"警告: 未知的中间图像类型 '{image_type}' (路径: {image_path})")

    def on_registration_finished(self, output_paths_dict):
        self.log_message("配准线程结束。")
        if output_paths_dict is None:
            self.log_message("配准未成功完成或被用户中止。")
        else:
            self.log_message("配准完成。正在更新最终图像显示：")
            for img_type, img_path in output_paths_dict.items():
                if img_path and os.path.exists(img_path):
                    self.display_intermediate_image(img_type, img_path)
                elif img_path:
                    self.log_message(f"警告: 图像文件 {img_type} 在 {img_path} 未找到。")

        self.register_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.thread = None
        self.worker = None

    def on_registration_error(self, error_message):
        self.log_message(f"配准错误: {error_message}")
        QMessageBox.critical(self, "配准错误", error_message)
        self.register_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.thread = None
        self.worker = None

    def closeEvent(self, event):
        if self.worker and hasattr(self.worker, '_is_running') and self.worker._is_running:
            reply = QMessageBox.question(self, '退出确认',
                                         "配准仍在进行中。确定要退出吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_registration()
                if self.thread and self.thread.isRunning(): self.thread.quit(); self.thread.wait(1000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = RegistrationApp()
    main_window.show()
    sys.exit(app.exec_())