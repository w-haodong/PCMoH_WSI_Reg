import cv2
import numpy as np
from phasepack import phasecong
from reg_function.PCMOH import compute_pcmoh_descriptors
from reg_function.FSC import fsc

def normalize_p(img):
    img_float = img.astype(np.float32)
    a = np.max(img_float)
    b = np.min(img_float)
    if (a - b) < 1e-9:
        return np.zeros_like(img_float, dtype=img_float.dtype)
    return (img_float - b) / (a - b)

def transform_eo_to_numpy_array(eo_orig_orient_scale, nscale, norient):
    eo_np_array = np.empty((nscale, norient), dtype=object)
    if not (isinstance(eo_orig_orient_scale, list) and
            all(isinstance(sublist, list) for sublist in eo_orig_orient_scale) and
            len(eo_orig_orient_scale) == norient and
            (len(eo_orig_orient_scale) == 0 or (norient > 0 and len(eo_orig_orient_scale[0]) == nscale))):
        actual_norient_val = len(eo_orig_orient_scale) if isinstance(eo_orig_orient_scale, list) else -1
        actual_nscale_val = -1
        if actual_norient_val > 0 and isinstance(eo_orig_orient_scale[0], list):
            actual_nscale_val = len(eo_orig_orient_scale[0])
        raise ValueError(
            f"EO structure mismatch. Expected {norient} orientations and {nscale} scales for [orient][scale] list input, "
            f"but received structure implying {actual_norient_val} orientations and {actual_nscale_val} scales for the first orientation.")
    for o_idx in range(norient):
        for s_idx in range(nscale):
            eo_np_array[s_idx, o_idx] = eo_orig_orient_scale[o_idx][s_idx]
    return eo_np_array

def select_keypoints_by_harris_response(
        image_norm_float32: np.ndarray,
        initial_fast_kps: list,
        target_count: int,
        harris_block_size: int = 3,
        harris_ksize: int = 3,
        harris_k: float = 0.04
) -> list:
    if not initial_fast_kps or target_count <= 0: return []
    img_for_harris = image_norm_float32.astype(np.float32)
    harris_responses_img = cv2.cornerHarris(img_for_harris, harris_block_size, harris_ksize, harris_k)
    kp_new_responses = []
    valid_kps_for_sort = []
    img_h, img_w = harris_responses_img.shape
    for kp in initial_fast_kps:
        c_idx, r_idx = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= r_idx < img_h and 0 <= c_idx < img_w:
            kp_new_responses.append(harris_responses_img[r_idx, c_idx])
            valid_kps_for_sort.append(kp)
    if not valid_kps_for_sort: return []
    sorted_indices_by_new_response = np.argsort(kp_new_responses)[::-1]
    selected_kps = [valid_kps_for_sort[i] for i in sorted_indices_by_new_response[:target_count]]
    return selected_kps

def kps_des_calculation(img1_for_process,img2_for_process, N_SCALES_PARAM_PHASECONG,
                        N_ORIENTS_PARAM_PHASECONG, INITIAL_FAST_CANDIDATES_MAX, TARGET_KEYPOINT_COUNT_AFTER_SELECTION,
                        HARRIS_BLOCK_SIZE,HARRIS_KSIZE, HARRIS_K_PARAM):

    print("1-运行相位一致性")
    m1_moment, _, _, _, _, eo1_orient_scale, _ = phasecong(img=img1_for_process,
                                                           nscale=N_SCALES_PARAM_PHASECONG,
                                                           norient=N_ORIENTS_PARAM_PHASECONG, minWaveLength=3, mult=1.6,
                                                           sigmaOnf=0.75, g=10.0, k=2.0)
    m1_norm = normalize_p(m1_moment)
    m2_moment, _, _, _, _, eo2_orient_scale, _ = phasecong(img=img2_for_process,
                                                           nscale=N_SCALES_PARAM_PHASECONG,
                                                           norient=N_ORIENTS_PARAM_PHASECONG, minWaveLength=3, mult=1.6,
                                                           sigmaOnf=0.75, g=10.0, k=2.0)
    m2_norm = normalize_p(m2_moment)

    eo1_scale_orient_np = transform_eo_to_numpy_array(eo1_orient_scale, N_SCALES_PARAM_PHASECONG,
                                                      N_ORIENTS_PARAM_PHASECONG)
    eo2_scale_orient_np = transform_eo_to_numpy_array(eo2_orient_scale, N_SCALES_PARAM_PHASECONG,
                                                      N_ORIENTS_PARAM_PHASECONG)

    fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    print("2-对图像进行关键点检测")
    kp1_cv_initial_fast = fast.detect((m1_norm * 255).astype(np.uint8), None)
    kp1_cv_initial_fast = sorted(kp1_cv_initial_fast, key=lambda kp: kp.response, reverse=True)[
                          :min(INITIAL_FAST_CANDIDATES_MAX, len(kp1_cv_initial_fast))]
    kp1_cv_selected = select_keypoints_by_harris_response(m1_norm, kp1_cv_initial_fast,
                                                          TARGET_KEYPOINT_COUNT_AFTER_SELECTION, HARRIS_BLOCK_SIZE,
                                                          HARRIS_KSIZE, HARRIS_K_PARAM)
    kp2_cv_initial_fast = fast.detect((m2_norm * 255).astype(np.uint8), None)
    kp2_cv_initial_fast = sorted(kp2_cv_initial_fast, key=lambda kp: kp.response, reverse=True)[
                          :min(INITIAL_FAST_CANDIDATES_MAX, len(kp2_cv_initial_fast))]
    kp2_cv_selected = select_keypoints_by_harris_response(m2_norm, kp2_cv_initial_fast,
                                                          TARGET_KEYPOINT_COUNT_AFTER_SELECTION, HARRIS_BLOCK_SIZE,
                                                          HARRIS_KSIZE, HARRIS_K_PARAM)

    m1_point_coords_for_pcmoh = np.array([kp.pt for kp in kp1_cv_selected],
                                         dtype=np.float32)
    m2_point_coords_for_pcmoh = np.array([kp.pt for kp in kp2_cv_selected],
                                         dtype=np.float32)

    kps1_coords_from_pcmoh, des_m1_array = np.empty((0, 2), dtype=np.float32), np.empty(
        (N_ORIENTS_PARAM_PHASECONG, 0, 0), dtype=np.float64)
    kps2_coords_from_pcmoh, des_m2_array = np.empty((0, 2), dtype=np.float32), np.empty(
        (N_ORIENTS_PARAM_PHASECONG, 0, 0), dtype=np.float64)

    if m1_point_coords_for_pcmoh.shape[0] > 0:
        print(f"3-对源图像进行描述符评估：({m1_point_coords_for_pcmoh.shape[0]})...")
        kps1_coords_from_pcmoh, des_m1_array = compute_pcmoh_descriptors(m1_norm, m1_point_coords_for_pcmoh,
                                                                         eo1_scale_orient_np,
                                                                         N_SCALES_PARAM_PHASECONG,
                                                                         N_ORIENTS_PARAM_PHASECONG)
    if m2_point_coords_for_pcmoh.shape[0] > 0:
        print(f"3-对目标图像进行描述符评估：({m2_point_coords_for_pcmoh.shape[0]})...")
        kps2_coords_from_pcmoh, des_m2_array = compute_pcmoh_descriptors(m2_norm, m2_point_coords_for_pcmoh,
                                                                         eo2_scale_orient_np,
                                                                         N_SCALES_PARAM_PHASECONG,
                                                                         N_ORIENTS_PARAM_PHASECONG)
    return kps1_coords_from_pcmoh, des_m1_array, kps2_coords_from_pcmoh, des_m2_array

def kps_match(des_m1_array, des_m2_array, kps1_coords_from_pcmoh, kps2_coords_from_pcmoh, AFFINE_FSC_ERROR_THRESHOLD, AFFINE_PIXEL_ERROR_THRESHOLD):

    bf_affine = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    max_inliers_affine = 0

    if not (des_m1_array.ndim == 3 and des_m2_array.ndim == 3 and des_m1_array.shape[2] > 0 and des_m2_array.shape[
        2] > 0):
        print("警告: 描述子数组维度或大小不足以进行仿射匹配。")
    else:
        des1_match_affine = des_m1_array[0, :, :].T.astype(np.float32)
        if des1_match_affine.shape[0] > 0:
            for orient_idx in range(des_m2_array.shape[0]):
                des2_match_affine = des_m2_array[orient_idx, :, :].T.astype(np.float32)
                if des2_match_affine.shape[0] < 4: continue
                matches_affine_raw = bf_affine.match(des1_match_affine, des2_match_affine)
                if not matches_affine_raw or len(matches_affine_raw) < 4: continue
                idx_pairs = np.array([[m.queryIdx, m.trainIdx] for m in matches_affine_raw])
                match_distances = np.array([m.distance for m in matches_affine_raw])
                pts1_cand = kps1_coords_from_pcmoh[idx_pairs[:, 0]]
                pts2_cand = kps2_coords_from_pcmoh[idx_pairs[:, 1]]
                if pts2_cand.shape[0] > 0:
                    _, unique_idx = np.unique(pts2_cand, axis=0, return_index=True)
                    pts1_unique = pts1_cand[unique_idx]
                    pts2_unique = pts2_cand[unique_idx]
                    distances_unique = match_distances[unique_idx]
                else:
                    continue
                if pts1_unique.shape[0] < 4: continue
                try:
                    H_cand = fsc(pts1_unique.astype(np.float64), pts2_unique.astype(np.float64), 'affine',
                                 AFFINE_FSC_ERROR_THRESHOLD)
                    if H_cand is None: continue
                except Exception as e:
                    print(f"FSC错误: {e}");
                    continue
                pts1_h = np.vstack((pts1_unique.T, np.ones(pts1_unique.shape[0])))
                proj_pts1_h = H_cand @ pts1_h
                proj_pts1_h[2, np.abs(proj_pts1_h[2, :]) < 1e-9] = 1e-9
                proj_pts1_cart = proj_pts1_h[:2, :] / proj_pts1_h[2, :]
                errors = np.sqrt(np.sum((proj_pts1_cart - pts2_unique.T) ** 2, axis=0))
                inliers_mask = errors < AFFINE_PIXEL_ERROR_THRESHOLD
                num_inliers = np.sum(inliers_mask)
                if num_inliers > max_inliers_affine:
                    max_inliers_affine = num_inliers
                    affine_inliers_mask = errors < AFFINE_PIXEL_ERROR_THRESHOLD
                    affine_inliers_mask_best = affine_inliers_mask
                    affine_pts1_unique_best = pts1_unique
                    affine_pts2_unique_best = pts2_unique
                    H_affine_best = H_cand
                    affine_match_distances_best = distances_unique
                    if max_inliers_affine > 0: print(
                        f"*** 仿射找到最佳: {max_inliers_affine} 内点 (目标描述子方向 {orient_idx}) ***")
        else:
            print("警告: 源图像的描述子集为空(固定方向0)。")

        return H_affine_best, affine_pts1_unique_best, affine_pts2_unique_best, affine_match_distances_best, affine_inliers_mask_best, max_inliers_affine



