import numpy as np
from scipy.linalg import norm as scipy_norm
# PC-MoH (Phase Congruency Moment-weighted Orientation Histogram)
# --- Configuration Constants ---
PATCH_RADIUS_DIAG_PROPORTION_FACTOR = 26 * 1.8510
PATCH_RADIUS_MIN_DIM_FRACTION = 1.0 / 3.0
PATCH_RADIUS_DIAG_DIVISOR = 1200.0
RING_REGION_OUTER_RADIUS_FACTOR = 0.8
RADIAL_THRESH_FACTORS = [0.2, 0.4, 0.6, 0.8]
NUM_ACTUAL_LOG_POLAR_RINGS = 3  # Number of rings actively filled based on RADIAL_THRESH_FACTORS
NUM_LOG_POLAR_RINGS_HIST_DIM = 3  # <<--- 修改：直方图数组的环维度大小，与实际填充的环数量一致
DESCRIPTOR_NORMALIZATION_CLIP_VALUE = 0.2
NUMERICAL_STABILITY_EPSILON = 1e-9


# ---------------------------------------------------------------------------

def _precompute_log_polar_grid(patch_radius: int, num_descriptor_orientations: int):
    patch_coord_range = np.arange(-patch_radius, patch_radius + 1)
    patch_grid_x, patch_grid_y = np.meshgrid(patch_coord_range, patch_coord_range)

    ring_region_outer_radius = patch_radius * RING_REGION_OUTER_RADIUS_FACTOR
    radial_thresholds = [
        np.log2(max(NUMERICAL_STABILITY_EPSILON, ring_region_outer_radius * factor))
        for factor in RADIAL_THRESH_FACTORS
    ]

    distances_from_patch_center = np.sqrt(patch_grid_x ** 2 + patch_grid_y ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_distances = np.log2(distances_from_patch_center)

    # log_amplitude_bins: 1 (center), 2 (ring1), 3 (ring2), 4 (ring3)
    # Points > radial_thresholds[-1] will get value len(radial_thresholds) + 1 = 5
    log_amplitude_bins = np.full_like(patch_grid_x, len(radial_thresholds) + 1, dtype=int)
    log_amplitude_bins[log_distances <= radial_thresholds[0]] = 1
    for i in range(len(radial_thresholds) - 1):  # Iterates 3 times for 4 thresholds
        log_amplitude_bins[
            (log_distances > radial_thresholds[i]) & (log_distances <= radial_thresholds[i + 1])
            ] = i + 2  # Assigns 2, 3, 4

    patch_angles_rad = np.arctan2(patch_grid_y, patch_grid_x)
    patch_angles_deg_0_360 = np.mod(np.degrees(patch_angles_rad), 360.0)

    log_angle_bins_float = patch_angles_deg_0_360 * num_descriptor_orientations / 360.0
    log_angle_bins = np.round(log_angle_bins_float).astype(int)

    log_angle_bins[log_angle_bins <= 0] += num_descriptor_orientations
    while np.any(log_angle_bins > num_descriptor_orientations):
        log_angle_bins[log_angle_bins > num_descriptor_orientations] -= num_descriptor_orientations
    log_angle_bins[log_angle_bins <= 0] += num_descriptor_orientations

    return log_amplitude_bins, log_angle_bins


def _build_histograms_for_patch(
        dominant_orientation_patch: np.ndarray,
        log_amplitude_bins_grid: np.ndarray,
        log_angle_bins_grid: np.ndarray,
        moment_patch_for_weighting: np.ndarray,  # <<--- 新增参数: 用于加权的moment patch
        patch_radius: int,
        num_dominant_orientation_types: int,
        num_descriptor_orientations: int
) -> tuple[np.ndarray, np.ndarray]:
    hist_rings = np.zeros(
        (num_dominant_orientation_types, num_descriptor_orientations, NUM_LOG_POLAR_RINGS_HIST_DIM),
        dtype=np.float64
    )
    hist_center = np.zeros(num_dominant_orientation_types, dtype=np.float64)

    patch_grid_rows, patch_grid_cols = log_angle_bins_grid.shape
    patch_center_coord_1idx = patch_radius + 1

    for r_patch_0idx in range(patch_grid_rows):
        r_patch_1idx = r_patch_0idx + 1
        for c_patch_0idx in range(patch_grid_cols):
            c_patch_1idx = c_patch_0idx + 1

            if (r_patch_1idx - patch_center_coord_1idx) ** 2 + \
                    (c_patch_1idx - patch_center_coord_1idx) ** 2 <= patch_radius ** 2:

                angle_bin_1idx = log_angle_bins_grid[r_patch_0idx, c_patch_0idx]
                amplitude_bin_1idx = log_amplitude_bins_grid[r_patch_0idx, c_patch_0idx]
                dom_orient_type_1idx = dominant_orientation_patch[r_patch_0idx, c_patch_0idx]

                # 获取当前像素的权重 (来自moment_map)
                pixel_weight = moment_patch_for_weighting[r_patch_0idx, c_patch_0idx]

                dom_orient_type_0idx = dom_orient_type_1idx - 1
                angle_bin_0idx = angle_bin_1idx - 1

                if amplitude_bin_1idx == 1:  # Center region
                    hist_center[dom_orient_type_0idx] += pixel_weight  # <<--- 修改为加权
                # amplitude_bin_1idx 2,3,4 correspond to the NUM_ACTUAL_LOG_POLAR_RINGS (3 rings)
                # These map to ring_idx_0idx = 0,1,2
                elif 2 <= amplitude_bin_1idx <= (NUM_ACTUAL_LOG_POLAR_RINGS + 1):  # amplitude_bin_1idx values 2,3,4
                    ring_idx_0idx = amplitude_bin_1idx - 2
                    # 确保ring_idx_0idx在 hist_rings 的维度范围内 (0 to NUM_LOG_POLAR_RINGS_HIST_DIM-1)
                    if 0 <= ring_idx_0idx < NUM_LOG_POLAR_RINGS_HIST_DIM:
                        hist_rings[dom_orient_type_0idx, angle_bin_0idx, ring_idx_0idx] += pixel_weight  # <<--- 修改为加权
    return hist_center, hist_rings


def _normalize_descriptor_vector(descriptor_vec: np.ndarray) -> np.ndarray:
    norm_val = scipy_norm(descriptor_vec)
    if norm_val > NUMERICAL_STABILITY_EPSILON:
        descriptor_vec /= norm_val
    descriptor_vec[descriptor_vec > DESCRIPTOR_NORMALIZATION_CLIP_VALUE] = DESCRIPTOR_NORMALIZATION_CLIP_VALUE
    norm_val = scipy_norm(descriptor_vec)
    if norm_val > NUMERICAL_STABILITY_EPSILON:
        descriptor_vec /= norm_val
    return descriptor_vec


def compute_pcmoh_descriptors(
        moment_map: np.ndarray,  # 这个 moment_map 将用于加权
        keypoint_locations: np.ndarray,
        phase_convolution_maps: np.ndarray,
        num_scales: int,
        num_orientations: int
) -> tuple[np.ndarray, np.ndarray]:
    if keypoint_locations.shape[0] == 0:
        return np.empty((0, 2), dtype=keypoint_locations.dtype), np.empty((0, 0, 0), dtype=np.float64)

    img_height, img_width = moment_map.shape

    convolution_sequence_sum = np.zeros((img_height, img_width, num_orientations), dtype=np.float64)
    for orient_idx in range(num_orientations):
        for scale_idx in range(num_scales):
            convolution_sequence_sum[:, :, orient_idx] += np.abs(phase_convolution_maps[scale_idx, orient_idx])

    dominant_orientation_map = np.argmax(convolution_sequence_sum, axis=2) + 1
    max_dominant_orientation_idx = np.max(dominant_orientation_map) if dominant_orientation_map.size > 0 else 1
    if max_dominant_orientation_idx == 0 and dominant_orientation_map.size > 0: max_dominant_orientation_idx = 1

    num_dominant_orientation_types = max_dominant_orientation_idx
    num_initial_keypoints = keypoint_locations.shape[0]
    keypoints_to_ignore_mask = np.zeros(num_initial_keypoints, dtype=bool)

    image_diagonal_proportion = np.maximum(
        np.sqrt(img_width ** 2 + img_height ** 2) / PATCH_RADIUS_DIAG_DIVISOR, 1.0)
    patch_radius_float = np.minimum(
        PATCH_RADIUS_DIAG_PROPORTION_FACTOR * image_diagonal_proportion,
        np.minimum(img_height, img_width) * PATCH_RADIUS_MIN_DIM_FRACTION)
    patch_radius = int(round(patch_radius_float))
    if patch_radius < 1: patch_radius = 1

    num_descriptor_orientations = 2 * num_dominant_orientation_types
    if num_descriptor_orientations == 0: num_descriptor_orientations = 1

    log_amplitude_bins_grid, log_angle_bins_grid = _precompute_log_polar_grid(
        patch_radius, num_descriptor_orientations)

    # descriptor_length 会因为 NUM_LOG_POLAR_RINGS_HIST_DIM 的改变而改变
    descriptor_length = num_dominant_orientation_types * \
                        (1 + num_descriptor_orientations * NUM_LOG_POLAR_RINGS_HIST_DIM)

    all_rift_descriptors = np.zeros(
        (num_descriptor_orientations, descriptor_length, num_initial_keypoints),
        dtype=np.float64)

    for kp_idx in range(num_initial_keypoints):
        kp_center_x_img_0idx = int(round(keypoint_locations[kp_idx, 0]))
        kp_center_y_img_0idx = int(round(keypoint_locations[kp_idx, 1]))

        kp_center_x_img_1idx = kp_center_x_img_0idx + 1
        kp_center_y_img_1idx = kp_center_y_img_0idx + 1

        patch_left_img_1idx = kp_center_x_img_1idx - patch_radius
        patch_right_img_1idx = kp_center_x_img_1idx + patch_radius
        patch_top_img_1idx = kp_center_y_img_1idx - patch_radius
        patch_bottom_img_1idx = kp_center_y_img_1idx + patch_radius

        if not (patch_left_img_1idx >= 1 and patch_right_img_1idx <= img_width and \
                patch_top_img_1idx >= 1 and patch_bottom_img_1idx <= img_height):
            keypoints_to_ignore_mask[kp_idx] = True
            continue

        dominant_orientation_patch = dominant_orientation_map[
                                     patch_top_img_1idx - 1: patch_bottom_img_1idx,
                                     patch_left_img_1idx - 1: patch_right_img_1idx
                                     ]
        # <<--- 新增：提取用于加权的moment_map patch ---
        moment_patch_for_hist_weighting = moment_map[
                                          patch_top_img_1idx - 1: patch_bottom_img_1idx,
                                          patch_left_img_1idx - 1: patch_right_img_1idx
                                          ]
        # -------------------------------------------------

        hist_center, hist_rings = _build_histograms_for_patch(
            dominant_orientation_patch,
            log_amplitude_bins_grid,
            log_angle_bins_grid,
            moment_patch_for_hist_weighting,  # <<--- 传递 moment_patch
            patch_radius,
            num_dominant_orientation_types,
            num_descriptor_orientations
        )

        rotated_descriptors_for_kp = np.zeros(
            (num_descriptor_orientations, descriptor_length), dtype=np.float64)

        current_hist_center = hist_center.copy()
        current_hist_rings = hist_rings.copy()

        base_descriptor_vec = np.concatenate(
            (current_hist_center.ravel(), current_hist_rings.ravel(order='F')))
        base_descriptor_vec = _normalize_descriptor_vector(base_descriptor_vec)
        rotated_descriptors_for_kp[0, :] = base_descriptor_vec

        for rot_idx in range(1, num_descriptor_orientations):
            current_hist_rings = np.roll(np.roll(current_hist_rings, shift=-1, axis=0), shift=1, axis=1)
            current_hist_center = np.roll(current_hist_center, shift=-1, axis=0)

            rotated_descriptor_vec = np.concatenate(
                (current_hist_center.ravel(), current_hist_rings.ravel(order='F')))
            rotated_descriptor_vec = _normalize_descriptor_vector(rotated_descriptor_vec)
            rotated_descriptors_for_kp[rot_idx, :] = rotated_descriptor_vec

        all_rift_descriptors[:, :, kp_idx] = rotated_descriptors_for_kp

    valid_keypoints_mask = ~keypoints_to_ignore_mask
    valid_keypoints = keypoint_locations[valid_keypoints_mask, :]
    rift_descriptors = all_rift_descriptors[:, :, valid_keypoints_mask]

    return valid_keypoints, rift_descriptors

