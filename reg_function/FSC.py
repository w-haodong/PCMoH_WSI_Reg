import math
import numpy as np

# 假设 LSM.py 的内容就是下面的 lsm 函数，所以我们直接把它放在这里
# 如果你的 LSM.py 有其他内容或你想保持分离，你需要确保 fsc 能正确导入 lsm
# from LSM import lsm # 如果 lsm 在单独的 LSM.py 文件中

"""
LSM函数是为了使用最小二乘法来计算两组点之间的仿射变换参数。
match1和match2：两组对应的点的坐标，每组点是一个Nx2的矩阵。
change_form：指定变换的类型，这里是（'affine'）'仿射'变换。
"""


def lsm(match1, match2, change_form):  # 保持你原始的lsm函数签名和实现
    # A的构建是正确的，对应参数顺序 p0,p1 (用于x'), p2,p3 (用于y'), p4 (tx), p5 (ty)
    A_matrix = np.zeros([2 * len(match1), 4])
    for i in range(len(match1)):
        A_matrix[2 * i:2 * i + 2] = np.tile(match1[i], (2, 2))
    B_transform = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    B_transform = np.tile(B_transform, (len(match1), 1))
    A_matrix = A_matrix * B_transform
    B_ones = np.array([[1, 0], [0, 1]])
    B_ones = np.tile(B_ones, (len(match1), 1))
    A_matrix = np.hstack((A_matrix, B_ones))

    b_vector = match2.reshape(1, int(len(match2) * len(match2[0]))).T

    parameters = np.zeros([8, 1])  # 初始化8个参数，仿射只用前6个
    rmse = float('inf')  # 初始化RMSE

    if change_form == "affine":
        if A_matrix.shape[0] < 6:  # 至少需要3对点 (6个方程) 来解6个参数
            # print("LSM Warning: Not enough points for affine LSM.")
            return np.squeeze(parameters), rmse  # 返回初始化的0参数和inf rmse

        try:
            # 使用最小二乘解 Ax = b
            # params_solved, residuals, rank, s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
            # params_solved 会是 (6,1)

            # 保持你原来的QR分解和solve方式
            Q, R = np.linalg.qr(A_matrix)
            if R.shape[0] < 6 or R.shape[1] < 6 or np.linalg.matrix_rank(R) < 6:
                # print("LSM Warning: R matrix is rank deficient after QR.")
                return np.squeeze(parameters), rmse  # 无法求解

            params_solved_for_qr = np.linalg.solve(R, np.dot(Q.T, b_vector))
            parameters[:6] = params_solved_for_qr  # 填充前6个

            N = len(match1)
            M_transform = np.array([[parameters[0, 0], parameters[1, 0]], [parameters[2, 0], parameters[3, 0]]])
            translations = np.array([parameters[4, 0], parameters[5, 0]])  # (2,)

            # match1_test_trans = M_transform.dot(match1.T) + np.tile(translations.reshape(2,1), (1, N))
            # 更NumPyic的写法：
            match1_test_trans = (match1 @ M_transform.T) + translations  # match1 (N,2), M_transform.T (2,2) -> (N,2)
            # translations (2,) -> broadcasting to (N,2)

            # match1_test_trans = match1_test_trans.T # 如果用 .dot(match1.T) 则需要转置回来

            test_diff = match1_test_trans - match2
            # rmse = math.sqrt(sum(sum(np.power(test_diff, 2))) / N) # 原来的 sum(sum())
            rmse = math.sqrt(np.sum(test_diff ** 2) / N)  # np.sum 会加总所有元素
        except np.linalg.LinAlgError:
            # print("LSM Error: Linear algebra error during solution.")
            # 保持返回初始化的 parameters 和 inf rmse
            pass  # parameters已经是0, rmse已经是inf

    # 其他 change_form 未实现，会返回初始化的 parameters (大部分0) 和 inf rmse
    return np.squeeze(parameters), rmse


def _check_collinearity_3pts(points_3x2: np.ndarray) -> bool:
    """检查3个2D点是否共线。True如果共线。"""
    if points_3x2.shape[0] != 3 or points_3x2.shape[1] != 2:
        return True  # 样本点数不对，视为退化
    # 使用面积法（行列式两倍）
    val = points_3x2[0, 0] * (points_3x2[1, 1] - points_3x2[2, 1]) + \
          points_3x2[1, 0] * (points_3x2[2, 1] - points_3x2[0, 1]) + \
          points_3x2[2, 0] * (points_3x2[0, 1] - points_3x2[1, 1])
    return np.abs(val) < 1e-9  # 用小容差判断是否接近零


def fsc(cor1: np.ndarray, cor2: np.ndarray, change_form: str, error_t: float):
    M, N_dims = np.shape(cor1)  # M = 点数, N_dims = 维度 (应为2)

    if M != cor2.shape[0] or N_dims != cor2.shape[1]:
        raise ValueError("cor1 和 cor2 的形状必须匹配。")
    if N_dims != 2:
        raise ValueError("此fsc实现目前仅支持2D点。")

    n_min_samples = 0
    if change_form == 'similarity':
        n_min_samples = 2
    elif change_form == 'affine':
        n_min_samples = 3
    elif change_form == 'perspective':
        n_min_samples = 4
    else:
        raise ValueError(f"不支持的变换类型: {change_form}")

    if M < n_min_samples:
        print(f"FSC Warning: 点数 ({M}) 少于估计 {change_form} 所需的最小点数 ({n_min_samples})。")
        return np.eye(3)  # 返回单位矩阵作为无效结果

    # 迭代次数上限
    # 原始代码中 max_iteration 的计算可能非常大
    # iterations = min(10000, int(combinatorial_max_iterations))
    # 我们直接使用一个固定的合理上限
    iterations = 10000
    if M <= n_min_samples * 2:  # 如果总点数很少，适当减少迭代
        iterations = min(iterations, M * (M - 1) if M > 1 else 1)  # 粗略减少

    best_consensus_count = -1  # 使用-1确保任何有效模型都能更新
    best_inlier_indices = np.array([], dtype=int)

    max_sampling_retries = 50  # 单次迭代中，为找到非退化样本的最大尝试次数

    for _ in range(iterations):
        sample_indices = np.array([], dtype=int)
        is_sample_degenerate = True
        retries = 0
        while is_sample_degenerate and retries < max_sampling_retries:
            sample_indices = np.random.choice(M, n_min_samples, replace=False)
            cor1_sample = cor1[sample_indices]
            # cor2_sample = cor2[sample_indices] # lsm会用到

            if change_form == 'affine':  # n_min_samples is 3
                if not _check_collinearity_3pts(cor1_sample) and \
                        not _check_collinearity_3pts(cor2[sample_indices]):  # 检查两边样本
                    is_sample_degenerate = False
            elif change_form == 'similarity':  # n_min_samples is 2
                if not np.allclose(cor1_sample[0], cor1_sample[1]):  # 确保两点不重合
                    is_sample_degenerate = False
            # TODO: Add degeneracy checks for 'perspective' (e.g., any 3 points collinear)
            else:  # For other types, assume non-degenerate for now if points are distinct
                is_sample_degenerate = False
            retries += 1

        if is_sample_degenerate:
            continue  # 无法找到好的样本，跳过此次迭代

        cor2_sample = cor2[sample_indices]

        # parameters 是 (8,) array, rmse 是 float
        parameters, _ = lsm(cor1_sample, cor2_sample, change_form)

        if parameters is None or len(parameters) < 6:  # lsm可能返回None或参数不足
            continue

        # 构建3x3变换矩阵 (基于fsc中原始代码对parameters的用法)
        # parameters[0]=a, parameters[1]=b, parameters[2]=c, parameters[3]=d
        # parameters[4]=tx, parameters[5]=ty
        # parameters[6]=0, parameters[7]=0 (来自lsm的初始化)
        current_solution_matrix = np.array([
            [parameters[0], parameters[1], parameters[4]],
            [parameters[2], parameters[3], parameters[5]],
            [parameters[6], parameters[7], 1.0]  # 确保H33是1.0
        ])
        # 对于严格仿射，最后一行应为 [0,0,1]
        if change_form == 'affine':
            current_solution_matrix[2, 0] = 0.0
            current_solution_matrix[2, 1] = 0.0

        # 验证模型，计算内点
        # cor1 (M,2) -> cor1_h (M,3) by adding 1s
        cor1_h = np.hstack((cor1, np.ones((M, 1))))
        # transformed_cor1_h (M,3) = cor1_h (M,3) @ current_solution_matrix.T (3,3)
        transformed_cor1_h = cor1_h @ current_solution_matrix.T

        # 透视除法
        w_transformed = transformed_cor1_h[:, 2]
        # 防止除以非常小的值或零
        w_transformed[np.abs(w_transformed) < 1e-9] = 1e-9

        transformed_cor1_cartesian = transformed_cor1_h[:, :2] / w_transformed[:, np.newaxis]

        errors_sq = np.sum((transformed_cor1_cartesian - cor2) ** 2, axis=1)  # 平方距离
        current_inlier_indices = np.where(errors_sq < error_t ** 2)[0]  # 与平方阈值比较
        current_consensus_count = len(current_inlier_indices)

        if current_consensus_count > best_consensus_count:
            best_consensus_count = current_consensus_count
            best_inlier_indices = current_inlier_indices

    if best_consensus_count < n_min_samples:
        print(f"FSC Warning: 最终最佳内点数 ({best_consensus_count}) 过少。")
        return np.eye(3)  # 返回单位矩阵

    cor1_best_inliers = cor1[best_inlier_indices]
    cor2_best_inliers = cor2[best_inlier_indices]

    # 精炼模型前对最佳内点集进行去重
    final_cor1_for_lsm = cor1_best_inliers
    final_cor2_for_lsm = cor2_best_inliers

    if cor1_best_inliers.shape[0] > 0:
        # 先基于cor1去重
        _, unique_idx_cor1 = np.unique(final_cor1_for_lsm, axis=0, return_index=True)
        final_cor1_for_lsm = final_cor1_for_lsm[unique_idx_cor1]
        final_cor2_for_lsm = final_cor2_for_lsm[unique_idx_cor1]

        # 再基于（已根据cor1去重后的）cor2去重
        if final_cor2_for_lsm.shape[0] > 0:
            _, unique_idx_cor2 = np.unique(final_cor2_for_lsm, axis=0, return_index=True)
            final_cor1_for_lsm = final_cor1_for_lsm[unique_idx_cor2]
            final_cor2_for_lsm = final_cor2_for_lsm[unique_idx_cor2]

    if final_cor1_for_lsm.shape[0] < n_min_samples:
        print(f"FSC Warning: 去重后内点数 ({final_cor1_for_lsm.shape[0]}) 不足以精炼模型。")
        # 可以选择返回用RANSAC迭代中找到的最佳模型（如果保存了的话），或直接返回None/单位阵
        # 这里我们还是尝试用这些点去拟合，如果lsm能处理
        if best_consensus_count >= n_min_samples:  # 如果RANSAC迭代时找到了足够点
            # fallback to use non-unique inliers if unique set is too small but original was ok
            parameters_final, _ = lsm(cor1_best_inliers, cor2_best_inliers, change_form)
        else:
            return np.eye(3)
    else:
        parameters_final, _ = lsm(final_cor1_for_lsm, final_cor2_for_lsm, change_form)

    if parameters_final is None or len(parameters_final) < 6:
        return np.eye(3)

        # 构建最终变换矩阵
    solution_final = np.eye(3)
    if change_form == 'affine':
        # 使用与RANSAC迭代中相同的矩阵构建逻辑
        solution_final = np.array([
            [parameters_final[0], parameters_final[1], parameters_final[4]],
            [parameters_final[2], parameters_final[3], parameters_final[5]],
            [0.0, 0.0, 1.0]  # 确保仿射的最后一行
        ])
    # TODO: 为其他变换类型构建最终矩阵

    return solution_final