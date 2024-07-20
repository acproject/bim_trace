# 该函数实现了基于MSAC算法的鲁棒性 pose估计。主要功能是通过迭代选择最佳的对应关系，消除外在因素导致的误差，最终得到准确的位姿估计。

# 输入参数包括可见边、角度、初始位姿、边缘图像、最终对应关系、协方差矩阵等。
# 函数内部进行MSAC算法迭代，每次从所有对应关系中随机选取一部分进行位姿估计。
# 通过判断位姿估计与真实值的差异，排除不符合条件的位姿估计。
# 使用马哈拉诺比斯距离等方法进一步去除异常值。
# 计算重投影误差，根据误差大小确定是否收敛。
# 最终返回最佳位姿、最佳对应关系等结果。  

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import pdist, cdist
from scipy.stats import chi2

from EstimatePose import EstimatePose
from EstimatePoseKalman import EstimatePoseKalman
from create_correspondences import create_correspondences
from world_to_image import world_to_image


def MSAC_robust(visible_edges, angles, t, edgeim, correspondence_final, Q_delX,
                MinSegmentLength, SearchLengthMSAC, SegmentLength, IRx, IRy, IPPM, f,
                AccurateMode, MaxMSACRuns, MSACSampleSize, JumpOutAngle, JumpoutTranslation,
                UseMahalanobis, MahalanobisDistance, PixelConvergenceThr, FastMode):

    np.random.seed(np.random.randint(10000))

    # MSAC for removing outliers
    n = 1
    actual_run = 0
    stop_run = False

    # Setting additional runs for higher accuracy
    accurate = 36 if AccurateMode else 2

    while actual_run <= MaxMSACRuns or n < accurate:
        actual_run += 1

        # Random permutation of the points
        p = np.random.permutation(len(correspondence_final[:, 0]))[:MSACSampleSize]
        xy = correspondence_final[p, :2]
        XYZ = correspondence_final[p, 2:]

        # Remove collinear lines
        if np.any(svd(xy)[1] < 0.5):
            continue

        # Estimate pose from 3 points
        R_ran, t_ran, angles_ran = EstimatePose(xy, XYZ, angles, t, f)

        # If empty vector
        if np.isnan(t_ran).any() or np.abs(angles_ran - angles).max() > JumpOutAngle * np.pi / 180 or \
           np.abs(t - t_ran).max() > JumpoutTranslation:
            continue

        # Mahalanobis outlier rejection based on covariance matrix
        if UseMahalanobis:
            _, flag = np.linalg.cholesky(Q_delX[3:6, 3:6])
            if flag == 0:
                D = cdist([t_ran], [t], metric='mahalanobis', VI=Q_delX[3:6, 3:6])
                if D > MahalanobisDistance:
                    continue

        # Create correspondences using new pose
        correspondence_final_new = create_correspondences(visible_edges, R_ran, t_ran, edgeim, f,
                                                          MinSegmentLength, SearchLengthMSAC,
                                                          SegmentLength, IRx, IRy, IPPM)

        # If bad pose ignore
        if correspondence_final_new is None:
            continue

        # Calculating residuals
        reproject_ransac = world_to_image(f, correspondence_final_new[:, 3:5], R_ran, t_ran)
        displacement = IPPM * (correspondence_final_new[:, :2] - reproject_ransac)
        disp_rms = np.sqrt(np.sum(displacement**2, axis=1))

        # MSAC
        pixel_thrushold_convergence = 1.96 * np.std(disp_rms)
        greater_than_thr = disp_rms > pixel_thrushold_convergence
        smaller_than_thr = disp_rms < pixel_thrushold_convergence
        a = len(disp_rms) - np.sum(greater_than_thr)
        disp_rms[greater_than_thr] = pixel_thrushold_convergence
        disp_rms = disp_rms**2
        total_dist = np.sum(disp_rms)

        # Check confidence
        if FastMode:
            ss = disp_rms < 1
            inlierProbability = (np.sum(ss) / len(correspondence_final_new[:, 0]))**MSACSampleSize
            N = np.ceil(np.log10(1 - 0.99) / np.log10(1 - inlierProbability)).astype(int)
            if actual_run > N and total_dist == np.min(total_dist):
                print('jumped out')
                break

        n += 1

    # MSAC index = select the best set
    max_index = np.argmin(total_dist)

    # Define convergence criteria
    s = disp_rms[max_index] < PixelConvergenceThr**2
    if np.sum(s) > 0.6 * len(correspondence_final_new):
        stop_run = True

    # Temporal reuse
    dis_best_set = np.sqrt(disp_rms[s])
    correspondences_reuse = correspondence_final_new[s, :]

    # Return best pose and the uncertainty after pose refinement
    xy = correspondence_final_new[smaller_than_thr, :2]
    XYZ = correspondence_final_new[smaller_than_thr, 2:]
    R_final, t_final, angles_final, Q_delX_best = EstimatePoseKalman(xy, XYZ, angles, t, f)

    return t_final, angles_final, R_final, stop_run, Q_delX_best, dis_best_set, correspondences_reuse
