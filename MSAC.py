# 该函数是一个基于MSAC算法的图像配准函数。函数输入包括当前帧和前一帧的边缘图像、可见边集合、kalman滤波器估计的角度和位移、以及一些参数。函数输出包括最终的位移、角度、迭代次数、残差等。

# 函数首先根据kalman滤波器的角度估计创建R_kalman矩阵。然后通过多次迭代更新对应关系，每次迭代分为两步：首先根据当前的R_kalman和位移估计创建多个对应关系；然后使用MSAC算法去除误差点，或者如果上一帧的对应关系可用，则利用它们来更新当前帧的对应关系。迭代直到收敛于一个局部最小值。

# 该函数的主要目的是通过迭代和去除误差点来估计两个连续帧之间的位移和旋转，从而实现图像配准。

import numpy as np

from MSAC_robust import MSAC_robust
from create_correspondences_multiple import create_correspondences_multiple
from makeR3 import makeR3
from temporal_reuse import temporal_reuse


def MASA(visible_edges, angles_kalman, t_kalman, frame, edgeim, edgeim_next,
                  correspondences_reuse, dis_best_set, Q_delX, MinSegmentLength,
                  SearchLength, SearchLengthMSAC, SegmentLength, IRx, IRy, IPPM, f,
                  AccurateMode, MaxMSACRuns, MSACSampleSize, JumpOutAngle,
                  JumpoutTranslation, UseMahalanobis, MahalanobisDistance,
                  PixelConvergenceThr, FastMode, ReusePixelThr, ReuseFactor,
                  CorrespondenceUpdates, ReuseData):

    R_kalman = makeR3(angles_kalman[0], angles_kalman[1], angles_kalman[2])
    t_reuse = []
    angles_reuse = []
    reuse = False

    for itr in range(CorrespondenceUpdates):  # NUMBER of correspondence updates

        correspondence_final = create_correspondences_multiple(visible_edges,
                                                               R_kalman, t_kalman, edgeim,
                                                               MinSegmentLength, SearchLength,
                                                               SegmentLength, IRx, IRy, IPPM, f)

        if ReuseData:
            if frame == 1:
                t_final, angles_final, R_final, stop_run, Q_delX, dis_best_set, correspondences_reuse = MSAC_robust(
                    visible_edges, angles_kalman, t_kalman, edgeim, correspondence_final,
                    Q_delX, MinSegmentLength, SearchLengthMSAC, SegmentLength, IRx, IRy, IPPM, f,
                    AccurateMode, MaxMSACRuns, MSACSampleSize, JumpOutAngle, JumpoutTranslation,
                    UseMahalanobis, MahalanobisDistance, PixelConvergenceThr, FastMode)
            else:
                dis_best_set, reuse, t_reuse, angles_reuse, Q_delX, correspondences_reuse = temporal_reuse(
                    correspondences_reuse, R_kalman, t_kalman, edgeim_next, dis_best_set,
                    angles_kalman, Q_delX, visible_edges, MinSegmentLength, SearchLength,
                    SegmentLength, IRx, IRy, IPPM, f, ReusePixelThr, ReuseFactor)
                if reuse:
                    t_final = t_reuse
                    angles_final = angles_reuse
                    print('reusing data')
                    break
                else:
                    t_final, angles_final, R_final, stop_run, Q_delX, dis_best_set, correspondences_reuse = MSAC_robust(
                        visible_edges, angles_kalman, t_kalman, edgeim, correspondence_final,
                        Q_delX, MinSegmentLength, SearchLengthMSAC, SegmentLength, IRx, IRy, IPPM, f,
                        AccurateMode, MaxMSACRuns, MSACSampleSize, JumpOutAngle, JumpoutTranslation,
                        UseMahalanobis, MahalanobisDistance, PixelConvergenceThr, FastMode)
        else:
            t_final, angles_final, R_final, stop_run, Q_delX, dis_best_set, correspondences_reuse = MSAC_robust(
                visible_edges, angles_kalman, t_kalman, edgeim, correspondence_final,
                Q_delX, MinSegmentLength, SearchLengthMSAC, SegmentLength, IRx, IRy, IPPM, f,
                AccurateMode, MaxMSACRuns, MSACSampleSize, JumpOutAngle, JumpoutTranslation,
                UseMahalanobis, MahalanobisDistance, PixelConvergenceThr, FastMode)

        if stop_run:
            break

        print('Updating correspondences')
        t_kalman = t_final
        R_kalman = R_final
        angles_kalman = angles_final


