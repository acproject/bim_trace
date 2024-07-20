# 该函数用于计算三维旋转矩阵R关于角速度向量Omega的导数。输入参数omega、phi、kappa为三个旋转角（单位为弧度），输出参数dR_Omega为一个3x3矩阵，包含了R关于omega的导数。该函数利用符号工具箱计算导数，并使用输入的旋转角进行数值评估。旋转矩阵R与makeR函数生成的矩阵一致。

import numpy as np

def derivativeR_Omega(omega, phi, kappa):
    """
    Calculates the derivatives of the 3D rotation matrix R with respect to Omega.

    Parameters:
        omega (float): Rotation angle around the first axis in radians.
        phi (float): Rotation angle around the second axis in radians.
        kappa (float): Rotation angle around the third axis in radians.

    Returns:
        numpy.ndarray: A 3x3 matrix containing the derivatives of R with respect to omega.
    """
    dR_Omega = np.array([
        [0,
         np.cos(kappa) * np.cos(omega) * np.sin(phi) - np.sin(kappa) * np.sin(omega),
         np.cos(omega) * np.sin(kappa) + np.cos(kappa) * np.sin(omega) * np.sin(phi)],
        [0,
         -np.cos(kappa) * np.sin(omega) - np.cos(omega) * np.sin(kappa) * np.sin(phi),
         np.cos(kappa) * np.cos(omega) - np.sin(kappa) * np.sin(omega) * np.sin(phi)],
        [0,
         -np.cos(omega) * np.cos(phi),
         -np.cos(phi) * np.sin(omega)]
    ])
    return dR_Omega

