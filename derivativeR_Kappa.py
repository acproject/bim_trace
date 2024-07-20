# 该函数用于计算三维旋转矩阵R关于Kappa的导数。输入参数为三个旋转角（omega、phi、kappa），输出参数dR_Kappa为一个3x3矩阵，包含了R关于Kappa的导数。该函数使用符号工具箱计算导数，并将导数在给定的旋转角（omega、phi、kappa）处进行求值。旋转矩阵R与makeR函数所生成的矩阵一致。

import numpy as np

def derivativeR_Kappa(omega, phi, kappa):
    """
    Calculates the derivatives of the 3D rotation matrix R with respect to Kappa.

    Parameters:
        omega (float): Rotation angle in radians.
        phi (float): Rotation angle in radians.
        kappa (float): Rotation angle in radians.

    Returns:
        dR_Kappa (numpy.ndarray): 3x3 matrix containing the derivatives of R with respect to Kappa.
    """
    dR_Kappa = np.array([
        [-np.cos(phi)*np.sin(kappa),   np.cos(kappa)*np.cos(omega) - np.sin(kappa)*np.sin(omega)*np.sin(phi),
         np.cos(kappa)*np.sin(omega) + np.cos(omega)*np.sin(kappa)*np.sin(phi)],
        [-np.cos(kappa)*np.cos(phi), - np.cos(omega)*np.sin(kappa) - np.cos(kappa)*np.sin(omega)*np.sin(phi),
         np.cos(kappa)*np.cos(omega)*np.sin(phi) - np.sin(kappa)*np.sin(omega)],
        [0,                                                        0,                                                      0]
    ])
    return dR_Kappa

