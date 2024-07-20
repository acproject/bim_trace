# 该函数用于计算三维旋转矩阵R关于角度Phi的导数。输入参数包括三个旋转角omega、phi和kappa，输出参数dR_Phi是一个3x3矩阵，包含了R关于phi的导数。该函数利用符号工具箱计算导数，并使用给定的旋转角进行数值评估。旋转矩阵R与makeR函数生成的矩阵一致。

import numpy as np

def derivativeR_Phi(omega, phi, kappa):
    """
    Calculates the derivatives of the 3D rotation matrix R with respect to Phi.

    Parameters:
        omega, phi, kappa : float
            Three rotation angles in radians.

    Returns:
        dR_Phi : numpy.ndarray
            A 3x3 matrix containing the derivatives of R with respect to phi.
    """
    dR_Phi = np.array([
        [-np.cos(kappa) * np.sin(phi),
         np.cos(kappa) * np.cos(phi) * np.sin(omega),
         -np.cos(kappa) * np.cos(omega) * np.cos(phi)],
        [np.sin(kappa) * np.sin(phi),
         -np.cos(phi) * np.sin(kappa) * np.sin(omega),
         np.cos(omega) * np.cos(phi) * np.sin(kappa)],
        [np.cos(phi),
         np.sin(omega) * np.sin(phi),
         -np.cos(omega) * np.sin(phi)]
    ])
    return dR_Phi


