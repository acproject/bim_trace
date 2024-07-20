# 该函数用于生成一个3D旋转矩阵，根据给定的绕三个轴的旋转角度。函数首先根据给定的rx、ry和rz创建三个旋转矩阵Rx、Ry和Rz，然后将它们按顺序相乘得到最终的旋转矩阵R。

import numpy as np

def makeR3(rx, ry, rz):
    """
    Makes a 3d rotation matrix from 3 rotation angles.
    
    Parameters:
        rx, ry, rz : float
            Rotation angles in degrees around the three axes.
        
    Returns:
        R : ndarray of shape (3, 3)
            A 3x3 rotation matrix.
    """
    # Convert degrees to radians
    rx, ry, rz = np.radians([rx, ry, rz])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), np.sin(rx)],
                   [0, -np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, -np.sin(ry)],
                   [0, 1, 0],
                   [np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), np.sin(rz), 0],
                   [-np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    return R
