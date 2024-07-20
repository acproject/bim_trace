# 该函数将世界坐标系中的点转换为图像坐标系中的点。函数输入包括焦距f、世界坐标系中的点worldPoints、旋转向量R和平移向量t。函数输出为图像坐标系中的点Points。

# 具体实现过程为：

# 根据旋转向量R和平移向量t，将世界坐标系中的点转换为相机坐标系中的点。
# 利用相机坐标系中的点和焦距f，计算图像坐标系中的点。
# 其中，A和B分别表示图像坐标系中的x和y坐标。

import numpy as np

def world_to_image(f, worldPoints, R, t):
    worldPoints = np.array(worldPoints)
    R = np.array(R)
    t = np.array(t)


    cameraPoints = np.dot(R, (worldPoints - t).T).T

    A = -f * (cameraPoints[:, 0] / cameraPoints[:, 2])
    B = -f * (cameraPoints[:, 1] / cameraPoints[:, 2])

    Points = np.column_stack((A, B))
    return Points
