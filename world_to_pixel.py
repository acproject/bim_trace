# 该函数将世界坐标系中的点转换为图像坐标系中的点。输入参数包括焦距f、世界坐标系中的点worldPoints、旋转向量R、平移向量t、图像分辨率IRx和IRy、图像每像素的毫米数IPPM。函数首先根据旋转向量和平移向量将世界坐标系中的点转换为相机坐标系中的点，然后根据焦距将相机坐标系中的点转换为图像坐标系中的点。最后，函数返回图像坐标系中的点Points。
import numpy as np
def world_to_pixel(f, worldPoints, R, t, IRx, IRy, IPPM):
    # input a numpy array to transform a vectory
    worldPoints = worldPoints.reshape(-1, 3)

    # compute the center of camera coordinates
    denominator = (R[2, 0] * (worldPoints[:, 0] - t[0]) +
                   R[2, 1] * (worldPoints[:, 1] - t[1]) +
                   R[2, 2] * (worldPoints[:, 2] - t[2]))
    AAA = -f * ((R[0, 0] * (worldPoints[:, 0] - t[0]) +
                 R[0, 1] * (worldPoints[:, 1] - t[1]) +
                 R[0, 2] * (worldPoints[:, 2] - t[2])) / denominator)
    BBB = -f * ((R[1, 0] * (worldPoints[:, 0] - t[0]) +
                 R[1, 1] * (worldPoints[:, 1] - t[1]) +
                 R[1, 2] * (worldPoints[:, 2] - t[2])) / denominator)

    # compute point of image coordinates
    Points = np.column_stack((IRx / 2 + IPPM * AAA, IRy / 2 - IPPM * BBB))

    return Points
