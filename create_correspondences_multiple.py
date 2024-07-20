# 函数的主要功能是为每个采样点找到两个假设。它接收多种输入参数，包括可见边、旋转矩阵、平移向量、边缘图像、最小线段长度、搜索长度、线段长度、相机参数以及焦距。

# 该函数的处理流程如下：

# 1. **计算斜率、线段数量和坐标**：函数首先计算模型线上每个点在图像上的距离，并去除那些长度小于`MinSegmentLength`的小线段。对于保留的线段，它会计算斜率、线段数量，并进行2D和3D坐标的分割。

# 2. **采样模型线**：对保留的线段，函数会在2D空间中进行采样，并计算与采样点垂直的像素点，用于后续匹配。

# 3. **建立对应关系**：通过检查边缘图像中的像素值，函数尝试建立2D图像点与3D世界点之间的对应关系。

# 4. **结果处理**：最后，函数将这些对应关系转换成图像坐标系下的坐标，并使用`unique`函数去除重复项，返回唯一的一组对应关系。

# 此函数主要用于计算机视觉中的特征匹配任务，帮助在3D模型和2D图像之间建立连接。


import numpy as np
from scipy.ndimage import label

from world_to_pixel import world_to_pixel


def create_correspondences_multiple(visible_edges, R, t, edgeim, MinSegmentLength, SearchLength,
                                    SegmentLength, IRx, IRy, IPPM, f):
    A = []
    B = []
    correspondence_final = []
    test_final = []

    A_old = world_to_pixel(f, visible_edges[:, 1:4], R, t, IRx, IRy, IPPM)
    B_old = world_to_pixel(f, visible_edges[:, 4:7], R, t, IRx, IRy, IPPM)

    # CALCULATING SLOPE, NUMBER OF SEGMENTS, SEGMENTED 2D COORDINATES
    # SEGMENTED 3D COORDINATES and REMOVING SMALL LINES FROM THE MODEL
    k = 1
    h = 1
    for n in range(len(visible_edges)):
        # calculate distance of each model line on image
        F = np.sqrt((A_old[n, 0] - B_old[n, 0]) ** 2 + (A_old[n, 1] - B_old[n, 1]) ** 2)
        centroid_visible_edges = (visible_edges[n, 1:4] + visible_edges[n, 4:7]) / 2

        # deleting small model lines
        if F > MinSegmentLength:
            A.append(A_old[n])
            B.append(B_old[n])
            slopes = (B[-1][1] - A[-1][1]) / (B[-1][0] - A[-1][0])
            visible_edges_reduced = visible_edges[n, 1:7]
            segments = int(np.floor(F / SegmentLength)) + 1

            if segments < 10:
                n_points = segments
            else:
                n_points = 10

            # sampling model lines in 2d space
            samp_points_2d = []
            samp_points_3d = []
            matches = []

            for m in range(1, n_points + 1):
                samp_points_3d.append([
                    visible_edges_reduced[0] - (m - 1) * (
                                visible_edges_reduced[0] - visible_edges_reduced[3]) / n_points,
                    visible_edges_reduced[1] - (m - 1) * (
                                visible_edges_reduced[1] - visible_edges_reduced[4]) / n_points,
                    visible_edges_reduced[2] - (m - 1) * (
                                visible_edges_reduced[2] - visible_edges_reduced[5]) / n_points
                ])
                samp_points_2d.append(world_to_pixel(f, samp_points_3d[-1], R, t, IRx, IRy, IPPM))

                # calculting pixels perpendicular to sampled points
                for len in range(SearchLength):
                    x1 = samp_points_2d[-1][0] + len / np.sqrt((-1 / slopes) ** 2 + 1)
                    x2 = samp_points_2d[-1][0] - len / np.sqrt((-1 / slopes) ** 2 + 1)
                    y1 = (-1 / slopes) * (x1 - samp_points_2d[-1][0]) + samp_points_2d[-1][1]
                    y2 = (-1 / slopes) * (x2 - samp_points_2d[-1][0]) + samp_points_2d[-1][1]

                    search_end_points = np.array([[x1, y1, x2, y2]])

                    if 0 < search_end_points[0, 0] < IRx + 1 and 0 < search_end_points[0, 1] < IRy + 1:
                        if edgeim[int(search_end_points[0, 1]), int(search_end_points[0, 0])]:
                            correspondence_final.append(
                                [search_end_points[0, 0], search_end_points[0, 1], samp_points_3d[-1], k])
                            test_final.append([search_end_points[0, 0], search_end_points[0, 1], samp_points_2d[-1],
                                               samp_points_3d[-1]])
                            h += 1

                    if 0 < search_end_points[0, 2] < IRx + 1 and 0 < search_end_points[0, 3] < IRy + 1:
                        if edgeim[int(search_end_points[0, 3]), int(search_end_points[0, 2])]:
                            correspondence_final.append(
                                [search_end_points[0, 2], search_end_points[0, 3], samp_points_3d[-1], k])
                            test_final.append([search_end_points[0, 2], search_end_points[0, 3], samp_points_2d[-1],
                                               samp_points_3d[-1]])
                            h += 1
            k += 1

    if not correspondence_final:
        correspondence_final_unique = []
        correspondence_final = []
        test_final = []
        return

    # convert to image coordinate from pixel coordinate
    correspondence_final = np.array(correspondence_final)
    correspondence_final[:, 0] = (correspondence_final[:, 0] - IRx / 2) / IPPM
    correspondence_final[:, 1] = (IRy / 2 - correspondence_final[:, 1]) / IPPM
    correspondence_final_unique = np.unique(np.hstack((correspondence_final[:, :6], test_final[:, :4])), axis=0)

    return correspondence_final_unique
