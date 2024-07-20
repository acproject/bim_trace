# `create_correspondences`的主要功能是为每个采样点找到一个假设，具体步骤如下：

# 1. **初始化变量**：创建空矩阵`A`, `B`, `correspondence_final`, 和 `test_final`用于存储计算结果。

# 2. **世界坐标到图像坐标的转换**：使用`world_to_pixel`函数将可见边界的3D世界坐标转换成2D图像坐标，并分别存储在`A_old`和`B_old`中。

# 3. **模型线的预处理**：
#   - 计算每条线段的长度`F`。
#   - 删除长度小于`minlinelength`的线段。
#   - 计算斜率`slopes`，分割数量`segments`，并存储减少后的可见边缘`visible_edges_reduced`。

# 4. **模型线的2D采样**：
#   - 对于每条保留的线段，计算其3D和2D采样点。
#   - 在采样点周围搜索一定距离内与该点垂直的像素点，以检测是否位于边缘图像上。

# 5. **对应关系的建立**：
#   - 如果找到匹配的边缘像素，则记录对应关系到`correspondence_final`中，包括像素位置、3D坐标和斜率。

# 6. **后处理**：
#   - 转换坐标系，从像素坐标到图像坐标。
#   - 使用`unique`函数去除重复的行，得到`correspondence_final_unique`。

# 该函数最终返回的是经过筛选和去重后的对应关系矩阵，用于后续的视觉处理或几何计算。


import numpy as np
import cv2

def create_correspondences(visible_edges, R, t, edge_image, f, min_line_length, search_len, seg_line_length, IRx, IRy, IPPM):
    A, B = [], []
    correspondence_final = []
    test_final = []

    A_old = world_to_pixel(f, visible_edges[:, 2:4], R, t, IRx, IRy, IPPM)
    B_old = world_to_pixel(f, visible_edges[:, 5:7], R, t, IRx, IRy, IPPM)

    k, h = 0, 0
    for n in range(len(visible_edges)):
        F = np.sqrt((A_old[n, 0] - B_old[n, 0])**2 + (A_old[n, 1] - B_old[n, 1])**2)
        centroid_visible_edges = (visible_edges[n, 2:4] + visible_edges[n, 5:7]) / 2
        cam_dist = np.sqrt((centroid_visible_edges[0] - t[0])**2 + (centroid_visible_edges[1] - t[1])**2 + (centroid_visible_edges[2] - t[2])**2)

        if F > min_line_length:
            A.append(A_old[n])
            B.append(B_old[n])
            slopes = (B[k][1] - A[k][1]) / (B[k][0] - A[k][0])
            segments = int(np.floor(F / seg_line_length)) + 1
            visible_edges_reduced = visible_edges[n, 2:7]

            n_points = segments if segments < 10 else 10

            samp_points_2d, samp_points_3d, matches = [], [], []
            for m in range(n_points + 1):
                samp_points_3d.append([
                    visible_edges_reduced[0] - (m - 1) * (visible_edges_reduced[0] - visible_edges_reduced[3]) / n_points,
                    visible_edges_reduced[1] - (m - 1) * (visible_edges_reduced[1] - visible_edges_reduced[4]) / n_points,
                    visible_edges_reduced[2] - (m - 1) * (visible_edges_reduced[2] - visible_edges_reduced[5]) / n_points
                ])
                samp_points_2d.append(world_to_pixel(f, samp_points_3d[m], R, t, IRx, IRy, IPPM))

                if (0 < samp_points_2d[m][0] < IRx) and (0 < samp_points_2d[m][1] < IRy):
                    search_end_points = []
                    for len in range(search_len):
                        x1 = samp_points_2d[m][0] + len / np.sqrt((-1/slopes)**2 + 1)
                        x2 = samp_points_2d[m][0] - len / np.sqrt((-1/slopes)**2 + 1)
                        y1 = (-1/slopes) * (x1 - samp_points_2d[m][0]) + samp_points_2d[m][1]
                        y2 = (-1/slopes) * (x2 - samp_points_2d[m][0]) + samp_points_2d[m][1]

                        if (0 < x1 < IRx) and (0 < y1 < IRy) and edge_image[int(y1), int(x1)]:
                            correspondence_final.append([int(x1), int(y1), samp_points_3d[m], slopes])
                            test_final.append([int(x1), int(y1), samp_points_2d[m], samp_points_3d[m]])
                            h += 1
                            break
                        if (0 < x2 < IRx) and (0 < y2 < IRy) and edge_image[int(y2), int(x2)]:
                            correspondence_final.append([int(x2), int(y2), samp_points_3d[m], slopes])
                            test_final.append([int(x2), int(y2), samp_points_2d[m], samp_points_3d[m]])
                            h += 1
                            break
            k += 1

    if not correspondence_final:
        correspondence_final_unique = []
        correspondence_final = []
        test_final = []
        return

    correspondence_final = np.array(correspondence_final)
    correspondence_final[:, :2] = ((correspondence_final[:, :2] - [IRx/2, IRy/2]) / IPPM)
    correspondence_final_unique = np.unique(correspondence_final[:, :6], axis=0)

# Example usage
# Assuming all necessary variables are defined
# create_correspondences(visible_edges, R, t, edge_image, f, min_line_length, search_len, seg_line_length, IRx, IRy, IPPM)
