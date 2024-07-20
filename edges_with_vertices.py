# 代码的主要功能是从文本文件中读取可见顶点和整个网格的边，然后根据每帧的顶点ID匹配出所有可见边，并将结果保存到.mat文件中。

# 使用dlmread函数从指定文件路径读取可见顶点和整个网格的边数据。
# 使用for循环对每帧的可见顶点进行处理，将每帧的顶点坐标保存到一个cell数组中。
# 再次使用for循环遍历所有边，通过ismember函数判断边的两个顶点是否在当前帧的可见顶点列表中。如果两个顶点都可见，则将该边的ID和两个顶点的坐标保存到另一个cell数组中。
# 最后，使用save函数将保存所有可见边的cell数组保存到名为visible_edges.mat的.mat文件中。

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

# 读取文本文件中的数据
visible_vertices = np.loadtxt('FILE,LOCATION,visible_vertices_all_frames.txt')
all_edges = np.loadtxt('FILE,LOCATION,visible_edges_whole_mesh.txt')

# 创建一个字典来存储每帧的可见顶点
visible_vertices_framewise = {}

# 检查每帧的顶点
for k in range(1, int(np.max(visible_vertices[:,5]))+1):
    aa = np.where(visible_vertices[:,5] == k)[0]
    visible_vertices_framewise[k] = visible_vertices[aa, :4]
    print(f'Processing frame {k}') # 显示进度

# 创建一个字典来存储每帧的可见边
visible_edges_all_frames = {}

# 遍历每帧的可见顶点
for m, visible_vertices_frame in visible_vertices_framewise.items():
    print(f'Processing frame {m}') # 显示进度
    visible_edges_frame = []

    for i in range(len(all_edges)):
        a = all_edges[i, 1] in visible_vertices_frame[:, 0]
        b = all_edges[i, 2] in visible_vertices_frame[:, 0]

        if a and b:
            idx1 = np.where(visible_vertices_frame[:, 0] == all_edges[i, 1])[0][0]
            idx2 = np.where(visible_vertices_frame[:, 0] == all_edges[i, 2])[0][0]

            edge_data = np.hstack((all_edges[i, 0], visible_vertices_frame[idx1, 1:], visible_vertices_frame[idx2, 1:]))
            visible_edges_frame.append(edge_data)

    visible_edges_all_frames[m] = np.array(visible_edges_frame)

# 将结果保存到.mat文件中
savemat('visible_edges.mat', {'visible_edges_all_frames': visible_edges_all_frames})


