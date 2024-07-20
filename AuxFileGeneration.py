# 该代码使用Canny边缘检测器对一组图像进行边缘检测，并将结果保存到.mat文件中。具体步骤包括：

# 清除工作空间。
# 加载已有的可见边缘图像数据。
# 获取指定目录下所有.png图像文件的名称。
# 对于每个图像文件，将其读入并转换为灰度图像，然后使用Canny边缘检测器进行边缘检测，将结果保存到edgeim中。
# 将edgeim和可见边缘图像数据保存到.mat文件中。

import cv2
import os
import numpy as np
import glob
import scipy.io

# Clear workspace equivalent in Python is not needed as we are starting a new script

# Load existing visible edges data
visible_edges = scipy.io.loadmat('visible_edges.mat')

# Get all .png image file names in the specified directory
image_dir = r'C:\Users\acharyad\Desktop\Research\Video\cases\640x480-56D-RealData_new\sequence'
imageNames = glob.glob(os.path.join(image_dir, '*.png'))

# Initialize an empty list to store the edge images
edgeim = []

for imageName in imageNames:
    # Read and convert each image to grayscale
    im = cv2.imread(imageName)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    im_bin = cv2.Canny(im_gray, 0, 15, apertureSize=3)

    # Append the result to the list
    edgeim.append(im_bin)

# Convert the list to a numpy array
edgeim = np.array(edgeim)

# Save the edge images and visible edges data to a .mat file
scipy.io.savemat('aux_data_real.mat', {'edgeim': edgeim, 'visible_edges_all_frames': visible_edges['visible_edges_all_frames']})


