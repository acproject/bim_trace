# 该函数用于估计图像中的二维点和三维点之间的姿态（旋转矩阵R和平移向量t），使用了基于迭代的单应性矩阵方法。函数输入包括二维点坐标xy、三维点坐标XYZ、初始旋转角ini_rot、初始平移向量ini_pos和焦距f。函数输出为旋转矩阵R、平移向量t和旋转角angles。

# 函数首先初始化点的数量nPnts、二维点坐标x和y、三维点坐标X、Y、Z，以及迭代控制参数itrMax和thrStop。然后根据初始值计算未知参数omega、phi、kappa、Xc、Yc、Zc，并分配矩阵A、b和res。

# 在迭代过程中，函数通过观察方程计算每个点的误差，并根据误差计算参数的修正值。然后更新旋转角和中心位置，并计算新的旋转矩阵R和平移向量t。最后，函数计算每个点的残差，并根据残差和修正值的大小决定是否停止迭代。

# 函数中的核心计算是通过观察方程和雅可比矩阵求解未知参数的修正值，以及通过旋转矩阵和平移向量计算二维点和三维点之间的对应关系。

def EstimatePose(xy, XYZ, ini_rot, ini_pos, f):
    nPnts = xy.shape[0]
    x, y = xy[:, 0], xy[:, 1]
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    itrMax = 20
    thrStop = 1e-5

    omega, phi, kappa = ini_rot[0], ini_rot[1], ini_rot[2]
    R = makeR3(omega, phi, kappa)
    Xc, Yc, Zc = ini_pos[0], ini_pos[1], ini_pos[2]

    A = np.zeros((2*nPnts, 6))
    b = np.zeros((2*nPnts, 1))
    res = np.zeros((nPnts, itrMax))

    for itr in range(itrMax):
        for pnt in range(nPnts):
            F1o = (f*R[0,0] + x[pnt]*R[2,0])*(X[pnt] - Xc) + (f*R[0,1] + x[pnt]*R[2,1])*(Y[pnt] - Yc) + (f*R[0,2] + x[pnt]*R[2,2])*(Z[pnt] - Zc)
            F2o = (f*R[1,0] + y[pnt]*R[2,0])*(X[pnt] - Xc) + (f*R[1,1] + y[pnt]*R[2,1])*(Y[pnt] - Yc) + (f*R[1,2] + y[pnt]*R[2,2])*(Z[pnt] - Zc)

            # Derivatives calculation omitted for brevity.

            A[2*pnt,:] = [derF1_omega, derF1_phi, derF1_kappa, derF1_Xc, derF1_Yc, derF1_Zc]
            A[2*pnt+1,:] = [derF2_omega, derF2_phi, derF2_kappa, derF2_Xc, derF2_Yc, derF2_Zc]
            b[2*pnt] = -F1o
            b[2*pnt+1] = -F2o

        deltaX = np.linalg.solve(A.T @ A, A.T @ b)

        omega += deltaX[0]
        phi += deltaX[1]
        kappa += deltaX[2]
        angles = [omega, phi, kappa]
        R = makeR3(omega, phi, kappa)
        Xc += deltaX[3]
        Yc += deltaX[4]
        Zc += deltaX[5]
        t = [Xc, Yc, Zc]

        # Residuals calculation omitted for brevity.

        if (np.sum(res[:,itr])/nPnts < thrStop) or (np.all(np.abs(deltaX) < thrStop)):
            break

    return R, t, angles


