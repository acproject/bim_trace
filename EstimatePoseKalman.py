# 该函数使用Kalman滤波方法来估计给定图像点和对应三维点集之间的姿态（旋转矩阵R和平移向量t）。具体来说，它通过最小化重投影误差来迭代更新初始旋转角和位置的估计值。函数的输入包括图像点、三维点集、初始旋转角和位置、焦距以及协方差矩阵。在每次迭代中，函数计算雅可比矩阵和残差，并使用这些值来更新估计值。迭代直到收敛于最小的重投影误差或达到最大迭代次数。输出包括最终的旋转矩阵R、平移向量t、旋转角的估计值以及雅可比矩阵的协方差矩阵。

def EstimatePoseKalman(xy, XYZ, ini_rot, ini_pos, f):
    var_XYZ = 50**2
    var_xy = (0.025)**2

    Q_xyXYZ = np.block([[var_xy*np.eye(2), np.zeros((2,3))],
                        [np.zeros((3,2)), var_XYZ*np.eye(3)]])

    nPnts = xy.shape[0]
    x = xy[:,0]
    y = xy[:,1]
    X = XYZ[:,0]
    Y = XYZ[:,1]
    Z = XYZ[:,2]

    itrMax = 20
    thrStop = 1e-5

    omega, phi, kappa = ini_rot
    R = makeR3(omega, phi, kappa)
    Xc, Yc, Zc = ini_pos

    A = np.zeros((2*nPnts, 6))
    b = np.zeros((2*nPnts, 1))
    res = np.zeros((nPnts, itrMax))
    J_b = np.zeros((2*nPnts, 5))

    for itr in range(itrMax):
        for pnt in range(nPnts):
            F1o = (f*R[0,0] + x[pnt]*R[2,0])*(X[pnt] - Xc) + (f*R[0,1] + x[pnt]*R[2,1])*(Y[pnt] - Yc) + (f*R[0,2] + x[pnt]*R[2,2])*(Z[pnt] - Zc)
            F2o = (f*R[1,0] + y[pnt]*R[2,0])*(X[pnt] - Xc) + (f*R[1,1] + y[pnt]*R[2,1])*(Y[pnt] - Yc) + (f*R[1,2] + y[pnt]*R[2,2])*(Z[pnt] - Zc)

            # Derivatives calculation...
            # ...

            A[2*pnt,:] = [derF1_omega, derF1_phi, derF1_kappa, derF1_Xc, derF1_Yc, derF1_Zc]
            A[2*pnt+1,:] = [derF2_omega, derF2_phi, derF2_kappa, derF2_Xc, derF2_Yc, derF2_Zc]

            b[2*pnt] = -F1o
            b[2*pnt+1] = -F2o

            # Jacobi_b calculation...
            # ...

        # Q_b calculation...
        # ...

        J_delX = np.linalg.solve(A.T@A, A.T)
        deltaX = J_delX@b

        # Q_delX calculation...
        # ...

        omega += deltaX[0]
        phi += deltaX[1]
        kappa += deltaX[2]
        R = makeR3(omega, phi, kappa)
        Xc += deltaX[3]
        Yc += deltaX[4]
        Zc += deltaX[5]

        # Update t and res...
        # ...

        if (np.sum(res[:,itr])/nPnts < thrStop) or np.all(np.abs(deltaX) < thrStop):
            res = res[:,:itr+1]
            break

    return R, np.array([Xc, Yc, Zc]), np.array([omega, phi, kappa]), Q_delX


