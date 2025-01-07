from itertools import accumulate
from typing import Tuple

import torch
import triton
import triton.language as tl

from triton import Config
from triton.language import dtype


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Triton内核函数，用于执行激活量化操作。

    参数:
    - x_ptr: 输入张量的指针。
    - y_ptr: 输出张量的指针。
    - s_ptr: 量化尺度的输出张针。
    - BLOCK_SIZE: 每个程序块处理的元素数，作为常量表达式。

    此函数将输入张量x_ptr指向的数据进行量化操作，并将结果存储在y_ptr指向的输出张量中。
    同时，每个程序块处理的量化尺度被存储在s_ptr指向的数组中。
    """
    # 获取当前程序块的ID
    pid = tl.program_id(axis=0)
    # 计算当前程序块的元素偏移量
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 从输入张量中加载数据，并转换为float32类型
    x = tl.load(x_ptr + offs).to(tl.float32)
    # 计算量化尺度
    s = tl.max(tl.abs(x)) / 448.
    # 执行量化操作
    y = x / s
    # 将量化后的数据转换为输出张量的元素类型，并存储到输出张量中
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    # 存储量化尺度
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对输入的张量x进行量化操作，并返回量化的张量y和量化尺度s。

    参数:
    x - 输入的张量，类型为torch.Tensor，且在调用前必须是连续存储的。
    block_size - 量化操作的块大小，默认值为128。输入张量的最后一个维度必须能被块大小整除。

    返回:
    一个元组，包含两个元素：
    - y: 量化的输出张量，具有与输入张量x相同的形状，但数据类型为float8_e4m3fn。
    - s: 量化尺度张量，数据类型为float32，形状为除了输入张量x的最后一个维度外，其他维度相同，最后一个维度的大小为x最后一个维度大小除以块大小。

    注释:
    - 该函数主要用于对激活函数的输出进行量化，以减少模型的存储和计算需求。
    - 量化操作是按块进行的，每个块内的数据使用相同的量化尺度。
    - 使用了triton库的cdiv函数来计算grid的大小，这通常用于CUDA编程中根据数据量和块大小来决定CUDA grid的维度。
    - act_quant_kernel是实际进行量化操作的内核函数，它根据输入张量x，生成量化的输出张量y和量化尺度s。
    """
    # 确保输入张量x是连续存储的，这对于后续的量化操作是必要的。
    assert x.is_contiguous()
    # 确保输入张量x的最后一个维度可以被块大小整除，这是进行量化操作的前提。
    assert x.size(-1) % block_size == 0
    # 创建一个与输入张量x形状相同但数据类型为float8_e4m3fn的空张量y，用于存储量化后的输出。
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # 创建一个用于存储量化尺度的空张量s，其形状除了最后一个维度外，其他维度与输入张量x相同。
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    # 定义grid函数，用于根据数据量和块大小决定CUDA grid的维度。
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BlOCK_SIZE']), )
    # 调用实际进行量化操作的内核函数act_quant_kernel，根据输入张量x生成量化的输出张量y和量化尺度s。
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    # 返回量化的输出张量y和量化尺度s。
    return y, s

@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    使用Triton JIT编译的权重去量化内核函数。

    参数:
    - x_ptr: 量化权重的指针
    - s_ptr: 尺度因子的指针
    - y_ptr: 去量化权重的输出指针
    - M: 矩阵的行数
    - N: 矩阵的列数
    - BLOCK_SIZE: 块大小，用于划分矩阵块
    """
    # 获取当前程序的行索引
    pid_m = tl.program_id(axis=0)
    # 获取当前程序的列索引
    pid_n = tl.program_id(axis=1)
    # 计算列方向上的块数量
    n = tl.cdiv(N, BLOCK_SIZE)
    # 计算当前块在矩阵中的行偏移量
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 计算当前块在矩阵中的列偏移量
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 计算当前块中每个元素在矩阵中的绝对位置
    offs = offs_m[:, None] * N + offs_n[None, :]
    # 生成一个掩码，用于确保不越界访问矩阵元素
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # 加载量化权重，并转换为float32类型
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # 加载对应的尺度因子
    s = tl.load(s_ptr + pid_m * n + pid_n)
    # 执行去量化操作，即量化权重乘以尺度因子
    y = x * s
    # 将去量化后的权重存储到输出矩阵中
    tl.store(y_ptr + offs, y, mask=mask)

def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    对权重进行去量化操作。

    该函数接受两个Tensor参数x和s，其中x是量化后的权重，s是对应的尺度因子。
    通过调用CUDA扩展函数weight_dequant_kernel来执行去量化操作，将量化后的数据转换回浮点数格式。

    参数:
    x (torch.Tensor): 量化后的权重Tensor，形状为(M, N)。
    s (torch.Tensor): 尺度因子Tensor，用于将量化后的数据转换回原始范围，形状为(M, N)。
    block_size (int): CUDA网格划分的块大小，默认值为128。

    返回:
    torch.Tensor: 去量化后的权重Tensor，具有与x相同的形状。

    注:
    该函数要求输入的x和s都是连续的二维Tensor，并且需要在支持triton操作的环境中运行。
    """
    # 确保输入Tensor是连续的，这是为了保证数据在内存中是连续存放的，避免在GPU计算时出现错乱。
    assert x.is_contiguous() and s.is_contiguous()
    # 确保输入Tensor是二维的，因为该函数设计用于矩阵操作。
    assert x.dim() == 2 and s.dim() == 2

    # 获取输入Tensor的维度，用于后续的网格划分。
    M, N = x.size()

    # 创建一个与输入x形状相同但类型为默认浮点类型的空Tensor，用于存放去量化后的结果。
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    # 定义网格划分函数，根据M和N的大小以及块大小来确定CUDA网格的维度。
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))

    # 调用CUDA扩展函数weight_dequant_kernel来执行去量化操作，具体计算在GPU上并行执行。
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)

    # 返回去量化后的结果Tensor。
    return y


# 为FP8 GEMM（矩阵乘法）操作配置优化参数
# 通过枚举不同的块尺寸和阶段数来寻找最优配置
# BLOCK_SIZE_M、BLOCK_SIZE_N、BLOCK_SIZE_K分别代表矩阵乘法中M、N、K维度的块大小
# num_stages代表流水线阶段数，num_warps代表warp的数量，这些参数对GPU性能有重要影响
fp8_gemm_config = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3,4, 5, 6]
]


# 使用Triton的自动调优和Just-In-Time(JIT)编译功能装饰矩阵乘法内核函数
# 该函数针对FP8精度的矩阵乘法进行了优化
@triton.autotune(configs=fp8_gemm_config, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N:tl.constexpr, K:tl.constexpr,
                    BLOCK_SIZE_M:tl.constexpr,
                    BLOCK_SIZE_N:tl.constexpr,
                    BLOCK_SIZE_K:tl.constexpr):
    """
    FP8精度矩阵乘法的Triton内核函数

    参数:
    - a_ptr: A矩阵的指针
    - b_ptr: B矩阵的指针
    - c_ptr: C矩阵的指针，结果存储在这里
    - a_s_ptr: A矩阵的比例因子指针
    - b_s_ptr: B矩阵的比例因子指针
    - M, N, K: 矩阵的维度
    - BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: 矩阵乘法的块大小，用于并行化和优化
    """
    # 获取当前程序的ID，用于计算矩阵中的位置
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 计算K维度上的块数量
    k = tl.cdiv(K, BLOCK_SIZE_K)

    # 计算矩阵中当前块的索引
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 计算A、B矩阵及其比例因子的指针
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    # 初始化累积矩阵，用于存储中间结果
    accumulate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 主循环，遍历K维度上的每个块
    for i in range(k):
        # 加载当前块的A、B矩阵数据及其比例因子
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)

        # 累积当前块的矩阵乘法结果
        accumulate += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        # 更新指针到下一个块
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1

    # 将累积结果转换为C矩阵的数据类型，并计算C矩阵的索引
    c = accumulate(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c_ptr + offs_m[:, None] * N + offs_n[None, :]

    # 创建一个掩码以确保结果矩阵的维度正确
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 将最终结果存储到C矩阵中
    tl.store(c_ptr, c, mask=mask)

def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    执行两个 FP8 格式张量的通用矩阵乘法（GEMM）。

    参数:
    a (torch.Tensor): 第一个输入张量，FP8 格式。
    a_s (torch.Tensor): 第一个输入张量的缩放因子张量。
    b (torch.Tensor): 第二个输入张量，FP8 格式。
    b_s (torch.Tensor): 第二个输入张量的缩放因子张量。

    返回:
    torch.Tensor: 矩阵乘法的结果张量。
    """
    # 确保所有输入张量在内存中是连续的，以避免计算过程中出现数据损坏
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    # K 是内层维度的大小，M 是第一个矩阵的行数，N 是第二个矩阵的列数
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    # 初始化输出张量 c，默认数据类型
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    # 定义 GEMM 内核启动的网格函数，根据 M 和 N 的大小确定所需的块数
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    # 启动 GEMM 内核执行矩阵乘法
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)

    # 返回结果张量 c
    return c
