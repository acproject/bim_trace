import math
from dataclasses import dataclass

from typing import Tuple, Optional, Literal

import torch
from torch import nn, masked
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    # 定义最大批次大小
    max_batch_size: int = 8
    # 定义最大序列长度
    max_seq_len: int = 4096 * 4
    # 定义数据类型，使用字面量类型限制值的范围
    dtype: Literal["bf16", "fp8"] = "bf16"
    # 定义词汇表大小
    vocab_size: int = 102400
    # 定义模型维度
    dim: int = 2048
    # 定义内部隐藏层维度
    inter_fim: int = 10944
    # 定义混合专家网络的内部维度
    moe_inter_dim: int = 1408
    # 定义层数
    n_layers: int = 27
    # 定义密集层的数量
    n_dense_layers: int = 1
    # 定义注意力头的数量
    n_heads: int = 16

    # 混合专家网络配置
    # 定义路由到的专家数量
    n_routed_experts: int = 64
    # 定义共享的专家数量
    n_shared_experts: int = 2
    # 定义激活的专家数量
    n_activated_experts: int = 6
    # 定义专家组的数量
    n_expert_groups: int = 1
    # 定义限制的专家组数量
    n_limited_groups: int = 1
    # 定义路由分数的计算函数
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    # 定义路由分数的缩放因子
    route_scale: float = 1.

    # 多线性注意力配置
    # 定义查询的低秩分解秩
    q_lora_rank: int = 0
    # 定义键值的低秩分解秩
    kv_lora_rank: int = 512
    # 定义查询键的非位置编码头维度
    qk_nope_head_dim: int = 128
    # 定义查询键的位置编码旋转头维度
    qk_rope_head_dim: int = 64
    # 定义值的头维度
    v_head_dim: int = 128

    # Yarn配置
    # 定义原始序列长度
    original_seq_len: int = 4096
    # 定义旋转位置编码的theta值
    rope_theta: float = 10000.0
    # 定义旋转位置编码的缩放因子
    rope_factor: float = 40
    # 定义快速beta值
    beta_fast: int = 32
    # 定义慢速beta值
    beta_slow: int = 1
    # 定义mscale因子
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """
   初始化函数，设置词汇表大小和维度。

   参数:
   vocab_size (int): 词汇表的大小。
   dim (int): 向量的维度。
   """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现前向传播过程。

        当世界大小大于1时，表示使用了并行处理，此时需要对输入进行特定处理。
        通过创建一个掩码来标识出不在词汇表范围内的元素，并将这些元素的值设置为0。
        这样做是为了确保并行处理时，不同设备间不会互相干扰。

        参数:
        x: 输入的张量，代表词汇表中的索引。

        返回:
        y: 经过嵌入层处理后的张量。
        """
        # 检查是否需要并行处理
        if world_size > 1:
            # 创建掩码，标识出不在词汇表范围内的元素
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 将词汇表索引调整到从0开始
            x = x - self.vocab_start_idx
            # 将掩码对应的元素设置为0
            x[mask] = 0
        # 使用嵌入层处理调整后的索引
        y = F.embedding(x, self.weight)
        # 再次检查是否需要并行处理
        if world_size > 1:
            # 将掩码对应的嵌入向量设置为0
            y[mask] = 0
            # 对嵌入向量进行全局规约操作，以完成并行处理
            dist.all_reduce(y)
        # 返回处理后的嵌入向量
        return y

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    对输入张量 x 应用线性变换: y = xA^T + b。

    该函数使用指定的 weight 对输入张量 x 进行线性操作，并可选地添加 bias。
    函数支持处理不同类型的数据和量化级别，根据需要调整计算方法以优化性能和资源使用。

    参数:
    - x: torch.Tensor, 输入特征张量。
    - weight: torch.Tensor, 线性层的权重参数。
    - bias: Optional[torch.Tensor], 线性层的可选偏置参数。默认为 None。

    返回:
    - torch.Tensor, 线性变换的结果。
    """
    # 如果权重的元素大小大于1，表示没有量化或特殊处理需要，直接使用 F.linear。
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    # 如果 gemm 实现为 "bf16"，表示使用 bfloat16 精度，先对权重进行反量化，然后进行线性计算。
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    # 对于其他情况，先对输入 x 进行量化，然后执行矩阵乘法，以优化资源使用和计算效率。
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        # 如果提供了偏置，将其加到输出中。
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        """
        初始化线性变换层。

        参数:
        in_features (int): 输入特征的数量。
        out_features (int): 输出特征的数量。
        bias (bool, 可选): 是否使用偏置项。默认为False。
        dtype (torch.dtype, 可选): 参数的数据类型。如果未提供，则使用默认数据类型。

        该方法会初始化线性变换层的权重和偏置（如果使用偏置的话），并根据数据类型设置相应的参数。
        """
        super().__init__()  # 初始化父类
        self.in_features = in_features  # 设置输入特征数量
        self.out_features = out_features  # 设置输出特征数量
        # 初始化权重参数，使用指定的数据类型
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        # 如果权重参数的元素大小为1，则进行量化处理
        if self.weight.element_size() == 1:
            # 计算量化后的输出特征数量
            scale_out_features = (out_features + block_size - 1) // block_size
            # 计算量化后的输入特征数量
            scale_in_features = (in_features + block_size -1) // block_size
            # 初始化量化后的权重参数
            self.weight.scale = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        else:
            # 如果不进行量化处理，则不注册scale参数
            self.register_parameter('scale', None)
        if bias:
            # 如果使用偏置项，则初始化偏置参数
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
        else:
            # 如果不使用偏置项，则不注册bias参数
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现前向传播过程，计算线性变换。

        该方法主要执行线性操作，即对输入张量x进行线性变换，通过矩阵乘法与权重参数相乘，并加上偏置参数。

        参数:
        x: torch.Tensor - 输入张量，通常代表一批数据或来自前一层的输出。

        返回:
        torch.Tensor - 经过线性变换后的输出张量。
        """
        return linear(x, self.weight, self.bias)

class ColumnParallelLinear(Linear):
    """
      列并行线性层类，继承自线性层类(Linear)。

      该类主要用于在并行计算框架中实现列并行的线性变换，目的是加速大规模矩阵运算。
      通过将输入矩阵按列分割，可以在多个处理器上并行执行矩阵乘法，最后合并结果。

      属性:
          in_features (int): 输入特征的数量。
          out_features (int): 输出特征的数量。
          bias (bool): 是否使用偏置项。
          parallelism (int): 并行度，即分割的列块数。
      """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        """
        初始化函数

        参数:
        in_features (int): 输入特征的数量
        out_features (int): 输出特征的数量，需要能够被world_size整除
        bias (bool): 是否使用偏置，默认为False
        dtype: 数据类型，默认为None

        异常:
        AssertionError: 如果out_features不能被world_size整除，则会抛出断言错误
        """
        # 确保输出特征数可以被世界大小整除，以支持并行计算
        assert out_features % world_size == 0
        # 计算每个世界部分的输出特征数
        self.part_out_features = out_features // world_size
        # 调用父类的初始化方法
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现前向传播过程，计算线性变换。

        该函数接收一个输入张量x，并应用线性变换，即矩阵乘法加上偏置，然后返回结果。

        参数:
        x (torch.Tensor): 输入张量，通常代表一批数据。

        返回:
        torch.Tensor: 经过线性变换后的输出张量。
        """
        # 应用线性变换，计算输出
        y = linear(x, self.weight, self.bias)
        # 返回计算结果
        return y

class RowParallelLinear(Linear):
    """
    行并行线性层类。

    该类继承自Linear类，用于在分布式环境中实现行并行的线性变换。
    行并行是将大型矩阵的行分割到不同的计算节点上进行并行计算，以提高计算效率和减少内存消耗。

    参数:
    - in_features (int): 输入特征的数量。
    - out_features (int): 输出特征的数量。
    - bias (bool): 是否使用偏置项，默认为True。
    - input_is_parallel (bool): 输入是否已经是并行分割的，默认为False。
    - gather_output (bool): 是否聚合输出，默认为True。
    - async_communication (bool): 是否使用异步通信，默认为False。
    - **kwargs: 其他传递给父类Linear的参数。

    属性:
    - weight (Tensor): 形状为[out_features, in_features]的权重参数。
    - bias (Tensor): 形状为[out_features]的偏置参数，如果使用偏置项的话。

    注意:
    - 该类主要用于大规模模型的分布式训练，能够有效提高计算效率和降低内存需求。
    - 参数input_is_parallel和gather_output控制输入和输出的并行行为，根据实际场景选择合适的配置。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        """
        初始化函数

        参数:
        in_features (int): 输入特征的数量
        out_features (int): 输出特征的数量
        bias (bool, optional): 是否使用偏置，默认为False
        dtype: 数据类型，可选，默认为None

        返回:
        无

        此函数是类的构造函数，用于初始化对象及其参数它确保输入特征可以均匀分布在世界大小上，并计算部分输入特征的数量
        """
        # 确保输入特征数能被世界大小整除，这是为了后续并行处理的需要
        assert in_features % world_size == 0
        # 计算每个世界大小部分的输入特征数
        self.part_in_features = in_features // world_size
        # 调用父类的初始化函数，传入部分输入特征数、输出特征数、是否使用偏置以及数据类型
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现前向传播过程。

        在前向传播中，此方法主要执行以下操作：
        1. 使用线性变换计算输入张量x与当前实例权重的乘积，得到输出y。
        2. 如果world_size大于1（即在分布式环境中），对y执行全局规约操作，以确保在所有分布式节点上的一致性。
        3. 如果偏置项self.bias存在，则将其加到y上，完成最终的输出计算。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 经过线性变换（和可能的分布式规约操作以及偏置添加）后的输出张量。
        """
        y = linear(x, self.weight)  # 执行线性变换
        if world_size > 1:
            dist.all_reduce(y)  # 在分布式环境中执行全局规约操作
        if self.bias is not None:
            y += self.bias  # 添加偏置项
        return y

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化函数，用于设置输入维度和防止除零错误的小 epsilon 值。

        参数:
        dim (int): 输入维度。
        eps (float): 防止除零错误的小 epsilon 值，默认为 1e-6。
        """
        # 初始化父类
        super().__init__()

        # 存储 epsilon 值用于后续操作
        self.eps = eps

        # 初始化权重参数，使用全为1的张量
        # 这里使用 nn.Parameter 是因为这些权重将被优化
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        对输入张量进行前向传播计算。

        该函数的目的是对输入张量x进行特定的数学变换，使其经过归一化和缩放操作，以达到预期的输出效果。
        主要步骤包括：将输入张量转换为浮点类型、计算归一化值、以及应用权重缩放。

        参数:
        - x (torch.Tensor): 输入张量，通常是一个批次的特征数据。

        返回:
        - torch.Tensor: 经过归一化和缩放操作后的输出张量。
        """
        # 确保输入张量为浮点类型，以便进行精确的数学计算
        x = x.float()

        # 计算输入张量的L2范数归一化，加上epsilon以避免除零错误
        # 这里的计算按通道进行，保持维度以支持广播操作
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # 将归一化后的输出与权重进行类型匹配，并应用权重缩放
        return y.type_as(self.weight) * self.weight

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算频率的复数表示，用于RoPE（旋转位置嵌入）。

    该函数根据模型参数计算频率值，并对其进行调整（如果序列长度大于原始序列长度），
    最后生成一个包含频率信息的复数张量。

    参数:
    - args: 包含模型参数的ModelArgs实例，包括qk_rope_head_dim、max_seq_len、beta_fast、
            beta_slow、rope_theta和rope_factor等。

    返回:
    - freqs_cis: 频率的复数表示，形状为[seqlen, dim // 2]。
    """
    # 获取模型参数
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        计算修正维度。

        该函数根据旋转次数、维度、基数和最大序列长度计算需要进行频率修正的维度。

        参数:
        - num_rotations: 旋转次数。
        - dim: 频率向量的维度。
        - base: RoPE中的基数。
        - max_seq_len: 最大序列长度。

        返回:
        - 修正维度的值。
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        计算修正范围。

        该函数根据低速和高速旋转次数、维度、基数和最大序列长度计算需要进行频率修正的维度范围。

        参数:
        - low_rot: 低速旋转次数。
        - high_rot: 高速旋转次数。
        - dim: 频率向量的维度。
        - base: RoPE中的基数。
        - max_seq_len: 最大序列长度。

        返回:
        - 修正维度的下限和上限。
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        生成线性坡道因子。

        该函数根据最小和最大值以及维度生成一个线性坡道因子，用于平滑频率修正。

        参数:
        - min: 最小修正维度。
        - max: 最大修正维度。
        - dim: 频率向量的维度。

        返回:
        - 线性坡道因子的张量。
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0 ,1)
        return ramp_func

    # 计算基础频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32)  / dim))
    # 如果序列长度大于原始序列长度，则进行频率修正
    if seqlen > args.original_seq_len:
        low , high = find_correction_range(beta_fast , beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 生成序列长度的张量
    t = torch.arange(seqlen)
    # 计算频率的外积
    freqs = torch.outer(t, freqs)
    # 生成频率的复数表示
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置嵌入（Rotary Position Embedding）到输入张量。

    该函数的目的是将旋转位置嵌入应用于输入张量x，以在自注意力机制中编码位置信息。
    它通过将输入张量的实部和虚部与预计算的复数频率相乘来实现这一点。

    参数:
    - x: 输入张量，其形状为(batch_size, seq_length, hidden_dim)。
    - freqs_cis: 复数频率张量，用于应用旋转位置嵌入。

    返回:
    - 应用了旋转位置嵌入后的张量。
    """
    # 保存输入张量的数据类型，以便在处理结束后恢复
    dtype = x.dtype

    # 将输入张量x转换为复数形式，以便进行旋转位置嵌入的计算
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))

    # 调整复数频率张量的形状，以使其与输入张量x兼容
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))

    # 应用旋转位置嵌入：将x与复数频率相乘，然后将结果转换回实数形式并展平
    y = torch.view_as_real(x * freqs_cis).flatten(3)

    # 将处理后的张量转换回原始数据类型，并返回
    return y.to(dtype)


class MLA(nn.Module):
    """
    多头注意力机制（MLA），结合了低秩适应（LoRA）和旋转位置嵌入（RoPE）机制。

    该类实现了一个多头注意力机制，适用于Transformer模型，集成了低秩适应（LoRA）以提高参数效率和旋转位置嵌入（RoPE）以更好地处理输入序列中的令牌位置。

    参数:
    - args (ModelArgs): 包含模型超参数和配置设置的命名空间。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 根据提供的参数初始化模型维度和其他参数。
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # 初始化查询、键和值的投影层。
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        # 根据序列长度和其他因素调整softmax缩放因子。
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 根据注意力实现方式初始化键和值的缓存。
        if attn_impl == 'naive':
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        多头注意力机制的前向传播。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。
        - start_pos (int): 输入序列的起始位置。
        - freqs_cis (torch.Tensor): 包含RoPE频率和相位信息的张量。
        - mask (Optional[torch.Tensor]): 可选的注意力掩码，应用于注意力分数。

        返回:
        - torch.Tensor: 应用多头注意力后的输出张量。
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == 'naive':
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_heads, self.qk_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


