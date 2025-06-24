from transformers import PretrainedConfig # type: ignore
import torch.nn as nn 
import torch

class ModelConfig(PretrainedConfig):
    model_type = 'Tiny-K'
    def __init__(
            self,
            dim: int = 768, # 模型维度
            n_layers: int = 12, # Transformer的层数
            n_heads: int = 16, # 注意力机制的头数
            n_kv_heads: int = 8, # 键值头的数量
            vocab_size: int = 6144, # 词汇表大小
            hidden_dim: int = None, # 隐藏层维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5, # 归一化层的eps
            max_seq_len: int = 512, # 最大序列长度
            dropout: float = 0.0, # dropout概率
            flash_attn: bool = True, # 是否使用 Flash_Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

class RMSNorn(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps 是为了防止除以0的情况
        self.eps = eps
        # weight 是一个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算 RMSNorm 的核心部分
        # x.pow(2).mean(-1, keepdim=True) 计算了输入x的平方的均值
        # torch.rsqrt时平方根倒数，就这样得到了RMSNorm的分母部分
        # 最后乘以x，得到 RMSNorm 的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # forward 函数是模型的前向传播
        # 首先将输入 x 转为 float 类型，然后进行 RMSRNorm，最后再转回原来的数据类型
        # 最后乘以 weight，这是 RMSNorm 的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
# 分组查询注意力机制 （Grouped-Query Attention，GQA）
# 他可以提高模型的效率，并节省一些显存占用

#在 LLaMA2 模型中，我们需要将键和值的维度扩展到和查询的维度一样，这样才能进行注意力计算
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小，序列长度，键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果重复次数为1， 则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x
    
    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:,:,:, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

#旋转嵌入
