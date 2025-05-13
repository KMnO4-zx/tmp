import torch
from torch import nn
from thop import profile 
from contextlib import redirect_stdout

from MHA import MultiHeadAttention
from MQA import MultiQueryAttention
from GQA import GroupedQueryAttention

def count_params_and_flops(module: nn.Module, input_shape: tuple):
    """
    统计指定模型模块的参数量和计算量(FLOPs)
    Args:
        module: PyTorch 模块对象
        input_shape: 输入张量的形状 (元组形式, 不包含 batch 维度)
    
    Returns:
        params_total: 总参数量
        flops_total: 总计算量
    """
    # 构造示例输入
    dummy_input = torch.randn(1, *input_shape)  # 添加 batch 维度
    
    # 计算参数量（单位：k）
    params_total = sum(p.numel() for p in module.parameters())
    
    # 计算计算量（单位：GFLOPs）
    with redirect_stdout(open("/dev/null", "w")):  # 屏蔽 thop 日志
        flops_total, _ = profile(module, inputs=(dummy_input,))
    
    return params_total, flops_total

if __name__ == '__main__':
    # 示例
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 8

    # 创建一个随机的 hidden_state
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    # 创建一个 attention mask (可选)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 5:] = 0  # 屏蔽掉每个 batch 中 seq_len 的后 5 个位置

    # 创建一个 MHA 实例
    mha = MultiHeadAttention(hidden_size, num_heads)
    # 通过 MHA 层
    mha_output = mha(hidden_state, attention_mask)
    # 打印输出形状
    print("MHA Output Shape:", mha_output.shape)
    # 统计参数量和计算量
    mha_params, mha_flops = count_params_and_flops(mha, (seq_len, hidden_size))
    print(f"MHA Params: {mha_params}, FLOPs: {mha_flops}")

    print("===" * 10)

    # 创建一个 MQA 实例
    mqa = MultiQueryAttention(hidden_size, num_heads)
    # 通过 MQA 层
    mqa_output = mqa(hidden_state, attention_mask)
    # 打印输出形状
    print("MQA Output Shape:", mqa_output.shape)
    # 统计参数量和计算量
    mqa_params, mqa_flops = count_params_and_flops(mqa, (seq_len, hidden_size))
    print(f"MQA Params: {mqa_params}, FLOPs: {mqa_flops}")

    print("===" * 10)

    # 创建一个 GQA 实例
    group_size = 2  # 每组 2 个头，共 4 组
    gqa = GroupedQueryAttention(hidden_size, num_heads, group_size)
    # 通过 GQA 层
    gqa_output = gqa(hidden_state, attention_mask)
    # 打印输出形状
    print("GQA Output Shape:", gqa_output.shape)
    # 统计参数量和计算量
    gqa_params, gqa_flops = count_params_and_flops(gqa, (seq_len, hidden_size))
    print(f"GQA Params: {gqa_params}, FLOPs: {gqa_flops}")