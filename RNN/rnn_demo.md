# 熟悉torch.nn.RNN

`torch.nn.RNN`参考文档：https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#

> - `torch.nn.RNN(*self*, *input_size*, *hidden_size*, *num_layers=1*, *nonlinearity='tanh'*, *bias=True*, *batch_first=False*, *dropout=0.0*, *bidirectional=False*, *device=None*, *dtype=None*)`
>     1. `input_size`：输入特征的维度。它表示输入序列中每个元素的特征数量。
>     2. `hidden_size`：隐藏状态的维度。它表示每个隐藏状态向量的维度。
>     3. `num_layers`：循环神经网络的层数。例如，设置`num_layers`=2意味着将两个`RNN`堆叠在一起形成一个堆叠`RNN`，第二个`RNN`接受第一个`RNN`的输出并计算最终结果。默认值为1。
>     4. `nonlinearity`：激活函数。可以是'`tanh`'（双曲正切）或'`relu`'（修正线性单元）。默认值为'`tanh`'。
>     5. `bias`：如果设置为`False`，则层不使用输入隐藏权重`b_ih`和隐藏隐藏权重`b_hh`。默认值为`True`。
>     6. `batch_first`：如果设置为True，则输入和输出张量的形状为(`batch`, `seq`, `feature`)；如果设置为False，则形状为(`seq`, `batch`, `feature`)。注意，这并不适用于隐藏或单元状态。默认值为`False`。
>     7. `dropout`：如果非零，则在除最后一层之外的每个`RNN`层的输出上引入一个`Dropout`层，其丢弃概率等于`dropout`。默认值为0。
>     8. `bidirectional`：如果设置为`True`，则成为双向`RNN`。默认值为`False`。

>- Inputs: input, h_0
>
>    1. `input`：输入序列的张量。对于非批量输入，形状应为(L, H_in)，其中L是序列长度，H_in是输入特征的维度。对于批量输入，形状应为(N, L, H_in)，其中N是批量大小，L是序列长度，H_in是输入特征的维度。
>
>    2. `h_0`：初始隐藏状态的张量。对于非批量输入，形状应为(`num_layers `* `num_directions`, `H_out`)，其中`num_layers`是循环神经网络的层数，`num_directions`是循环神经网络的方向数（1表示单向，2表示双向），H_out是隐藏状态的维度。对于批量输入，形状应为(N, `num_layers `* `num_directions`,
>
>        | 缩写      | 名称                                |
>        | --------- | ----------------------------------- |
>        | N         | batch size                          |
>        | L         | sequence length                     |
>        | D         | 2 if bidirectional=True otherwise 1 |
>        | $H_{in}$  | input size                          |
>        | $H_{out}$ | hidden size                         |

> - Outputs: output, h_n
>     1. `output`：输出序列的张量。对于非批量输入，形状应为(L, D * H_out)，其中L是序列长度，D是循环神经网络的方向数（1表示单向，2表示双向），H_out是隐藏状态的维度。对于批量输入，形状应为(N, L, D * H_out)，其中N是批量大小，L是序列长度，D是循环神经网络的方向数，H_out是隐藏状态的维度。如果输入是一个`torch.nn.utils.rnn.PackedSequence`，输出也将是一个打包的序列。
>     2. `h_n`：最终隐藏状态的张量。对于非批量输入，形状应为(num_layers * num_directions, H_out)，其中num_layers是循环神经网络的层数，num_directions是循环神经网络的方向数（1表示单向，2表示双向），H_out是隐藏状态的维度。对于批量输入，形状应为(N, `num_layers `* `num_directions`, `H_out`)，其中N是批量大小，`num_layers`是循环神经网络的层数，`num_directions`是循环神经网络的方向数，`H_out`是隐藏状态的维度。

```python
import torch
import torch.nn as nn
```

$ h_t $状态公式
$$
h_t = tanh(W_{ih}*x_t+b_{ih}+W_{hh}*h_{t-1}+b_{hh})
$$


```python
# 单向、单层rnn
single_rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=1, batch_first=True) # batch_first=True表示输入数据的维度为[batch_size, seq_len, input_size]
input = torch.randn(1, 5, 4) # 输入数据维度为[batch_size, seq_len, input_size]
output, h_n = single_rnn(imput) # output维度为[batch_size, seq_len, hidden_size=3]，h_n维度为[num_layers=1, batch_size, hidden_size=3]
print(output, output.shape, h_n, h_n.shape,  sep='\n')
```

    tensor([[[-0.6629, -0.7484,  0.3153],
             [ 0.0195,  0.4842,  0.1950],
             [ 0.0107, -0.3933, -0.0298],
             [-0.5639,  0.7052, -0.1440],
             [ 0.6051, -0.5736,  0.4397]]], grad_fn=<TransposeBackward1>)
    torch.Size([1, 5, 3])
    tensor([[[ 0.6051, -0.5736,  0.4397]]], grad_fn=<StackBackward0>)
    torch.Size([1, 1, 3])


output是一个三维张量，维度分别为[batch_size, seq_len, hidden_size]

$h_n: $ 最终输出结果的$h_n$三个维度，第一个维度和`num_layers`相关，第二个维度和`batch_size`相关，第三个维度和`hidden_size`相关


```python
output[:, 2, :] # 二维数组[batch_size, hidden_size]
```


    tensor([[ 0.0107, -0.3933, -0.0298]], grad_fn=<SliceBackward0>)


```python
# 双向、单层rnn
bi_rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=1, batch_first=True, bidirectional=True)
bi_output, bi_h_n = bi_rnn(imput)
print(bi_output, bi_output.shape, bi_h_n, bi_h_n.shape, sep='\n')
```

    tensor([[[ 0.1199,  0.3156,  0.0051,  0.7060,  0.4004, -0.3551],
             [-0.7382, -0.7216, -0.3452,  0.0537,  0.4494, -0.5666],
             [-0.3262,  0.4341,  0.7014,  0.2500,  0.3134, -0.4126],
             [-0.0573, -0.8624, -0.3747,  0.5334,  0.7036, -0.0644],
             [-0.9678,  0.3215,  0.4936,  0.8410, -0.2907,  0.1378]]],
           grad_fn=<TransposeBackward1>)
    torch.Size([1, 5, 6])
    tensor([[[-0.9678,  0.3215,  0.4936]],
    
            [[ 0.7060,  0.4004, -0.3551]]], grad_fn=<StackBackward0>)
    torch.Size([2, 1, 3])


# 单向RNN手写实现


```python
import torch
import torch.nn as nn
```


```python
batch_size, seq_len, input_size, hidden_size = 2, 3, 2, 3 # 批次大小、序列长度、输入维度、隐藏层维度
num_layers = 1 # rnn层数

input = torch.randn(batch_size, seq_len, input_size) # 初始化输入数据
h_prev = torch.zeros(batch_size, hidden_size) # 初始化隐藏层状态
```

## step 1 调用pytorch实现单向，单层rnn


```python
rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # 初始化rnn
rnn_output, h_n = rnn(input, h_prev.unsqueeze(0)) # rnn输出和隐藏层状态
print(rnn_output, rnn_output.shape, h_n, h_n.shape, sep='\n')
```

    tensor([[[ 0.0372, -0.2753,  0.1908],
             [ 0.2962, -0.3922, -0.3102],
             [ 0.7279, -0.0109, -0.3366]],
    
            [[ 0.8038, -0.0675, -0.4083],
             [ 0.7672,  0.1979, -0.3658],
             [ 0.8139,  0.2514, -0.4217]]], grad_fn=<TransposeBackward1>)
    torch.Size([2, 3, 3])
    tensor([[[ 0.7279, -0.0109, -0.3366],
             [ 0.8139,  0.2514, -0.4217]]], grad_fn=<StackBackward0>)
    torch.Size([1, 2, 3])

```python
rnn.state_dict()
```


    OrderedDict([('weight_ih_l0',
                  tensor([[ 0.2614, -0.2825],
                          [ 0.0188, -0.0833],
                          [-0.3619, -0.0004]])),
                 ('weight_hh_l0',
                  tensor([[-0.2454, -0.0358, -0.5600],
                          [ 0.0774,  0.2310, -0.5770],
                          [-0.1034,  0.2437,  0.0019]])),
                 ('bias_ih_l0', tensor([ 0.4399, -0.2331,  0.0026])),
                 ('bias_hh_l0', tensor([ 0.2184,  0.0657, -0.2305]))])

## step 2 手写一个rnn_forward函数

$$
h_t = tanh(W_{ih}*x_t+b_{ih}+W_{hh}*h_{t-1}+b_{hh})
$$


```python
def rnn_forward(input, W_ih, W_hh, b_ih, b_hh, h_prev):
    batch_size, seq_len, input_size = input.shape
    hidden_size = W_ih.shape[0] # 隐藏层维度, seq_len就等于hidden_size，所以是W_ih.shape[0]
    h_output = torch.zeros(batch_size, seq_len, hidden_size) # 初始化一个输出矩阵output 看官方参数来定义
    for t in range(seq_len):
        x_t = input[:, t, :].unsqueeze(2) # input[:,t,:].shape = [batch_size,input_size] -> (batch_size,input_size,1)

        # w_ih_batch.shape = [hidden_size,input_size]->(1,hidden_size,input_size)->(batch_size,hidden_size,input_size)
        # tile(batch_size, 1, 1): 第0维变成原来的batch_size倍（默认行复制）其他两维为1保持不动-> (batch_size,hidden_size,input_size)
        w_ih_batch = W_ih.unsqueeze(0).tile(batch_size, 1, 1)

        # w_hh_batch.shaoe = [hidden_size,input_size]->(1,hidden_size,input_size)->(batch_size,hidden_size,input_size)
        w_hh_batch = W_hh.unsqueeze(0).tile(batch_size, 1, 1)

        # w_ih_times_x.shape=(batch_size,hidden_size,1) -> (batch_size,hidden_size)
        w_ih_times_x = torch.bmm(w_ih_batch, x_t).squeeze(-1)  # W_ih * x_t

        # h_prev.unsqueeze(2) : (batch_size,hidden_size,1)
        # w_hh_times_h.shape =(batch_size,hidden_size,1)->(batch_size,hidden_size)
        w_hh_times_h = torch.bmm(w_hh_batch, h_prev.unsqueeze(2)).squeeze(-1)

        # h_prev = (1,batch_size,hidden_size)->(batch_size, hidden_size)
        h_prev = torch.tanh(w_ih_times_x + b_ih + w_hh_times_h + b_hh)

        h_output[:,t,:] = h_prev
        
    # 按官方api格式返回
    # h_prev.unsqueeze(0) : (1,batch_size,hidden_size) 因为官方参数为(D∗num_layers,bs,hidden_size)
    return h_output, h_prev.unsqueeze(0)
```

## 验证一下写的对不对


```python
rnn_output, h_n = rnn(input, h_prev.unsqueeze(0))
custom_output, custom_hn = rnn_forward(input, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0, h_prev)
print('custom', rnn_output, rnn_output.shape, h_n, h_n.shape, sep='\n')
print('torch api', custom_output, custom_output.shape, custom_hn, custom_hn.shape, sep='\n')
```

    custom
    tensor([[[ 0.0372, -0.2753,  0.1908],
             [ 0.2962, -0.3922, -0.3102],
             [ 0.7279, -0.0109, -0.3366]],
    
            [[ 0.8038, -0.0675, -0.4083],
             [ 0.7672,  0.1979, -0.3658],
             [ 0.8139,  0.2514, -0.4217]]], grad_fn=<TransposeBackward1>)
    torch.Size([2, 3, 3])
    tensor([[[ 0.7279, -0.0109, -0.3366],
             [ 0.8139,  0.2514, -0.4217]]], grad_fn=<StackBackward0>)
    torch.Size([1, 2, 3])
    torch api
    tensor([[[ 0.0372, -0.2753,  0.1908],
             [ 0.2962, -0.3922, -0.3102],
             [ 0.7279, -0.0109, -0.3366]],
    
            [[ 0.8038, -0.0675, -0.4083],
             [ 0.7672,  0.1979, -0.3658],
             [ 0.8139,  0.2514, -0.4217]]], grad_fn=<CopySlices>)
    torch.Size([2, 3, 3])
    tensor([[[ 0.7279, -0.0109, -0.3366],
             [ 0.8139,  0.2514, -0.4217]]], grad_fn=<UnsqueezeBackward0>)
    torch.Size([1, 2, 3])

