# Attention 机制的理解

> 原文：https://jaykmody.com/blog/attention-intuition/ 

> 译：不要葱姜蒜

ChatGPT和其他大型语言模型使用一种特殊的神经网络，称为变形金刚（吴恩达老师亲口说的）（transformer）。变形金刚的主要特点是注意力机制。注意力可以用以下方程定义：

$$ \text{attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

注意力有多种形式，但这种版本的注意力（称为缩放点积注意力）最初在原始的变形金刚论文中提出，并且仍然是许多基于变形金刚的神经网络的基础。在这篇文章中，我们将通过从基础出发推导这个方程来构建对上述方程的直观理解。

## Key-Value Lookups

注意力旨在解决的问题是键值查找。键值（kv）查找涉及三个组件：

- 一组 $n_k$ 键值对
- 一组 $n_q$ 查询，一一对应的值
- 一个查询（query），我们希望根据键匹配并获取相应的值

你可能熟悉这个概念，就像字典或哈希表一样：

```python
>>> d = {
>>>     "apple": 10,
>>>     "banana": 5,
>>>     "chair": 2,
>>> }
>>> d.keys()
['apple', 'banana', 'chair']
>>> d.values()
[10, 5, 2]
>>> query = "apple"
>>> d[query]
10
```

字典允许我们基于精确的字符串匹配执行查找。

如果我们想根据单词的含义进行查找，该怎么办?

## Key-Value Lookups based on Meaning

在我们之前的例子中，如果我们想要查找单词 "fruit"，我们如何选择最佳的 "key" 匹配呢？

显然 "chair" 不是一个好的匹配，但 "apple" 和 "banana" 似乎都是合适的匹配。很难选择其中一个，因为 "fruit" 感觉更像是 "apple" 和 "banana" 的结合，而不是严格匹配任何一个。

所以，我们不选择单一的匹配。相反，我们将采取 "apple" 和 "banana" 的组合。例如，假设我们为 "apple" 分配了 60% 的意义匹配，为 "banana" 分配了 40% 的匹配，为 "chair" 分配了 0% 的匹配。我们计算最终的输出值为这些值的加权和，权重由百分比决定：

```python
>>> query = "fruit"
>>> d = {"apple": 10, "banana": 5, "chair": 2}
>>> 0.6 * d["apple"] + 0.4 * d["banana"] + 0.0 * d["chair"]
8
```

在某种意义上，我们正在确定我们的查询应该根据意义向每个键值对支付多少“注意力”。这种“注意力”的数量以小数百分比的形式表示，称为注意力分数。数学上，我们可以将我们的输出定义为一个简单的加权和：

$$
\sum_{i} \alpha_i v_i
$$

其中 $\alpha_i$ 是第 $i$ 个键值对的注意力分数，$v_i$ 是第 $i$ 个值。记住，注意力分数是小数百分比，即它们必须在0到1之间（包括0和1）（ $0 \leq \alpha_i \leq 1$），并且它们的总和必须为1（$\sum_i a_i = 1$）。

但是，我们从哪里得到这些注意力分数呢？在我的示例中，我只是根据我的感觉选择了它们。虽然我认为我做得相当不错，但这种方法似乎不可持续（除非你能找到一种方法将我复制到你的电脑里）。

相反，让我们看看 **词向量** 如何帮助我们解决确定注意力分数的问题。

## Word Vectors and Similarity

想象我们用一组数字的向量来表示一个单词。理想情况下，向量中的值应该以某种方式捕捉它所代表的单词的意义。例如，我们可以有以下词向量（在二维空间中可视化）：

> 此处图片损坏，请自行想象

你可以看到相似的单词聚集在一起。水果聚集在右上角，蔬菜聚集在左上角，家具聚集在底部。实际上，你甚至可以看到蔬菜和水果群比它们与家具群更接近，因为它们是更密切相关的事物。

你甚至可以想象在词向量上进行算术运算。例如，给定单词“king”、“queen”、“man”和“woman”及其相应的向量表示 $\boldsymbol{v}_{\text{king}}$ ,  $\boldsymbol{v}_{\text{queen}}$ , $\boldsymbol{v}_{\text{man}}$ ,  $\boldsymbol{v}_{\text{women}}$ ，我们可以想象：

$$
\boldsymbol{v}_{\text{queen}} - \boldsymbol{v}_{\text{women}} + \boldsymbol{v}_{\text{man}} \sim \boldsymbol{v}_{\text{king}}
$$

也就是说，“queen”减去“women”加上“man”应该得到一个与“king”的向量相似的向量。

但两个向量之间的相似性到底是什么意思呢？在水果/蔬菜的例子中，相似性是通过向量之间的距离（即它们的欧几里得距离）来描述的。

还有其他方法可以衡量两个向量之间的相似性，每种方法都有其优缺点。可能最简单的两个向量之间相似性的度量是它们的点积：

$$
\boldsymbol{v} \cdot \boldsymbol{w} = \sum_{i} v_i w_i
$$

对我们来说，只需要知道：

- 如果两个向量指向相同的方向，点积将大于0（即相似）
- 如果它们指向相反的方向，点积将小于0（即不相似）
- 如果它们完全垂直，点积将为0（即中性）

利用这些信息，我们可以定义一个简单的启发式方法来确定两个词向量之间的相似性：点积越大，两个单词在意义上越相似。

好的，这些词向量实际上来自哪里呢？在神经网络的背景下，它们通常来自于某种学习到的嵌入或潜在表示。也就是说，最初词向量只是一些随机数字，但随着神经网络的训练，它们的值被调整，以成为越来越好的单词表示。神经网络是如何学习这些更好的表示的呢？这超出了这篇博客文章的范围，你需要参加一个深度学习的入门课程来了解。现在，我们只需要接受词向量的存在，并且它们以某种方式能够捕捉单词的含义。

## Attention Scores using the Dot Product

让我们回到水果的例子，但这次使用词向量来表示我们的单词。即 $\boldsymbol{q} = \boldsymbol{v}_{\text{fruit}}$ 和$\boldsymbol{k} = [\boldsymbol{v}_{\text{apple}} \ \boldsymbol{v}_{\text{banana}} \ \boldsymbol{v}_{\text{chair}}]$，使得 $\boldsymbol{v} \in \mathbb{R}^{d_k}$（即每个向量的维度都是 $d_k$，这是我们在训练神经网络时选择的一个值）。

使用我们的新点积相似性度量，我们可以计算查询和第 $i$ 个键之间的相似性：

$$
x_i = \boldsymbol{q} \cdot \boldsymbol{k}_i
$$

进一步推广，我们可以计算所有 $n_k$ 个键的点积：

$$
\boldsymbol{x} = \boldsymbol{q}{K}^T
$$

其中 $\boldsymbol{x}$ 是我们的点积向量 $\boldsymbol{x} = [x_1, x_2, \ldots, x_{n_k - 1}, x_{n_k}]$，$K$ 是我们的键向量的行向量矩阵（即将我们的键向量堆叠在一起形成一个 $n_k$ 乘 $d_k$ 的矩阵，使得 $k_i$ 是 $K$ 的第 $i$ 行）。如果你对这部分有困难，请参见下面的脚注。

记住，我们的注意力分数需要是小数百分比（在0到1之间，总和为1）。然而，我们的点积值可以是任何实数（即在 $-\infty$ 和$\infty$ 之间）。为了将我们的点积值转换为小数百分比，我们将使用softmax函数：

$$
\text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

```python
>>> import numpy as np
>>> def softmax(x):
>>>     # assumes x is a vector
>>>     return np.exp(x) / np.sum(np.exp(x))
>>>
>>> softmax(np.array([4.0, -1.0, 2.1]))
[0.8648, 0.0058, 0.1294]
```

注意：

- ✅ 每个数字都在0和1之间
- ✅ 数字总和为1
- ✅ 值较大的输入获得更多的“权重”
- ✅ 排序顺序被保留（即4.0仍然是最大的，-1.0仍然是最低的），这是因为softmax是一个单调函数

这满足了注意力分数的所有期望属性。因此，我们可以计算第 $i$ 个键值对的注意力分数：

$$
\alpha_i = \text{softmax}(\boldsymbol{x})_i = \text{softmax}(\boldsymbol{q}K^T)_i
$$

将其插入我们的加权和中，我们得到：

$$
\sum_{i}\alpha_iv_i = \sum_i \text{softmax}(\boldsymbol{x})_iv_i = \sum_i \text{softmax}(\boldsymbol{q}K^T)_i\boldsymbol{v_i} = \text{softmax}(\boldsymbol{q}K^T)v
$$

这就是我们的注意力方程。我们可以看到，它实际上是一个加权和，其中权重由我们的查询和键之间的点积决定，然后通过softmax函数进行归一化。

$$
\text{attention}(Q, K, V) = \text{softmax}(QK^T)v
$$

```python
import numpy as np

def get_word_vector(word, d_k=8):
    """Hypothetical mapping that returns a word vector of size
    d_k for the given word. For demonstrative purposes, we initialize
    this vector randomly, but in practice this would come from a learned
    embedding or some kind of latent representation."""
    return np.random.normal(size=(d_k,))

def softmax(x):
    # assumes x is a vector
    return np.exp(x) / np.sum(np.exp(x))

def attention(q, K, v):
    # assumes q is a vector of shape (d_k)
    # assumes K is a matrix of shape (n_k, d_k)
    # assumes v is a vector of shape (n_k)
    return softmax(q @ K.T) @ v

def kv_lookup(query, keys, values):
    return attention(
        q = get_word_vector(query),
        K = np.array([get_word_vector(key) for key in keys]),
        v = values,
    )

# returns some float number
print(kv_lookup("fruit", ["apple", "banana", "chair"], [10, 5, 2]))
```

## Scaled Dot Product Attention

原则上，我们在上一节推导出的注意力方程是完整的。然而，为了与 *Attention is All You Need* 中的版本相匹配，我们需要做一些改动。

### Values as Vectors

目前，键值对中的值只是数字。然而，我们也可以用一定大小的向量来代替它们$d_v$。比如，我们可以用一个$d_v$维的向量来表示每个值。

```python
d = {
    "apple": [0.9, 0.2, -0.5, 1.0]
    "banana": [1.2, 2.0, 0.1, 0.2]
    "chair": [-1.2, -2.0, 1.0, -0.2]
}
```

当我们通过加权求和计算输出时，我们将对向量而非数字执行加权求和（即进行标量-向量乘法而不是标量-标量乘法）。这是可取的，因为向量使我们能够存储/传达比单一数字更多的信息。

为了对我们的方程进行这种更改，我们不是将注意力分数乘以一个向量\(v\)，而是将它们乘以值向量的行矩阵\(V\)(类似于我们如何将键堆叠起来形成\(K\))：

$$
\text{attention}(\boldsymbol{q}, K, V) = \text{softmax}(\boldsymbol{q}K^T)V
$$

当然，我们的输出不再是一个标量，而是一个维度为 $d_v$ 的向量。

## Scaling

我们的查询和键之间的点积如果 $d_k$ 较大，其幅度可能会变得非常大。这会使`softmax`函数的输出更加极端。例如，`softmax([3, 2, 1]) = [0.665, 0.244, 0.090]`，但是当数值较大时，`softmax([30, 20, 10]) = [9.99954600e-01, 4.53978686e-05, 2.06106005e-09]`。在训练神经网络时，这意味着梯度会变得非常小，这是不利的。作为解决方案，我们通过 $\frac{1}{\sqrt{d_k}}$ 缩放我们的预softmax分数：

$$
\text{attention}(\boldsymbol{q}, K, V) = \text{softmax}(\frac{\boldsymbol{q}K^T}{\\sqrt{d_k}})V
$$

## Multiple Queries

在实践中，我们经常想要为 $n_q$ 个不同的查询执行多个查找，而不仅仅是单个查询。当然，我们总是可以一次又一次地这样做，将每个查询单独插入门限方程。然而，如果我们像处理 $K$ 和 $V$ 一样，将查询向量按行堆叠成一个矩阵 $Q$ ，我们可以计算出一个 $n_q$  by $d_v$ 的矩阵作为输出，其中行 $i$ 是第 $i$ 个查询的注意力输出向量：

$$
\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

也就是说，$\text{attention}(Q, K, V)_i = \text{attention}(q_i, K, V)$ 。

这比我们为每个查询依次运行注意力（例如，在for循环中）要快，因为我们可以并行化计算（特别是在使用GPU时）。

注意，我们的softmax输入成为一个矩阵，而不是一个向量。当我们在这里写softmax时，我们的意思是独立地对矩阵的每一行进行softmax，就像我们是在依次做事情一样。

## Result

因此，我们得到了原始论文中缩放点积注意力的最终方程：

$$
\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

code:

```python
import numpy as np

def softmax(x):
    # assumes x is a matrix and we want to take the softmax along each row
    # (which is achieved using axis=-1 and keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def attention(Q, K, V):
    # assumes Q is a matrix of shape (n_q, d_k)
    # assumes K is a matrix of shape (n_k, d_k)
    # assumes v is a matrix of shape (n_k, d_v)
    # output is a matrix of shape (n_q, d_v)
    d_k = K.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d_k)) @ V
```