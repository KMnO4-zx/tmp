# xtuner 使用

> 本笔记基于 Intern Studio 平台制作

## 安装 xtuner

step 1 

首先创建一个 `python 3.10` 纯净环境。默认读者都是会安装 torch 、transformers的，哈哈哈。

```bash
conda create --name xtuner0.1.9 python=3.10 -y
conda activate xtuner0.1.9
```

step 2 

安装 xtuner，默认在这一步之前torch和transformers都安装好了。

拉取最新xtuner仓库，从源码构建安装。

```bash
cd /root && mkdir code && cd code
git clone https://github.com/InternLM/xtuner.git
cd xtuner
```

从源码安装 XTuner

```bash
pip install -e '.[all]'
```

使用如下命令可以看到xtuner所有的内置训练配置：

> 这个配置文件就在 `xtuner/xtuner/configs` 路径下，大家也可以直接去看看。

```bash
xtuner list-cfg
```

## 数据 && 模型 && 配置文件

step 1

首先搞点数据，贴心的 Intern Studio 平台在share目录下为我们准备了很多数据集和模型。直接复制即可。

> share 目录下有很多数据集，大家随便算一个自己喜欢的即可。但每种数据集的格式不一样的，所以在需要选配配置文件，这个待会后面再讲~

```bash
cd /root
mkdir dataset && cd dataset
cp -r /root/share/temp/datasets/openassistant-guanaco .
```

step 2

搞个模型，贴心的 Intern Studio 平台在share目录下为我们准备了很多模型。直接复制即可。

```bash
cd /root
mkdir model && cp -r /root/share/temp/model_repos/internlm-chat-7b /root/model/Shanghai_AI_Laboratory
```

step 3

复制一个配置文件，然后修改一下配置文件的模型和数据集路径。

> 这里直接去`xtuner/xtuner/configs`目录下复制一个配置文件也行，用`xtuner copy-cfg`命令也可以。都行，看你喜欢！

```bash
cd /root/code && mkdir xtuner-demo && cd xtuner-demo
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```

step 4 

修改配置文件，如下所示，修改模型地址 `pretrained_model_name_or_path` 和，数据集地址 `data_path` 。

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/root/model/Shanghai_AI_Laboratory/internlm-chat-7b'

# Data
data_path = '/root/dataset/openassistant-guanaco'
prompt_template = PROMPT_TEMPLATE.internlm_chat
max_length = 2048
pack_to_max_length = True
```

> 这里修改完了之后，就已经可以开始微调了，但我还是想给大家解释一下几个参数的含义。
> 以下参数解读内容可以不看，不影响模型训练~

这一部分主要是对 `tokenizer` 和 `model` 的配置。可以看到在`quantization_config`这里对模型配置了量化参数，这里使用的是`4bits`量化，`nf4`映射。在lora配置中使用了默认的配置，lora缩放为4.

```python
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

```

这部分主要处理数据集，可以看到`dataset_map_fn`参数，如果你使用了和我不一样的数据集，就要修改这个参数啦，大家可以直接点进去源码看看，这里就不细说了。

`template_map_fn`是对模型`tokenizer input`的处理设置，假如你要配置不同的模型，那就需要修改这里了。

```python
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=data_path),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=oasst1_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
```

这里不用多说，就是训练参数的配置了，可以看到很多熟悉的参数，比如：`lr`、`weight_decay`、`max_epochs`、`max_norm`等等。

```python
#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = dict(
    type=CosineAnnealingLR,
    eta_min=0.0,
    by_epoch=True,
    T_max=max_epochs,
    convert_to_iter_based=True)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
```

## 训练！！！

进入训练配置文件的目录，然后开始训练就OK了，还是很简单的，也比较容易二次开发。

```bash
cd /root/code/xtuner-demo
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py
```

后续怎么加载lora文件，这就是老生常谈了，和正常加载没什么区别。