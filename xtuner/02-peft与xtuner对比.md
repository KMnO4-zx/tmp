# Peft 和 XTuner 的 QLora 微调使用

在此处我们统一使用 [OpneXLab](https://openxlab.org.cn/datasets?lang=zh-CN) 上的 [Math](https://openxlab.org.cn/datasets/OpenDataLab/MATH) 数据集。

两个微调脚本均在 InternStudio 平台上完成，感谢上海人工智能实验室提供算力资源。

## Peft QLora 方法

### Step-1 配置环境

创建好虚拟环境之后，首先安装一些必要的依赖（默认torch已经装好）

```bash
pip install transformers==4.36.2
pip install peft==0.4.0
pip install datasets==2.10.1
pip install accelerate==0.20.3
pip install tiktoken
pip install transformers_stream_generator
pip install bitsandbytes==0.41.1
```

### Step-2 数据集处理

我们需要的数据集形式是， `input`、`output`形式，如下：

```text
{
        "input": "",
        "output": ""
}
```

下载好数据集之后，用以下代码对数据集进行处理：

```python
res = []

for filepath, dirnames, filenames in os.walk('../../data/MATH/'):
    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(filepath, filename), 'r') as f:
                    data = json.load(f)
                    tmp = {
                        'input': data['problem'],
                        'output': data['solution'],
                    }
                    res.append(tmp)

with open('math.json', 'w') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
```

而后，我们要用`datasets`包加载数据集，并使用`internLM`模型的`tokenizer`将数据集转为`token`形式。

```python
# 使用datasets读取数据
df = pd.read_json('/root/data/math.json')
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("/root/model/internlm-chat-7b/", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def process_func(example):

    system_prompt = "You're a professor of mathematics."

    MAX_LENGTH = 512    # 分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<s><|System|>:{system_prompt}\n<|User|>:{example['input']}\n<|Bot|>:", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<s>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
```

注意：我们在`process_func`函数中对数据集的处理是有要求的，需要构建`internLM`模型的`Prompt_template`，具体可以参考[这里](https://huggingface.co/internlm/internlm-chat-7b/blob/739d3699446ad35bff7123c47e1e54bc9acdf79c/modeling_internlm.py#L859)的`build_inputs`函数。

```python
def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = [], meta_instruction=""):
    prompt = ""
    if meta_instruction:
        prompt += f"""<s><|System|>:{meta_instruction}\n"""
    else:
        prompt += "<s>"
    for record in history:
        prompt += f"""<|User|>:{record[0]}\n<|Bot|>:{record[1]}<eoa>\n"""
    prompt += f"""<|User|>:{query}\n<|Bot|>:"""
    return tokenizer([prompt], return_tensors="pt")
```

### Step-3 创建模型

这里我们使用了`4bits`量化了`InternLM`模型，并将精度类型设置为`nf4`，下方的代码也有比较详细的代码注释。

```python
model = AutoModelForCausalLM.from_pretrained(
    "/root/model/internlm-chat-7b/", 
    torch_dtype=torch.half, 
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,   # 是否使用低CPU内存
    load_in_4bit=True,  # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
    bnb_4bit_compute_dtype=torch.half,  # 4位精度计算的数据类型。这里设置为torch.half，表示使用半精度浮点数。
    bnb_4bit_quant_type="nf4", # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。
    bnb_4bit_use_double_quant=True  # 是否使用双精度量化。如果设置为True，则使用双精度量化。
    )
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
```

而后，创建 `LoraConfig` ，其中的`target_modules`参数就是我们要微调的模块，这里我们要微调的模块是`q_proj`、`k_proj`、`v_proj`。

```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
```

再然后就是，合并lora参数，创建`LoraModel`，并将`LoraModel`加载到`PEFT`中。

```python
model = get_peft_model(model, config)
```

可以使用这个函数`print_trainable_parameters()`来查看模型的参数量：

```python
model.print_trainable_parameters()
```

### Step-4 配置训练参数 && 训练模型

都是一些比较常用的参数，比如`batch_size`、`learning_rate`等等。

```python
args = TrainingArguments(
    output_dir="./output/math-internlm-chat-7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-5,
    save_on_each_node=True,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
```

这里我们基本使用了所有能降低训练显存的方法，比如`gradient_accumulation_steps`、`gradient_checkpointing`等等，并且也采用了`4bits`量化。

> 训练模型大概需要3个小时左右，训练期间显存占用在7.8G左右。

## XTuner QLora

### Step-1 配置环境

创建好虚拟环境之后，首先安装一些必要的依赖（默认torch已经装好），这里我用的xtuner 0.1.9

```bash
git clone -b v0.1.9  https://github.com/InternLM/xtuner
cd xtuner
# 从源码安装 XTuner
pip install -e '.[all]'
```

### Step-2 数据集处理

我们需要的数据集形式是， `input`、`output`形式，如下：

```text
{
        "conversation": [
            {
                "input": "请介绍一下你自己",
                "output": "我是不要葱姜蒜大佬的小助手，内在是上海AI实验室书生·浦语的7B大模型哦"
            }
        ]
}
```

下载好数据集之后，用以下代码对数据集进行处理：

```python
res = []

for filepath, dirnames, filenames in os.walk('../../data/MATH/'):
    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(filepath, filename), 'r') as f:
                    data = json.load(f)
                    tmp = {
                                "conversation": [
                                    {
                                        "input": data['problem'],
                                        "output": data['solution']
                                    }
                                ]
                            }
                    res.append(tmp)

with open('math_xtuner.json', 'w') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
```

### copy XTuner 配置文件


XTuner 使用就比较简单了，直接就是复制配置文件！

拷贝一个配置文件到当前目录：`xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}` 在本例中：（注意最后有个英文句号，代表复制到当前路径）

```bash
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```

然后有些地方需要稍稍修改，保持和Peft的脚本一致就行（可以对比一下效率）

```python
# PART 1 中
# 预训练模型存放的位置
pretrained_model_name_or_path = '/root/personal_assistant/model/Shanghai_AI_Laboratory/internlm-chat-7b'

# 微调数据存放的位置
data_path = '/root/personal_assistant/data/personal_assistant.json'

# 训练中最大的文本长度
max_length = 512

# 每一批训练样本的大小
batch_size = 2

# 最大训练轮数
max_epochs = 3

# system prompt
SYSTEM = "You're a professor of mathematics."

# 验证的频率
evaluation_freq = 90

# 用于评估输出内容的问题（用于评估的问题尽量与数据集的question保持一致）
evaluation_inputs = [
    'Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pairings are possible?  One such pairing is a blue bowl and a yellow glass.', 
    'The distance between two cities on a map is 15 inches. If the scale is 0.25 inches = 3 miles, how many miles apart are the actual cities?' 
                     ]

# PART 3 中
dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))
dataset_map_fn=None
```

### Step-3 训练模型

直接就是训练模型，这里我用的是`internlm-chat-7b`模型，训练大概需要2个小时左右，训练期间显存占用在14.5G左右。

XTuner 训练主打一个无脑，且训练简单。

手搓训练代码固然能加深对训练的认知，但当你真的要训练模型，我建议直接就是  

***XTuner，启动！***

```bash
xtuner train internlm_chat_7b_qlora_oasst1_e3_copy.py
```