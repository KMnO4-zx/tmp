#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   model_download.py
@Time    :   2023/11/04 17:55:41
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

# os.system('pip install -U huggingface_hub hf_transfer') # 安装依赖

import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

"""
    - distilbert-base-uncased-finetuned-sst-2-english
"""
# 下载模型
os.system('huggingface-cli download --resume-download distilbert-base-uncased-finetuned-sst-2-english --local-dir distilbert-base-uncased-finetuned-sst-2-english')
