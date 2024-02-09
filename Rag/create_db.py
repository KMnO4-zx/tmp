#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   create_db.py
@Time    :   2024/02/09 21:56:12
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import numpy as np
import pandas as pd
from openai import OpenAI

import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

client = OpenAI()

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if not magnitude:
        return 0
    return dot_product / magnitude

def read_file_content(file_path):
    # 根据文件扩展名选择读取方法
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.md'):
        return read_markdown(file_path)
    elif file_path.endswith('.txt'):
        return read_text(file_path)
    else:
        raise ValueError("Unsupported file type")

def read_pdf(file_path):
    # 读取PDF文件
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text

def read_markdown(file_path):
    # 读取Markdown文件
    with open(file_path, 'r', encoding='utf-8') as file:
        md_text = file.read()
        html_text = markdown.markdown(md_text)
        # 从HTML中提取纯文本
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text = text_maker.handle(html_text)
        return text

def read_text(file_path):
    # 读取文本文件
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
    return file_list


def get_chunk(text):
    """
    text: str
    return: chunk_text
    """
    max_token_len = 600
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    lines = text.split('\n')  # 假设以换行符分割文本为行

    for line in lines:
        line_len = len(enc.encode(line))
        if line_len > max_token_len:
            print('warning line_len = ', line_len)
        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = line
            curr_len = line_len
    
    if curr_chunk:
        chunk_text.append(curr_chunk)
    
    return chunk_text

file_list = get_files("data")
content = read_file_content(file_list[1])
print(content)