# -*- coding: utf-8 -*-
# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus


import os
import random
from http import HTTPStatus
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
os.environ['DASHSCOPE_API_KEY'] = "sk-564d0387e773436bb170123e8a6cc7f1"
dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

def multi_round_conversation(query,system=None,history=None,model="qwen-72b-chat"):
    messages = []
    system = str(system)
    query = str(query)
    print("old history is ,",history)
    if history is None:
        if system is None or system == "None":
            messages.append({'role': 'system', 'content': 'You are a helpful assistant.'})
        else:
            print("system",system)
            messages.append({'role': 'system', 'content': system})
    else:
        print("History is Not None", history)
        messages = history
    messages.append({'role': 'user', 'content': query})
    print("messages" , messages)

    response = Generation.call(
        model=model,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message"  format.
    )
    if response.status_code == HTTPStatus.OK:
        messages.append({'role': response.output.choices[0]['message']['role'],
                         'content': response.output.choices[0]['message']['content']})
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
    print(response)
    response_content = response.output.choices[0]['message']['content']
    return response_content,messages


def multi_round_conversation_stream(query,system=None,history=None,model="qwen-72b-chat"):
    messages = []
    system = str(system)
    query = str(query)
    if history is None:
        if system is None or system == "None":
            messages.append({'role': 'system', 'content': 'You are a helpful assistant.'})
        else:
            messages.append({'role': 'system', 'content': system})
    else:
        messages = history
    messages.append({'role': 'user', 'content': query})

    responses = Generation.call(
        model=model,
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message"  format.
        stream=True,
        output_in_full=True  # get streaming output incrementally
    )
    full_content = ''
    response_content = ''
    new_message = None
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            full_content += response.output.choices[0]['message']['content']
            print(response.output.choices[0]['message']['content'])
            if response.output.choices[0]['finish_reason'] == "stop":
                new_message = response.output.choices[0]['message']
                response_content = response.output.choices[0]['message']['content']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    if new_message == None:
        pass
    else:
        messages.append( new_message )

    return response_content,messages





import re
def remove_html_tags_and_replace_with_newlines(text):
    # 定义HTML标签的正则表达式
    html_tag_pattern = re.compile(r'<[^>]+>')
    # 使用换行符替换所有HTML标签
    text_with_newlines = html_tag_pattern.sub('\n', text)
    # 将连续的换行符替换为单个换行符
    clean_text = re.sub(r'\n+', '\n', text_with_newlines)
    # 移除字符串首尾的换行符
    return clean_text.strip()

'''
history = None
system = None
if __name__ == '__main__':
    while True:
 
        query = remove_html_tags_and_replace_with_newlines(query)
        model = "qwen-72b-chat"
        system = "从现在开始，你是一名专业的文章信息内容提取专家，你可以结合文章信息内容，输出我期望你输出的目标格式文本。\n"
        response_content,history = multi_round_conversation(query=query,system= system,history=history,model=model)
        #response_content, history = multi_round_conversation_stream(query=query, system=system, history=history, model=model)
        print("response_content",response_content)
        print("history", history)
        print( "----------------------------\n" )
'''

