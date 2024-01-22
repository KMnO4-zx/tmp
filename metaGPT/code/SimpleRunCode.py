#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   SimpleRunCode.py
@Time    :   2024/01/22 17:46:22
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import asyncio
import re
import subprocess

import fire

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message


class SimpleRunCode(Action):
    name: str = "SimpleRunCode"

    async def run(self, code_text: str):
        result = subprocess.run(["python3", "-c", code_text], capture_output=True, text=True)
        code_result = result.stdout
        logger.info(f"{code_result=}")
        return code_result

class SimpleWriteCode(Action):

    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    name: str = "SimpleWirteCode"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        resp = await self._aask(prompt)
        code_text = self.parse_code(resp)
        return code_text
    
    @staticmethod
    def parse_code(rep):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rep, re.DOTALL)
        code_text = match.group(1) if match else None
        return code_text


class RunnableCoder(Role):
    name: str = "KMnO4-zx-Runer"
    profile: str = "RunnableCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([SimpleWriteCode, SimpleRunCode])
        self._set_react_mode(react_mode="by_order")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        # By choosing the Action by order under the hood
        # todo will be first SimpleWriteCode() then SimpleRunCode()
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0]  # find the most k recent messages
        result = await todo.run(msg.content)

        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg
    
async def main():
    msg = "两数之和"
    role = RunnableCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)


if __name__== "__main__":
    asyncio.run(main())