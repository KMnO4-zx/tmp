from metagpt.actions import Action
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message

from metagpt.logs import logger

import re
import asyncio

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
    

class SimpleCoder(Role):
    name: str = "KMnO4-zx"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([SimpleWriteCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0]
        code_text = await todo.run(msg.content)
        # 把信息以Message的形式存入记忆
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))
        return msg

async def main():
    msg = "write a function that calculates the sum of a list"
    role = SimpleCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

if __name__ == "__main__":
    asyncio.run(main())