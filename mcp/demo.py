import requests
import json
import asyncio  # 添加 asyncio 导入

from mcp.client.sse import sse_client
from mcp import ClientSession

import pprint

mcpServers= {
    "amap-amap-sse": {
        "url": "https://mcp.amap.com/sse?key=6f1c7170730dd6d79e7d8d87fdf0c9a2"
    }
}

async def main():
    async with sse_client("https://mcp.amap.com/sse?key=6f1c7170730dd6d79e7d8d87fdf0c9a2") as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = await session.list_tools()
            # print(tools.tools)

            for tool in tools.tools:
                print(tool.name)
                print(tool.description)
                print(tool.inputSchema)
                print('=========================================================\n\n')
                # print(await session.run_tool(tool))
    return session
    
if __name__ == "__main__":
    asyncio.run(main())


