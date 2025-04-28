import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
import pprint

from utils import Tool, Configuration


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        # self.stdio_context: Any | None = None
        self.sse_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection using SSE."""
        if not self.config.get("url"):
            raise ValueError("The URL must be provided in the configuration.")
        if not self.config.get("timeout"):
            raise ValueError("Timeout must be provided in the configuration.")
        if not self.config.get("sse_read_timeout"):
            raise ValueError("SSE read timeout must be provided in the configuration.")
        try:
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(
                    url=self.config["url"],
                    headers=self.config["headers"],
                    timeout=self.config["timeout"],
                    sse_read_timeout=self.config["sse_read_timeout"],
                )
            )
            read, write = sse_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()

        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

                for tool in item[1]:
                    print(tool)

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                # self.stdio_context = None
                self.sse_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")

class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, servers: list[Server], config: dict[str, Any]) -> None:
        self.servers: list[Server] = servers
        self.config: dict[str, Any] = config
        
    async def get_tools(self) -> list[Tool]:
        """Get tools from all servers.

        Returns:
            A list of tools from all servers.
        """
        tools_llm_format = []
        tools = await self.servers.list_tools()
        for tool in tools:
            tools_llm_format.append(tool.format_for_llm())
        return tools_llm_format
    





if __name__ == "__main__":
    config = Configuration()
    # Load server configuration
    server_config = config.load_config("./config.json")

    # Create a Server instance
    server = Server("MCP Server", server_config)

    # Run the server initialization and tool listing
    async def main():
        await server.initialize()
        tools = await server.list_tools()
        for tool in tools:
            print(tool.format_for_llm())
        # 关闭

        tool_response = await server.execute_tool(
            tool_name="maps_weather",
            arguments={
                "city": "北京",
            },
        )
        print(f"Tool response: {tool_response}")
        await server.cleanup()

    asyncio.run(main())


