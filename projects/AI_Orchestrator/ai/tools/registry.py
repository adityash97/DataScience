import logging

from . import calculator_tool, database_tool, web_search_tool

logger = logging.getLogger(__name__)

_TOOLS = {
    web_search_tool.name: web_search_tool,
    database_tool.name: database_tool,
    calculator_tool.name: calculator_tool,
}


def get_tool(tool_name: str):
    tool = _TOOLS.get(tool_name)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found. Available: {list_tools()}")
    return tool


def list_tools() -> list[str]:
    return list(_TOOLS.keys())


def run_tool(tool_name: str, payload: dict) -> dict:
    logger.info('[ToolRegistry] Running tool: %s', tool_name)
    tool = get_tool(tool_name)
    return tool.run(payload)
