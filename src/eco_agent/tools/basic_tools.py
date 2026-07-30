"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import json

import json

from langchain_core.tools import tool
from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from eco_agent.context import Context

import ast
import operator as op

# async def search(query: str) -> Optional[dict[str, Any]]:
#     """Search for general web results.

#     This function performs a search using the Tavily search engine, which is designed
#     to provide comprehensive, accurate, and trusted results. It's particularly useful
#     for answering questions about current events.
#     """
#     runtime = get_runtime(Context)
#     wrapped = TavilySearch(max_results=runtime.context.max_search_results)
#     return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))



# ### 花费api token
# @tool
# async def search(query: str) -> str:
#     """Search for general web results.

#     This function performs a search using the Tavily search engine, which is designed
#     to provide comprehensive, accurate, and trusted results. It's particularly useful
#     for answering questions about current events.
#     """
#     runtime = get_runtime(Context)
#     wrapped = TavilySearch(max_results=runtime.context.max_search_results)
#     result = await wrapped.ainvoke({"query": query})

#     # 🔥关键：统一转成字符串
#     return str(result)


# 安全运算符
allowed_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def safe_eval(node):
    if isinstance(node, ast.Constant):  # number
        return node.value

    if isinstance(node, ast.BinOp):
        return allowed_operators[type(node.op)](
            safe_eval(node.left),
            safe_eval(node.right)
        )

    if isinstance(node, ast.UnaryOp):
        return allowed_operators[type(node.op)](
            safe_eval(node.operand)
        )

    raise ValueError("Unsupported expression")


@tool
def calculate(expression: str) -> str:
    """
    Safe mathematical calculation tool.

    Examples:
    - (100 + 200) * 3
    - (120 - 100) / 100 * 100
    - 50 * 0.13
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree.body)
        return json.dumps({
            "success": True,
            "result": result
        }, ensure_ascii=False)

    except Exception as e:

        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)

# @tool
# def lookup_value(
#     table: str,
#     year: str,
#     region: str,
#     metric: str,
# ) -> str:
#     return ""




BASIC_TOOLS: List[Callable[..., Any]] = [
    calculate
]
