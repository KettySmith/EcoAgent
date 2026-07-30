"""工具包 —— 汇总所有可用工具，供 ReAct 子智能体使用。"""

from typing import List

from eco_agent.tools.basic_tools import BASIC_TOOLS
from eco_agent.tools.sql_tools import SQL_TOOLS

# 所有工具的扁平列表
TOOLS: List = [
    *BASIC_TOOLS,
    *SQL_TOOLS,
]

__all__ = ["TOOLS"]
