"""SQL 生成与执行工具 —— 从原 sql_generate_node / sql_exec_node 重构而来。

供 numerical ReAct 子智能体按需调用：
- get_table_info  : 获取表的列结构 + 前 N 行预览
- generate_sql    : 用 LLM 生成 SQL（可带 sql_error 重试）
- execute_sql     : 执行 SELECT 查询并返回 Markdown 结果
"""

from __future__ import annotations

import json
import re
from typing import List

from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from eco_agent.context import Context
from eco_agent.prompts import SQL_GENERATION_PROMPT
from eco_agent.utils import load_chat_model, QueryExecutor
from eco_agent.state import State


# 旧版SQL相关节点

# ### SQL生成节点
# async def sql_generate_node(state: State, runtime: Runtime[Context]):

#     model = load_chat_model(runtime.context.model)
#     # plan = state.query_plan
#     print("sql生成节点的table_name:", state["table_name"])
 
#     prompt =SQL_GENERATION_PROMPT.format(
#         question=state["question"],
#         # query_plan=json.dumps(plan.__dict__, ensure_ascii=False, indent=2),
#         table=state["table"],
#         table_name=state["table_name"],
#         sql_error=state.get("sql_error", "")
#     )

#     response = await model.ainvoke(prompt)
#     sql = response.content.strip()
#     sql = re.sub(r"```sql", "", sql)
#     sql = re.sub(r"```", "", sql).strip()

#     return {
#         "sql": sql
       
#     }

# async def sql_exec_node( state: State,runtime: Runtime[Context]):

#     sql = state.get("sql", "")
#     ### 检查危险操作SQL
#     upper_sql = sql.upper()
#     FORBIDDEN_SQL = [
#     "DROP",
#     "DELETE",
#     "UPDATE",
#     "INSERT",
#     "ALTER"
#     ]
#     for keyword in FORBIDDEN_SQL:
#         if keyword in upper_sql:
#             return {
#                 "sql_success": False,
#                 "sql_error": f"Forbidden SQL keyword: {keyword}"
#             }
    
#     executor = QueryExecutor()
#     try:
#         headers, results = executor.execute_sql(sql)

#         # SQL执行失败
#         if headers is None:
#             return {
#                 "sql_success": False,
#                 "sql_error": "SQL execution failed，headers is None",
#             }

#         return {
#             "sql_success": True,
#             "sql_result": results,
#             "sql_headers": headers
#         }

#     except Exception as e:

#         return {
#             "sql_success": False,
#             "sql_error": str(e)
#         }

#     finally:
#         executor.close()



#从数据库获取表格信息
# @tool
# async def get_table_info(table_name: str, preview_rows: int = 5) -> str:
#     """获取数据库表结构信息，帮助编写正确的 SQL。

#     返回：列名列表 + 前 N 行数据预览（Markdown 表格）。
#     在写 SQL 之前先调用此工具，确认列名和数据类型是否正确。

#     Args:
#         table_name: 数据库中的表名
#         preview_rows: 预览行数，默认 5 行
#     """
#     executor = QueryExecutor()
#     try:
#         # 1. 获取列结构
#         headers, rows = executor.get_table_data(table_name, limit=preview_rows)

#         if headers is None:
#             return json.dumps({
#                 "success": False,
#                 "error": f"无法获取表 '{table_name}' 的结构信息，请检查表名是否正确。"
#             }, ensure_ascii=False)

#         # 2. 转为 Markdown 格式
#         markdown = executor.table_to_markdown(headers, rows)

#         return json.dumps({
#             "success": True,
#             "table_name": table_name,
#             "columns": headers,
#             "column_count": len(headers),
#             "preview": markdown
#         }, ensure_ascii=False)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": str(e)
#         }, ensure_ascii=False)
#     finally:
#         executor.close()


# ============================================================
# 工具 2：LLM 生成 SQL
# ============================================================

@tool
async def generate_sql(
    question: str,
    table_info: str,
    table_name: str,
    sql_error: str = ""
) -> str:
    """用专用 SQL 生成提示词生成 MySQL 查询语句。

    适用场景：
    - 多次执行 SQL 失败需要重新生成
    - 表结构复杂，需要借助专用生成能力
    - sql_error 参数用于传递上一次执行失败的错误信息

    Args:
        question:   用户的原始问题
        table_info: 表格结构/内容信息（如列名、示例数据等）
        table_name: 要查询的数据库表名
        sql_error:  上一次 SQL 执行的错误信息（重试时传入，首次为空）
    """
    runtime = get_runtime(Context)
    model = load_chat_model(runtime.context.model)

    prompt = SQL_GENERATION_PROMPT.format(
        question=question,
        table=table_info,
        table_name=table_name,
        sql_error=sql_error,
    )

    response = await model.ainvoke(prompt)
    sql = response.content.strip()
    sql = re.sub(r"```sql", "", sql)
    sql = re.sub(r"```", "", sql).strip()

    return sql


# ============================================================
# 工具 3：执行 SQL
# ============================================================

# 禁止的关键词（安全约束）
_FORBIDDEN_SQL_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT",
    "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE",
]


@tool
async def execute_sql(sql_query: str) -> str:
    """执行 MySQL SELECT 查询，返回 Markdown 表格格式的结果。

    安全约束：仅允许 SELECT 查询，任何修改/删除操作都会被拒绝。
    执行前会检查危险关键词（DROP、DELETE、UPDATE 等）。

    Args:
        sql_query: 一条合法的 MySQL SELECT 查询语句
    """
    # ---------- 安全检查 ----------
    upper_sql = sql_query.upper()
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if keyword in upper_sql:
            return json.dumps({
                "success": False,
                "error": f"禁止的 SQL 关键词: {keyword}。仅允许 SELECT 查询。"
            }, ensure_ascii=False)

    # ---------- 执行 ----------
    executor = QueryExecutor()
    try:
        headers, results = executor.execute_sql(sql_query)

        if headers is None:
            return json.dumps({
                "success": False,
                "error": "SQL 执行失败：未返回列头信息，请检查 SQL 语法。"
            }, ensure_ascii=False)

        # 结果转为 Markdown 表格（LLM 更易读）
        markdown = executor.table_to_markdown(headers, results)

        return json.dumps({
            "success": True,
            "headers": headers,
            "row_count": len(results) if results else 0,
            "result": markdown,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)
    finally:
        executor.close()


# ============================================================
# 本模块的工具汇总
# ============================================================

SQL_TOOLS: List = [
    generate_sql,
    execute_sql,
]
