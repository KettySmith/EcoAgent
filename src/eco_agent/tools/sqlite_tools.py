"""SQLite 内存表工具 —— 针对 Meilisearch 检索到的 Markdown 表格进行 SQL 查询。

与 sql_tools.py（MySQL 数据库表）不同，本模块的工具操作的是：
  1. 从 Meilisearch 检索到的 JSON → Markdown 表格字符串（由 rag_node._format_table_docs 生成）
  2. 这些 Markdown 表格会被解析并加载到 SQLite **内存**数据库
  3. LLM 可以像查询普通数据库一样对这些临时表生成并执行 SQL

模块组成：
  - TempTableManager : 管理 SQLite 内存连接，支持多表共存
  - create_table_from_markdown : 解析 Markdown 表格 → 创建 SQLite 内存表
  - generate_sqlite_sql         : LLM 生成 SQLite 查询（复用 SQL_GENERATION_PROMPT）
  - execute_sqlite              : 在内存表上执行 SELECT 查询
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from eco_agent.context import Context
from eco_agent.prompts import SQLITE_GENERATION_PROMPT
from eco_agent.utils import load_chat_model


# ============================================================
# SQLite 内存表管理器（模块级单例，同一 ReAct 循环内共享）
# ============================================================

class TempTableManager:
    """管理 SQLite 内存数据库中的临时表。

    每个表使用独立的 :memory: 连接（线程安全），支持：
    - 从 Markdown 表格创建表
    - 查询表结构
    - 执行 SELECT
    """

    def __init__(self):
        # table_name -> {"conn": sqlite3.Connection, "columns": [(col_name, col_type), ...], "row_count": int}
        self._tables: Dict[str, dict] = {}

    def _infer_sqlite_type(self, value: str) -> str:
        """推断 SQLite 列类型。返回 'REAL' 或 'TEXT'。"""
        if value is None or value.strip() == "":
            return "TEXT"
        try:
            float(value)
            return "REAL"
        except (ValueError, TypeError):
            return "TEXT"

    def _parse_markdown_table(
        self, markdown: str
    ) -> Tuple[Optional[str], Optional[List[str]], Optional[List[List[str]]]]:
        """从 Markdown 字符串中解析出表格。

        支持的格式：
        1. rag_node._format_table_docs 的 key-value 格式（| 指标 | 值 |）
        2. 通用多列 Markdown 表格

        Returns:
            (title, headers, rows) 或 (None, None, None)
        """
        # 提取标题（【...】格式）
        title = ""
        title_match = re.search(r"【(.+?)】", markdown)
        if title_match:
            title = title_match.group(1).strip()

        # 找到 Markdown 表格行：以 | 开头、| 结尾
        table_lines: List[str] = []
        in_table = False
        for line in markdown.split("\n"):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                table_lines.append(stripped)
                in_table = True
            elif in_table and stripped == "":
                # 空行结束表格
                break
            elif in_table and not stripped.startswith("|"):
                break

        if len(table_lines) < 2:
            return (None, None, None)

        # 解析表头（第一行）
        header_cells = [c.strip() for c in table_lines[0].split("|")[1:-1]]
        if not header_cells:
            return (None, None, None)

        # 跳过分隔行（|---|---|）
        data_start = 1
        if table_lines[1].replace("|", "").replace("-", "").replace(" ", "").strip() == "":
            data_start = 2

        # 解析数据行
        rows: List[List[str]] = []
        for line in table_lines[data_start:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # 补齐缺少的列（防御）
            while len(cells) < len(header_cells):
                cells.append("")
            rows.append(cells[: len(header_cells)])

        if not rows:
            return (None, None, None)

        return (title, header_cells, rows)

    def _sanitize_table_name(self, raw_name: str) -> str:
        """将原始表名转为合法的 SQLite 表名。

        非法字符替换为下划线，前导数字前加前缀。
        """
        if not raw_name:
            return "temp_table"

        # 只保留中文、字母、数字、下划线
        safe = re.sub(r"[^\w一-鿿]", "_", raw_name).strip("_")

        if not safe:
            return "temp_table"

        # SQLite 允许中文表名，但需要用双引号包裹
        # 为简单起见，统一生成一个英文别名
        # 这里保留原样，在实际建表时用双引号包裹
        return safe

    def create_table(self, markdown: str, table_name: str = "") -> str:
        """解析 Markdown 表格，在 SQLite 内存中建表。

        Args:
            markdown:   Markdown 表格字符串
            table_name: 自定义表名（可选，不传则从 Markdown 标题提取）

        Returns:
            JSON 字符串，包含建表结果信息
        """
        title, headers, rows = self._parse_markdown_table(markdown)

        if headers is None:
            return json.dumps({
                "success": False,
                "error": "无法从输入中解析出 Markdown 表格，请确认格式为 | col1 | col2 | ... |"
            }, ensure_ascii=False)

        # 确定表名
        raw_name = table_name or title or "temp_table"
        safe_name = self._sanitize_table_name(raw_name)

        # 去重：同名表追加序号
        original_name = safe_name
        counter = 1
        while safe_name in self._tables:
            counter += 1
            safe_name = f"{original_name}_{counter}"

        # 推断列类型（基于第一行数据）
        col_defs: List[str] = []
        col_names: List[str] = []
        sample_row = rows[0] if rows else []

        for i, h in enumerate(headers):
            # 列名清洗
            col_name = re.sub(r"[^\w一-鿿]", "_", h).strip("_")
            if not col_name:
                col_name = f"col_{i}"
            # 确保列名不以数字开头
            if col_name[0].isdigit():
                col_name = f"c_{col_name}"
            col_names.append(col_name)

            # 推断类型
            sample_val = sample_row[i] if i < len(sample_row) else ""
            col_type = self._infer_sqlite_type(sample_val)
            col_defs.append(f'"{col_name}" {col_type}')

        # 创建 SQLite 内存连接
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        create_sql = f'CREATE TABLE "{safe_name}" ({", ".join(col_defs)})'

        try:
            conn.execute(create_sql)

            # 插入数据
            if rows:
                placeholders = ", ".join(["?" for _ in col_names])
                quoted_cols = [f'"{c}"' for c in col_names]
                insert_sql = f'INSERT INTO "{safe_name}" ({", ".join(quoted_cols)}) VALUES ({placeholders})'

                for row in rows:
                    values = []
                    for i in range(len(col_names)):
                        val = row[i] if i < len(row) else ""
                        if col_defs[i].endswith("REAL"):
                            try:
                                values.append(float(val))
                            except (ValueError, TypeError):
                                values.append(val)
                        else:
                            values.append(val)
                    conn.execute(insert_sql, values)

            conn.commit()

        except sqlite3.Error as e:
            conn.close()
            return json.dumps({
                "success": False,
                "error": f"SQLite 建表/插入失败: {str(e)}"
            }, ensure_ascii=False)

        # 记录
        self._tables[safe_name] = {
            "conn": conn,
            "columns": list(zip(col_names, [d.split()[-1] for d in col_defs])),
            "row_count": len(rows),
            "title": title,
        }

        # 返回表信息
        col_info = [f"{name} ({col_type})" for name, col_type in self._tables[safe_name]["columns"]]

        return json.dumps({
            "success": True,
            "table_name": safe_name,
            "title": title,
            "columns": col_info,
            "column_count": len(col_names),
            "row_count": len(rows),
            "message": f"表 '{safe_name}' 创建成功，包含 {len(col_names)} 列、{len(rows)} 行数据。",
        }, ensure_ascii=False)

    def get_table_schema(self, table_name: str) -> Optional[str]:
        """获取已建内存表的列结构 Markdown 描述。"""
        info = self._tables.get(table_name)
        if info is None:
            return None

        lines = [f"表名: {table_name}"]
        if info.get("title"):
            lines.append(f"标题: {info['title']}")
        lines.append(f"行数: {info['row_count']}")
        lines.append("")
        lines.append("| 列名 | 类型 |")
        lines.append("| --- | --- |")
        for col_name, col_type in info["columns"]:
            lines.append(f"| {col_name} | {col_type} |")

        return "\n".join(lines)

    def execute(self, table_name: str, sql: str) -> Tuple[Optional[List[str]], Optional[List[tuple]]]:
        """在指定内存表上执行 SELECT 查询。

        Returns:
            (headers, rows) 或 (None, None)
        """
        info = self._tables.get(table_name)
        if info is None:
            return None, None

        conn = info["conn"]
        try:
            cur = conn.execute(sql)
            headers = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            return headers, rows
        except sqlite3.Error as e:
            raise e

    def has_table(self, table_name: str) -> bool:
        return table_name in self._tables

    def close_all(self):
        """关闭所有连接。"""
        for info in self._tables.values():
            try:
                info["conn"].close()
            except Exception:
                pass
        self._tables.clear()

    def close_table(self, table_name: str):
        """关闭并移除指定表。"""
        info = self._tables.pop(table_name, None)
        if info:
            try:
                info["conn"].close()
            except Exception:
                pass


# 模块级单例
_temp_manager = TempTableManager()


# ============================================================
# 工具 1：Markdown → SQLite 内存表
# ============================================================

@tool
async def create_table_from_markdown(markdown_table: str, table_name: str = "") -> str:
    """将 Markdown 表格字符串解析并加载到 SQLite 内存数据库，创建一个临时表。

    适用场景：
    - 用户的 retrieved_tables 中包含 Markdown 格式的表格数据
    - 需要对这些数据进行 SQL 查询（排序、筛选、聚合等）
    - 在调用 execute_sqlite 之前必须先调用此工具建表

    Args:
        markdown_table: Markdown 表格字符串，格式为 | col1 | col2 | ... |
                        支持从 rag_node 检索结果中直接传入
        table_name:     自定义表名（可选），不传则从 Markdown 标题自动提取

    Returns:
        JSON，包含表名、列信息、行数，后续执行 SQL 时需要用到表名
    """
    return _temp_manager.create_table(markdown_table, table_name)


# ============================================================
# 工具 2：LLM 生成 SQLite SQL
# ============================================================

@tool
async def generate_sqlite_sql(
    question: str,
    table_info: str,
    table_name: str,
    sql_error: str = ""
) -> str:
    """用 LLM 生成针对 SQLite 内存临时表的查询 SQL。

    适用场景：
    - 需要根据用户问题对已创建的临时表生成 SQL 查询
    - 上一次 SQL 执行失败，需要根据错误信息修正 SQL
    - 表结构复杂，需要借助 LLM 生成正确的 SQL

    Args:
        question:    用户的原始问题
        table_info:  表格结构/内容描述（可从 create_table_from_markdown 的返回结果获取，
                     或直接用 Markdown 表格原始内容）
        table_name:  已创建的 SQLite 临时表名（必须与 create_table_from_markdown 返回的一致）
        sql_error:   上一次 SQL 执行的错误信息（重试时传入，首次为空）

    Returns:
        纯 SQL 语句字符串
    """
    runtime = get_runtime(Context)
    model = load_chat_model(runtime.context.model)

    prompt = SQLITE_GENERATION_PROMPT.format(
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
# 工具 3：执行 SQLite 查询
# ============================================================

_FORBIDDEN_SQLITE_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT",
    "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE",
    "ATTACH", "DETACH",
]


def _table_to_markdown(headers: List[str], rows: List[tuple], max_rows: int = 50) -> str:
    """将查询结果转为 Markdown 表格。"""
    if not headers:
        return "(empty result)"

    lines = []
    # 表头
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    # 分隔
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # 数据行
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    if len(rows) > max_rows:
        lines.append(f"... (共 {len(rows)} 行，仅显示前 {max_rows} 行)")

    return "\n".join(lines)


@tool
async def execute_sqlite(sql_query: str, table_name: str = "") -> str:
    """在已创建的 SQLite 内存临时表上执行 SELECT 查询。

    安全约束：仅允许 SELECT 查询。
    **必须在 create_table_from_markdown 成功创建表之后才能调用。**

    Args:
        sql_query:  一条合法的 SQLite SELECT 查询语句
        table_name: 要查询的表名（由 create_table_from_markdown 返回）
                    如果不指定，会在所有已创建的表中查找引用的表名

    Returns:
        JSON，包含查询结果（Markdown 表格格式）
    """
    # ---------- 安全检查 ----------
    upper_sql = sql_query.upper()
    for keyword in _FORBIDDEN_SQLITE_KEYWORDS:
        # 使用单词边界匹配避免误杀（如 "update_time" 列名不会触发）
        if re.search(r'\b' + keyword + r'\b', upper_sql):
            return json.dumps({
                "success": False,
                "error": f"禁止的 SQL 关键词: {keyword}。仅允许 SELECT 查询。"
            }, ensure_ascii=False)

    # ---------- 确定查询目标表 ----------
    target_table = table_name

    if not target_table:
        # 尝试从 SQL 语句中自动识别表名
        # 匹配 FROM "table_name" 或 FROM table_name
        from_match = re.search(
            r'FROM\s+"?([a-zA-Z_一-鿿][\w一-鿿]*)"?',
            sql_query, re.IGNORECASE
        )
        if from_match:
            target_table = from_match.group(1)
        else:
            return json.dumps({
                "success": False,
                "error": "无法从 SQL 中自动识别表名，请显式指定 table_name 参数。"
            }, ensure_ascii=False)

    if not _temp_manager.has_table(target_table):
        available = list(_temp_manager._tables.keys())
        return json.dumps({
            "success": False,
            "error": f"表 '{target_table}' 不存在。当前可用的表: {available}。请先调用 create_table_from_markdown 创建表。"
        }, ensure_ascii=False)

    # ---------- 执行 ----------
    try:
        headers, rows = _temp_manager.execute(target_table, sql_query)

        if headers is None:
            return json.dumps({
                "success": False,
                "error": "SQL 执行失败：未返回列头信息，请检查 SQL 语法。"
            }, ensure_ascii=False)

        markdown = _table_to_markdown(headers, rows)

        return json.dumps({
            "success": True,
            "headers": headers,
            "row_count": len(rows) if rows else 0,
            "result": markdown,
        }, ensure_ascii=False)

    except sqlite3.Error as e:
        return json.dumps({
            "success": False,
            "error": f"SQLite 执行错误: {str(e)}"
        }, ensure_ascii=False)


# ============================================================
# 辅助工具 4：查看已创建表的信息
# ============================================================

@tool
async def get_temp_table_info(table_name: str = "") -> str:
    """查看已创建的 SQLite 内存临时表的结构信息。

    适用场景：
    - 在写 SQL 之前确认表的列名和数据类型
    - 查看当前有哪些可用表

    Args:
        table_name: 表名。如果为空，则列出所有已创建的表名。
    """
    if not table_name:
        all_tables = list(_temp_manager._tables.keys())
        if not all_tables:
            return json.dumps({
                "success": True,
                "tables": [],
                "message": "当前没有已创建的临时表。请先调用 create_table_from_markdown。"
            }, ensure_ascii=False)
        return json.dumps({
            "success": True,
            "tables": all_tables,
            "message": f"共 {len(all_tables)} 个可用表。"
        }, ensure_ascii=False)

    schema = _temp_manager.get_table_schema(table_name)
    if schema is None:
        available = list(_temp_manager._tables.keys())
        return json.dumps({
            "success": False,
            "error": f"表 '{table_name}' 不存在。当前可用的表: {available}"
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "table_name": table_name,
        "schema": schema,
    }, ensure_ascii=False)


# ============================================================
# 本模块的工具汇总
# ============================================================

SQLITE_TOOLS: List = [
    create_table_from_markdown,
    generate_sqlite_sql,
    execute_sqlite,
    get_temp_table_info,
]
