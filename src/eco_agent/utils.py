"""Utility & helper functions."""

import json
import os
import pymysql
from pymysql import DatabaseError

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
# JSON_PATH="table_enriched_result.json"

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        All providers use OpenAI-compatible API (ChatOpenAI).
    """
    _provider, model = fully_specified_name.split("/", maxsplit=1)
    return ChatOpenAI(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )



class QueryExecutor:
    def __init__(self, host='10.10.1.205', database='test_db_refactored', user='root', password='123456', port=4406):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.conn = None
        self.cur = None

        # 👉 提前加载 mapping（避免重复 IO）
        # self.table_mapping = self._load_table_enriched_mapping(JSON_PATH)
    def connect(self):

        try:
            self.conn = pymysql.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                charset="utf8mb4"
            )

            self.cur = self.conn.cursor()

        except DatabaseError as e:
            print(f"Database connection error: {e}")
            raise

    def execute_sql(self, sql_statement):

        if not self.conn or not self.cur:
            self.connect()

        try:
            self.cur.execute(sql_statement)

            result = self.cur.fetchall()

            headers = [desc[0] for desc in self.cur.description]

            self.conn.commit()

            return headers, result

        except Exception as e:
            print(f"Error executing SQL statement: {e}")

            return None, None

    def get_table_data(self, table_name, limit=None):
        """
        查询指定表的数据
        :param table_name: 表名
        :param limit: 限制返回行数
        :return: headers, rows
        """

        if limit:
            sql = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        else:
            sql = f"SELECT * FROM `{table_name}`"

        return self.execute_sql(sql)

    def table_to_markdown(self, headers, rows, max_rows=None):
        """
        将表格数据转换为 Markdown 格式

        :param headers: 列名
        :param rows: 数据行
        :param max_rows: 最多展示多少行
        :return: markdown字符串
        """

        if headers is None or rows is None:
            return "No data."

        markdown = ""

        # 表头
        markdown += "| " + " | ".join(map(str, headers)) + " |\n"

        # 分隔行
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        # 数据行
        show_rows = rows[:max_rows] if max_rows else rows

        for row in show_rows:
            markdown += "| " + " | ".join(map(str, row)) + " |\n"

        return markdown
    

    # def _load_table_enriched_mapping(self, json_path):
    #     """只加载一次"""

    #     base_dir = os.path.dirname(os.path.abspath(__file__))
    #     json_path = os.path.join(base_dir, JSON_PATH)


    #     if not os.path.exists(json_path):
    #         print("没找到文件")
    #         return {}

    #     with open(json_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)

    #     mapping = {}

    #     for item in data:
    #         table_name = item.get("table_name")
    #         col_map = item.get("llm_refactored_columns", {})

    #         if table_name:
    #             mapping[table_name] = col_map

    #     return mapping
    
    # def replace_table_columns(self, headers, table_name):

    #     if table_name not in self.table_mapping:
    #         return headers

    #     col_map = self.table_mapping[table_name]

    #     return [
    #         col_map.get(col, col)
    #         for col in headers
    #     ]

    # =========================
    # ENRICHED QUERY
    # =========================
    # def get_table_data_with_enrichment(self, table_name, limit=None):

    #     headers, rows = self.get_table_data(table_name, limit)

    #     if headers is None:
    #         return None, None

    #     new_headers = self.replace_table_columns(headers, table_name)

    #     return new_headers, rows
    

    def close(self):

        if self.cur:
            self.cur.close()

        if self.conn:
            self.conn.close()