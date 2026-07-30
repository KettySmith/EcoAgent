import json
import pymysql
from pymysql.cursors import DictCursor

# =========================
# 配置
# =========================
JSON_PATH = "/home/zhangruochi/eco_model/model/learning/react-agent/src/eco_agent/table_enriched_result_humanfixed.json"   # 你的json文件
SOURCE_DB = "test_db"           # 原数据库
TARGET_DB = "test_db_refactored"  # 新schema

MYSQL_CONFIG = {
    "host": "10.10.1.205",
    "port": 4406,
    "user": "root",
    "password": "123456",
    "charset": "utf8mb4",
    "cursorclass": DictCursor
}


# =========================
# 删除数据库函数
# =========================
def drop_database(cursor, db_name):
    """
    删除数据库（如果存在）
    """
    print(f"\n开始删除数据库: {db_name}")

    sql = f"DROP DATABASE IF EXISTS `{db_name}`;"

    cursor.execute(sql)

    print(f"数据库删除完成: {db_name}")


# =========================
# 读取JSON
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    tables_meta = json.load(f)

# =========================
# 连接数据库
# =========================
conn = pymysql.connect(**MYSQL_CONFIG)

# 注意这里使用 DictCursor
cursor = conn.cursor(DictCursor)

try:

    # =========================
    # 删除旧数据库
    # =========================
    drop_database(cursor, TARGET_DB)

    # =========================
    # 创建新的 schema
    # =========================
    cursor.execute(
        f"""
        CREATE DATABASE `{TARGET_DB}`
        DEFAULT CHARACTER SET utf8mb4;
        """
    )

    print(f"\n创建数据库完成: {TARGET_DB}")

    # =========================
    # 处理每张表
    # =========================
    for item in tables_meta:

        table_name = item["table_name"]

        # key: 原列名
        # value: 新列名
        rename_mapping = item.get("llm_refactored_columns", {})

        print(f"\n开始处理表: {table_name}")

        try:

            # -----------------------------------
            # 1. 复制原表结构 + 数据
            # -----------------------------------
            create_sql = f"""
            CREATE TABLE `{TARGET_DB}`.`{table_name}`
            AS
            SELECT *
            FROM `{SOURCE_DB}`.`{table_name}`;
            """

            cursor.execute(create_sql)

            # -----------------------------------
            # 2. 获取新表字段信息
            # -----------------------------------
            cursor.execute(f"""
                SHOW COLUMNS
                FROM `{TARGET_DB}`.`{table_name}`
            """)

            columns_info = cursor.fetchall()

            column_type_map = {}

            for col in columns_info:
                column_type_map[col["Field"]] = col["Type"]

            # -----------------------------------
            # 3. 修改列名
            # -----------------------------------
            for old_col, new_col in rename_mapping.items():

                if old_col not in column_type_map:
                    print(f"列不存在，跳过: {old_col}")
                    continue

                col_type = column_type_map[old_col]

                alter_sql = f"""
                ALTER TABLE `{TARGET_DB}`.`{table_name}`
                CHANGE COLUMN `{old_col}` `{new_col}` {col_type};
                """

                print(f"重命名列: {old_col} -> {new_col}")

                cursor.execute(alter_sql)

            conn.commit()

            print(f"表处理完成: {table_name}")

        except Exception as e:

            conn.rollback()

            print(f"处理失败: {table_name}")
            print(e)

finally:

    # =========================
    # 关闭连接
    # =========================
    cursor.close()
    conn.close()

print("\n全部完成")