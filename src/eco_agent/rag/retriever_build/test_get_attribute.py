"""
    序号 1
    从表格提取元信息并存入json
    识别表格列类型 实体列 / 数据列
"""
import json
import os
import csv
import re
import pymysql
from tqdm import tqdm
from pymysql.cursors import Cursor


# -------------------------- 核心函数 --------------------------

def identify_table_columns(cursor, table_name="Unknown"):
    """
    识别表格列类型（实体列 vs 数据列）
    包含全空列过滤和 Debug 采样功能
    """
    # 1. 正则定义
    year_col_pattern = re.compile(r'year', re.IGNORECASE)
    year_extract_pattern = re.compile(r'(19\d{2}|20\d{2})')
    year_format_pattern = re.compile(r'^(19\d{2}|20\d{2})(年|[-/.]\d{1,2})?([-/.]\d{1,2}|年\d{1,2}月)?$')

    numeric_pattern = re.compile(r'^[-+]?\d+(,\d{3})*(\.\d+)?([eE][-+]?\d+)?$')
    entity_char_pattern = re.compile(r'.*[\u4e00-\u9fa5a-zA-Z①②③()（）]+.*')

    # 无效字符正则
    invalid_pattern = re.compile(r'^[*#NAna/—空\s-]+$')

    # 2. 获取元数据
    try:
        column_names = [desc[0] for desc in cursor.description]
    except AttributeError:
        return [], []

    rows = cursor.fetchall()
    if not rows:
        return [], []

    entity_columns = []
    data_columns = []

    # Debug 采样
    debug_samples = {name: [] for name in column_names}

    # 3. 逐列分析
    for col_idx, col_name in enumerate(column_names):
        data_count = 0
        entity_count = 0
        total_valid = 0
        is_year_column = bool(year_col_pattern.search(col_name))

        for row_idx, row in enumerate(rows):
            val = row[col_idx]

            # --- 数据清洗 ---
            if val is None:
                val_str = ""
            elif isinstance(val, bytes):
                try:
                    val_str = val.decode('utf-8').strip()
                except:
                    val_str = str(val).strip()
            else:
                val_str = str(val).strip()

            val_str = val_str.replace('\ufeff', '').replace('\u200b', '').replace('\xa0', '')

            # --- Debug 采样逻辑 ---
            should_log = len(debug_samples[col_name]) < 3
            if not val_str:
                if should_log:
                    debug_samples[col_name].append({"val": str(val), "clean": "", "status": "Skipped(Empty)"})
                continue

            if invalid_pattern.match(val_str):
                if should_log:
                    debug_samples[col_name].append(
                        {"val": str(val), "clean": val_str, "status": "Skipped(InvalidPattern)"})
                continue

            total_valid += 1

            # --- 判定逻辑 ---
            match_type = "Unknown"
            if is_year_column and year_format_pattern.match(val_str):
                if year_extract_pattern.search(val_str):
                    entity_count += 1
                    match_type = "Year-Entity"
            elif entity_char_pattern.match(val_str):
                entity_count += 1
                match_type = "Char-Entity"
            elif numeric_pattern.match(val_str):
                data_count += 1
                match_type = "Numeric-Data"
            else:
                entity_count += 1
                match_type = "Fallback-Entity"

            if should_log:
                try:
                    hex_val = val_str.encode('utf-8').hex()
                except:
                    hex_val = "Error"
                debug_samples[col_name].append({
                    "val": str(val)[:20],
                    "clean": val_str,
                    "hex": hex_val,
                    "status": f"Valid({match_type})"
                })

        # --- 列归类 ---
        if total_valid == 0:
            pass  # 丢弃全空/无效列
        else:
            data_ratio = data_count / total_valid
            if data_ratio >= 0.5:
                data_columns.append(col_name)
            else:
                entity_columns.append(col_name)

    # 4. Debug 输出 (仅当无数据列且非空表时)
    if len(data_columns) == 0 and len(column_names) > 0:
        print(f"\n{'=' * 20} DEBUG: 表 {table_name} 无数据列 {'=' * 20}")
        # 这里为了不刷屏，仅在真正调试时可以取消注释
        # for col in column_names:
        #     ... (保持原有的打印逻辑)

    return entity_columns, data_columns


# -------------------------- 执行流程 --------------------------

# 1. 数据库配置
db_config = {
    "host": "10.10.1.205",
    "user": "root",
    "password": "123456",
    "database": "test_db",
    "charset": "utf8mb4",
    "port": 4406
}

print("正在连接数据库...")
try:
    conn = pymysql.connect(**db_config)
    cursor: Cursor = conn.cursor()
    print("数据库连接成功！")
except Exception as e:
    print(f"数据库连接失败: {e}")
    exit()

# 2. 优先加载字段元数据 (Map: 英文列名 -> 中文列名)
field_hash = {}
print("\n正在加载字段元数据...")
try:
    # 假设表结构：... | ... | english_name | chinese_desc | ...
    cursor.execute("select * from table_field_meta")
    table_field_data = cursor.fetchall()
    for table_field in tqdm(table_field_data, desc="构建字段映射"):
        # 注意：需确保索引与数据库实际结构一致
        # table_field[2] 是英文名, table_field[3] 是中文名
        eng_name = table_field[2]
        chn_name = table_field[3]
        if eng_name:
            field_hash[eng_name] = chn_name
    print(f"成功加载 {len(field_hash)} 个字段的元数据！")
except Exception as e:
    print(f"加载字段元数据失败: {e}")
    # 不退出，继续运行，只是中文名会变为空

# 3. 处理 CSV 并生成 JSON
csv_path = 'test_table_meta.csv'
json_output_path = 'table_analysis_result.json'
json_results = []

if os.path.exists(csv_path):
    with open(csv_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # 跳过 CSV 表头

        rows = list(csv_reader)

        for row in tqdm(rows, desc="处理表格"):
            if len(row) < 2:
                continue

            # 读取 CSV 信息
            table_id = row[0]  # 第0列
            table_name = row[1]  # 第1列
            table_desc = row[3] if len(row) > 3 else ""  # 第3列 (防止越界)

            try:
                cursor.execute(f"select * from {table_name}")
                e_cols, d_cols = identify_table_columns(cursor, table_name)

                # 如果没有数据列，跳过该表
                if len(d_cols) == 0:
                    continue

                # --- 构建 [英文, 中文] 二元对 ---
                # 使用 get(col, "")，如果字典里没有就返回空字符串

                entity_col_pairs = [
                    [col, field_hash.get(col, "")]
                    for col in e_cols
                ]

                data_col_pairs = [
                    [col, field_hash.get(col, "")]
                    for col in d_cols
                ]

                # --- 组装结果对象 ---
                table_result = {
                    "table_id": table_id,
                    "table_name": table_name,
                    "table_description": table_desc,
                    "entity_columns": entity_col_pairs,
                    "data_columns": data_col_pairs
                }

                json_results.append(table_result)

                # print(f" [录入] {table_name}: Entity={len(e_cols)}, Data={len(d_cols)}")

            except Exception as e:
                print(f" [错误] 处理表 {table_name} 失败: {e}")

    # 4. 保存为 JSON 文件
    print(f"\n正在保存结果到 {json_output_path} ...")
    try:
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            # ensure_ascii=False 保证中文正常显示，indent=4 美化输出
            json.dump(json_results, json_file, ensure_ascii=False, indent=4)
        print(f"保存成功！共处理有效表格 {len(json_results)} 个。")
    except Exception as e:
        print(f"保存 JSON 失败: {e}")

else:
    print(f"未找到文件: {csv_path}")

cursor.close()
conn.close()