"""
序号4
将表格数据配合元信息，制作成一个个经济数据块，准备存入meilisearch搜索引擎
"""

import json
import re
import os
import pymysql
from typing import Dict, List, Any, Optional

# ================= 配置区域 =================
# 数据库配置
DB_CONFIG = {
    "host": "10.10.1.205",
    "user": "root",
    "password": "123456",
    "database": "test_db",
    "charset": "utf8mb4",
    "port": 4406
}

# 路径配置
NER_RESULTS_DIR = "ner_results"
TABLE_METADATA_PATH = "table_enriched_result.json"
OUTPUT_FILE = "knowledge_base_index_optimized.jsonl"


# ================= 工具函数 =================

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def extract_time_from_text(text: str) -> Optional[str]:
    """正则提取时间"""
    if not text:
        return None
    pattern = r"(\d{4}年\d{1,2}月|\d{4}年|\d{4}-\d{4}年?|\d{4}[-/.]\d{1,2}|\d{4})"
    matches = re.findall(pattern, text)
    if matches:
        return max(matches, key=len)
    return None


def clean_supplementary_info(original_text: str, locations: List[str], time_str: str) -> str:
    """提取补充实体信息"""
    if not original_text:
        return ""
    text = original_text
    sorted_locs = sorted(locations, key=len, reverse=True)
    for loc in sorted_locs:
        text = text.replace(loc, "")
    if time_str:
        text = text.replace(time_str, "")
    text = re.sub(r"^[、，, \t\r\n]+|[、，, \t\r\n]+$", "", text)
    return text.strip()


def is_valid_value(val: Any) -> bool:
    """
    【极简暴力版】判断值是否有效
    策略：只要内容中不包含任何阿拉伯数字 (0-9)，就视为无效数据。
    效果：
    - "123" -> True
    - "44.99" -> True
    - "2020年" -> True (含数字)
    - "*" -> False (无数字)
    - "-" -> False
    - "b'*'" -> False
    - "N/A" -> False
    """
    if val is None:
        return False

    # 转为字符串
    s_val = str(val)

    # 正则查找：只要有任意一个数字 \d，就保留；否则丢弃
    if re.search(r'\d', s_val):
        return True

    return False


# ================= 核心逻辑 =================

def build_knowledge_base():
    print("正在加载表格元信息...")
    if not os.path.exists(TABLE_METADATA_PATH):
        print(f"错误：找不到元信息文件 {TABLE_METADATA_PATH}")
        return

    with open(TABLE_METADATA_PATH, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    metadata_map = {item['table_name']: item for item in all_metadata}

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        print("数据库连接成功。")
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return

    ner_files = [f for f in os.listdir(NER_RESULTS_DIR) if f.endswith('_ner_result.json')]
    print(f"共发现 {len(ner_files)} 个表格待处理。")

    processed_count = 0
    skipped_rows_count = 0

    for ner_file in ner_files:
        try:
            file_path = os.path.join(NER_RESULTS_DIR, ner_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                ner_data = json.load(f)

            table_name = ner_data['table_name']
            if table_name not in metadata_map: continue
            meta = metadata_map[table_name]

            data_cols_info = meta.get('data_columns', [])
            if not data_cols_info: continue

            db_col_names = [col[0] for col in data_cols_info]
            refactored_cols = meta.get('llm_refactored_columns', {})

            cols_sql = ",".join([f"`{col}`" for col in db_col_names])
            cursor.execute(f"SELECT {cols_sql} FROM `{table_name}`")
            db_rows = cursor.fetchall()

            if len(db_rows) != len(ner_data['ner_results']): continue

            batch_results = []

            for row_idx, ner_row in enumerate(ner_data['ner_results']):
                original_text = ner_row.get('original_text', '')
                entities = ner_row.get('entities', [])
                db_values = db_rows[row_idx]

                row_data_values = {}
                attributes_list = []  # 这个列表将用于存储这一行的所有 Key

                for col_i, col_key in enumerate(db_col_names):
                    val = db_values[col_i]

                    # 使用强力过滤 (必须包含数字)
                    if is_valid_value(val):
                        attr_name = refactored_cols.get(col_key, col_key)

                        val_str = str(val).strip()
                        if val_str.startswith("b'") and val_str.endswith("'"):
                            val_str = val_str[2:-1]

                        row_data_values[attr_name] = val_str

                        # 【重点】把有效的列名(Key)加入列表
                        attributes_list.append(attr_name)

                if not row_data_values:
                    skipped_rows_count += 1
                    continue

                # --- 索引构建 ---
                found_locations = []
                for ent in entities:
                    if ent['type'] in ['LOCATION', 'PERSON']:
                        found_locations.append(ent['text'])
                found_locations = list(dict.fromkeys(found_locations))
                location_idx = " ".join(found_locations) if found_locations else meta.get('location', '未知地区')

                extracted_time = extract_time_from_text(original_text)
                time_idx = extracted_time if extracted_time else meta.get('time_period', '未知时间')

                supp_idx = clean_supplementary_info(original_text, found_locations, extracted_time)
                if not supp_idx: supp_idx = None

                # 构建全文搜索字段
                # 注意：attributes_list 里的内容 (比如 "GRP", "GDP") 也会被拼接到这里，供全文检索
                full_text = f"{location_idx} {time_idx} {supp_idx if supp_idx else ''} {' '.join(attributes_list)} {meta.get('table_description', '')}"

                entry = {
                    "id": f"{meta['table_id']}_{row_idx}",
                    "meta": {
                        "table_id": meta['table_id'],
                        "table_name": meta['table_description'],
                        "original_text": original_text,
                    },
                    "index": {
                        "location": location_idx,
                        "time": time_idx,
                        "entity_supp": supp_idx,
                        # 【关键】新增字段：显式存储这一行有哪些数据列名
                        # 这样 Meilisearch 就可以通过 filter="index.attribute_names = 'GRP'" 来搜索了
                        "attribute_names": attributes_list
                    },
                    "data": row_data_values,
                    "full_text_search": full_text
                }

                batch_results.append(entry)

            if batch_results:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
                    for entry in batch_results:
                        out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"已处理 {processed_count} 个表格 (累计跳过无数据行: {skipped_rows_count})...")

        except Exception as e:
            print(f"处理文件 {ner_file} 异常: {e}")

    cursor.close()
    conn.close()
    print(f"处理完成！累计跳过无数据行: {skipped_rows_count}")
    print(f"结果已保存至: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    build_knowledge_base()