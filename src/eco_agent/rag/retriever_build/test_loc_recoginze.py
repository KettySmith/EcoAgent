"""
    序号2
    hanlp识别每行数据是否属于 时间/地点信息
    该程序存在的意义是 补充表格元信息中不存在的时空信息
"""
import json
import os
import pymysql
from tqdm import tqdm
from pymysql.cursors import Cursor
import hanlp

# 数据库配置
db_config = {
    "host": "10.10.1.205",
    "user": "root",
    "password": "123456",
    "database": "test_db",
    "charset": "utf8mb4",
    "port": 4406
}

# 创建存储结果的目录
output_dir = "ner_results"
os.makedirs(output_dir, exist_ok=True)

print("正在连接数据库...")
try:
    conn = pymysql.connect(**db_config)
    cursor: Cursor = conn.cursor()
    print("数据库连接成功！")
except Exception as e:
    print(f"数据库连接失败: {e}")
    exit()

# 关键优化1：只加载一次模型（放在循环外）
print("正在加载HanLP模型...")
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
print("模型加载完成！")

# 读取分析结果
with open("table_enriched_result.json", "r", encoding="utf-8") as f:
    table_analysis_result = json.load(f)

# 处理每个表
for table in tqdm(table_analysis_result, desc="处理表格"):
    table_name = table["table_name"]

    # 如果没有实体列，跳过
    if len(table["entity_columns"]) == 0:
        print(f"跳过 {table_name}：无实体列")
        continue

    # 关键优化2：简化字段名拼接
    # 构建类似 `column1`,`column2`,`column3` 的字符串
    columns = [f"`{col[0]}`" for col in table["entity_columns"]]
    table_col = ",".join(columns)

    # 查询数据
    try:
        cursor.execute(f"SELECT {table_col} FROM `{table_name}`")
        entities = cursor.fetchall()
    except Exception as e:
        print(f"查询表 {table_name} 失败: {e}")
        continue

    # 存储本表的所有结果
    table_results = {
        "table_name": table_name,
        "entity_columns": table["entity_columns"],
        "total_records": len(entities),
        "ner_results": []
    }

    # 处理每条记录
    for entity in entities:
        # 将元组转换为用逗号拼接的字符串
        entity_text = ','.join(map(str, entity))

        # 进行NER识别
        try:
            result_msra = HanLP(entity_text, tasks='ner/msra')

            # 提取结构化NER信息
            ner_info = {
                "original_text": entity_text,
                "entities": []
            }

            # 解析NER结果 - 根据实际格式调整（四元组格式）
            if 'ner/msra' in result_msra:
                ner_data = result_msra['ner/msra']

                if isinstance(ner_data, list):
                    for entity_item in ner_data:
                        try:
                            # 处理四元组格式: (entity_text, entity_type, start, end)
                            if isinstance(entity_item, (list, tuple)) and len(entity_item) >= 4:
                                entity_text_segment, label, start, end = entity_item[0], entity_item[1], entity_item[2], \
                                entity_item[3]

                                # 验证位置信息是否有效
                                start = int(start)
                                end = int(end)

                                # 如果位置信息异常，尝试从原始文本中定位
                                if start < 0 or end > len(entity_text) or start >= end:
                                    # 尝试从文本中查找实体
                                    found_pos = entity_text.find(str(entity_text_segment))
                                    if found_pos >= 0:
                                        start = found_pos
                                        end = found_pos + len(str(entity_text_segment))
                                    else:
                                        # 无法定位，跳过该实体
                                        continue

                                ner_info["entities"].append({
                                    "text": str(entity_text_segment),
                                    "type": str(label),
                                    "start": start,
                                    "end": end
                                })
                            else:
                                # 处理其他可能的格式
                                print(f"警告: 未知实体格式: {entity_item}")

                        except Exception as unpack_error:
                            print(f"解析实体项失败: {unpack_error}, 原始数据: {entity_item}")
                            # 跳过无法解析的实体

            table_results["ner_results"].append(ner_info)

        except Exception as e:
            print(f"处理记录失败（表{table_name}）: {e}")
            # 即使失败也记录原始文本
            table_results["ner_results"].append({
                "original_text": entity_text,
                "entities": [],
                "error": str(e)
            })

    # 关键优化3：每个表保存为单独的JSON文件
    output_file = os.path.join(output_dir, f"{table_name}_ner_result.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(table_results, f, ensure_ascii=False, indent=2)
        print(f"已保存: {output_file}")
    except Exception as e:
        print(f"保存文件 {output_file} 失败: {e}")

# 关闭连接
cursor.close()
conn.close()
print("处理完成！")