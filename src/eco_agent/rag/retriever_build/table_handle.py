"""
    序号3
    使用大模型提取表格元信息中的时间空间信息
"""

import json
import os
import time
from typing import List, Dict
from tqdm import tqdm

# LangChain & Pydantic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# -------------------------- 配置区域 --------------------------

INPUT_FILE = 'table_analysis_result.json'
OUTPUT_FILE = 'table_enriched_result.json'
BATCH_SIZE = 8  # 建议设置为 5-10，取决于表格描述的长度，过大可能导致 Output Token 超限

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0,
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-ClU92L9UXW9Nt8jQkyoGAZPyo6AnHMd6QMQ6k9uuGDcHIrk6"
)


# -------------------------- 定义输出结构 --------------------------

# 单个表格的处理结果结构
class SingleTableResult(BaseModel):
    table_id: str = Field(description="必须与输入信息中的 table_id 保持完全一致，用于数据对齐")
    location: str = Field(description="地理位置，如'中国'、'美国'、'全球'等")
    time_period: str = Field(description="时间段，如'2023年'、'2015-2020年'等")
    refactored_columns: Dict[str, str] = Field(
        description="键为原英文列名，值为重构后的中文列名。新列名需结合表信息，使其独立、完整、自解释。")


# 批量返回的结构
class BatchResponse(BaseModel):
    results: List[SingleTableResult] = Field(description="包含本批次所有表格处理结果的列表")


# 初始化解析器
parser = JsonOutputParser(pydantic_object=BatchResponse)

# -------------------------- 定义 Prompt --------------------------

batch_prompt_template = """
你是一位专业的宏观经济数据分析师。我将一次性提供多个表格的元数据，请你批量处理它们。

【任务目标】
对于每一个表格，请执行以下操作：
1. **分析元数据**：根据表名和描述，提取数据的**地理位置 (location)** 和 **时间周期 (time_period)**。
2. **重构列名**：
   - 原始列名（如“数值”、“同比增长”）脱离表名后无法独立理解。
   - 请结合表名和描述，将每个数据列的中文名重构为**完整、独立、自解释**的名称（包含主体+属性+单位）。
   - 例如：将“数值”重构为“2023年中国GDP绝对值(亿元)”。

【输入数据列表】
{batch_input_text}

【输出要求】
1. 请返回一个 JSON 对象，其中包含一个 `results` 列表。
2. 列表中的每个对象对应一个输入表格，**必须包含原始的 table_id** 以便系统匹配。
3. 请确保不要遗漏任何一个表格。

{format_instructions}
"""

prompt = PromptTemplate(
    template=batch_prompt_template,
    input_variables=["batch_input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser


# -------------------------- 辅助函数 --------------------------

def format_batch_input(tables: List[dict]) -> str:
    """将多个表格对象格式化为一个清晰的文本块供 LLM 阅读"""
    text_parts = []
    for i, tbl in enumerate(tables):
        # 格式化列信息
        cols = tbl.get('data_columns', [])
        # 为了节省 Token，只取前 20 个列（如果列非常多）
        cols_preview = cols[:20]
        cols_str = ", ".join([f"{c[0]}({c[1]})" for c in cols_preview])
        if len(cols) > 20: cols_str += "..."

        # 截断过长描述
        desc = tbl.get('table_description', '')
        if len(desc) > 300: desc = desc[:300] + "..."

        item_text = (
            f"--- 表格 {i + 1} ---\n"
            f"ID: {tbl.get('table_id')}\n"
            f"Name: {tbl.get('table_name')}\n"
            f"Desc: {desc}\n"
            f"Columns: {cols_str}\n"
        )
        text_parts.append(item_text)
    return "\n".join(text_parts)


# -------------------------- 主逻辑 --------------------------

def process_batch_enrichment():
    # 1. 读取源数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误：未找到输入文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # 2. 读取/初始化结果数据（支持断点续传）
    final_results_map = {}  # 使用 ID 为 Key 的字典，方便去重和更新

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_list = json.load(f)
                for item in existing_list:
                    final_results_map[item['table_id']] = item
            print(f"已加载历史记录，共 {len(final_results_map)} 条。")
        except:
            print("历史文件读取失败，重新开始。")

    # 3. 筛选未处理的表格
    pending_tables = [t for t in source_data if t['table_id'] not in final_results_map]
    print(f"剩余待处理任务数: {len(pending_tables)} (总数 {len(source_data)})")

    if not pending_tables:
        print("所有表格已处理完毕！")
        return

    # 4. 分批处理
    # 将列表切分为多个 Batch
    batches = [pending_tables[i:i + BATCH_SIZE] for i in range(0, len(pending_tables), BATCH_SIZE)]

    print(f"即将开始处理，共 {len(batches)} 个批次，每批次 {BATCH_SIZE} 表...")

    try:
        for batch_idx, batch_tables in enumerate(tqdm(batches, desc="Batch Processing")):

            # 构造输入文本
            batch_text = format_batch_input(batch_tables)

            try:
                # 调用 LLM
                response = chain.invoke({"batch_input_text": batch_text})

                # 解析结果列表
                results_list = response.get('results', [])

                # 建立当前批次的 ID 映射，方便快速查找源数据
                batch_source_map = {t['table_id']: t for t in batch_tables}

                # 合并结果
                for res_item in results_list:
                    tid = res_item.get('table_id')

                    # 找到对应的原始数据
                    if tid in batch_source_map:
                        original_record = batch_source_map[tid]
                        # 创建新记录：原始数据 + LLM 提取数据
                        enriched_record = original_record.copy()
                        enriched_record['location'] = res_item.get('location', '')
                        enriched_record['time_period'] = res_item.get('time_period', '')
                        enriched_record['llm_refactored_columns'] = res_item.get('refactored_columns', {})

                        # 存入总结果字典
                        final_results_map[tid] = enriched_record
                    else:
                        print(f"\n[Warning] LLM 返回了一个未知的 table_id: {tid}")

            except Exception as e:
                print(f"\n[Error] 第 {batch_idx + 1} 批次处理失败: {e}")
                # 简单的错误处理：如果是网络波动，这里直接跳过，下次运行代码时会重新抓取这些 pending 的表
                time.sleep(2)

                # 每处理完一个 Batch，保存一次文件
            # 将字典转回列表保存
            current_output_list = list(final_results_map.values())
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(current_output_list, f, ensure_ascii=False, indent=4)

            # 稍微停顿，防止触发 API 速率限制 (Rate Limit)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n用户中断，已保存当前进度。")
    finally:
        # 最终保存
        final_list = list(final_results_map.values())
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)
        print(f"\n全部完成！结果已保存至 {OUTPUT_FILE}。")


if __name__ == "__main__":
    process_batch_enrichment()