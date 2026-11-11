"""
序号3的单次版，我没试过批量，所以我不知道另一个序号3能否运行
我运行的其实是这个
识别元信息中的时间空间信息
"""
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# -------------------------- 配置区域 --------------------------

INPUT_FILE = 'table_analysis_result.json'
OUTPUT_FILE = 'table_enriched_result.json'

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-ClU92L9UXW9Nt8jQkyoGAZPyo6AnHMd6QMQ6k9uuGDcHIrk6"  # 请确保 Key 安全
)


# -------------------------- 定义输出结构 --------------------------

# 定义我们需要大模型返回的 JSON 数据结构，用于 LangChain 自动校验
class TableEnrichment(BaseModel):
    location: str = Field(description="该表格数据涉及的地理位置，如'中国'、'全球'、'美国'等")
    time_period: str = Field(description="该表格数据涉及的时间段")
    refactored_columns: Dict[str, str] = Field(
        description="键为原英文列名，值为重构后的中文列名。重构后的列名应结合表格上下文，使其在脱离表格描述时也能独立表达含义。")


# 初始化解析器
parser = JsonOutputParser(pydantic_object=TableEnrichment)

# -------------------------- 定义 Prompt --------------------------

prompt_template = """
你是一位专业的宏观经济数据分析师。请根据提供的表格元数据，提取关键上下文信息，并对数据列的中文名称进行语义重构。

【表格信息】
- 表名: {table_name}
- 描述: {table_description}
- 包含的数据列 (格式为 [英文列名, 原中文列名]): 
{data_columns_str}

【任务要求】
1. **提取地理位置 (location)**: 分析表名和描述，指明数据覆盖的区域（例如：中国、各省份、全球、OECD国家等）。
2. **提取时间周期 (time_period)**: 分析表名和描述，指明数据的时间属性，必须不能有错误，统一格式为每个具体的时间点，使我可以使用字符串进行匹配时可以检索到其具体的时间段,如果出现多个可能的时间你需要逐个将其逐个列出（例如：2022年）。
3. **重构数据列名 (refactored_columns)**: 
   - 原始的列名通常很简单（如"数值"、"同比"），脱离表名后无法理解。
   - 请结合表名和描述，赋予每个数据列一个**完整、独立、自解释**的新中文名称。
   - 新名称应包含：主体 + 属性 + 单位(如有)。
   - 格式要求：返回一个字典，Key是原始英文列名，Value是重构后的新中文名。

【输出格式】
请仅返回标准的 JSON 格式，不要包含任何 Markdown 标记或额外的解释。格式如下：
{format_instructions}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["table_name", "table_description", "data_columns_str"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser


# -------------------------- 处理逻辑 --------------------------

def process_table_enrichment():
    # 1. 读取输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：未找到输入文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # 2. 读取已处理的数据 (用于断点续传)
    processed_data = []
    processed_ids = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                # 记录已处理的 table_id，防止重复跑
                for item in processed_data:
                    processed_ids.add(item.get('table_id'))
            print(f"发现已处理记录 {len(processed_data)} 条，将跳过这些表格。")
        except json.JSONDecodeError:
            print("输出文件损坏或为空，将重新开始。")

    # 3. 循环处理
    print(f"开始处理，共 {len(source_data)} 个表格任务...")

    # 打开文件准备写入（每处理一条就更新一次文件，防止程序崩溃丢失数据）
    # 但由于 JSON 是整体结构，简单的 append 会破坏格式。
    # 策略：内存中更新 processed_data，每隔 N 条或出错时全量保存一次。

    try:
        for index, table in enumerate(tqdm(source_data, desc="LLM Processing")):
            table_id = table.get('table_id')

            # 跳过已处理
            if table_id in processed_ids:
                continue

            table_name = table.get('table_name', '')
            table_desc = table.get('table_description', '')
            # 截断过长的描述以节省 Token (题目要求描述本身可能不到100字，这里做个保险)
            if len(table_desc) > 500:
                table_desc = table_desc[:500] + "..."

            # 格式化列名信息给 Prompt
            # data_columns 结构是 [[en, cn], [en, cn]...]
            data_cols = table.get('data_columns', [])
            if not data_cols:
                continue

            data_columns_str = "\n".join([f"- {col[0]}: {col[1]}" for col in data_cols])

            # 调用 LLM
            try:
                # 使用 invoke 调用链
                result = chain.invoke({
                    "table_name": table_name,
                    "table_description": table_desc,
                    "data_columns_str": data_columns_str
                })

                # 将 LLM 结果合并回原对象
                # 深拷贝一份以免修改原始引用（虽然这里不需要）
                new_record = table.copy()
                new_record['location'] = result.get('location', '')
                new_record['time_period'] = result.get('time_period', '')
                new_record['llm_refactored_columns'] = result.get('refactored_columns', {})

                processed_data.append(new_record)

            except Exception as e:
                print(f"\n[Error] 处理表 {table_name} 时 LLM 报错: {e}")
                # 即使报错，也可以选择把原数据存进去，只是相关字段为空
                error_record = table.copy()
                error_record['error_log'] = str(e)
                processed_data.append(error_record)
                time.sleep(1)  # 避让一下 API 限制

            # 每处理 5 条保存一次，避免完全重跑
            if (index + 1) % 5 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=4)

    except KeyboardInterrupt:
        print("\n用户手动中断，正在保存当前进度...")
    finally:
        # 最后再一次性保存
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        print(f"\n处理完成！结果已保存至 {OUTPUT_FILE}，共包含 {len(processed_data)} 条数据。")


if __name__ == "__main__":
    process_table_enrichment()