"""
序号5，将jsonl存入搜索引擎
"""
import meilisearch
import json
import os
import time

# ================= 配置区域 =================
MEILI_HOST = 'http://10.10.1.217:7700/'
MEILI_KEY = 'FEFIR3VpjrhNrc9gPXklB_jv6keqz7mnzciKZ8_2ClE'  # 如果没有key，留空字符串 ''
INDEX_NAME = 'stats_kb'  # 索引名称
DATA_FILE = 'knowledge_base_index_optimized.jsonl'  # 上一步生成的文件名
BATCH_SIZE = 2000  # 每次上传文档数量，防止请求体过大


def setup_meilisearch():
    print(f"🔄 正在连接 Meilisearch: {MEILI_HOST} ...")
    try:
        client = meilisearch.Client(MEILI_HOST, MEILI_KEY)
        client.health()  # 检查健康状态
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 1. 创建或获取索引
    # get_or_create_index 在新版SDK可能稍有不同，这里用标准流程
    try:
        index = client.get_index(INDEX_NAME)
        print(f"✅ 索引 '{INDEX_NAME}' 已存在。")
    except:
        print(f"🆕 索引 '{INDEX_NAME}' 不存在，正在创建...")
        task = client.create_index(INDEX_NAME, {'primaryKey': 'id'})
        client.wait_for_task(task.task_uid)
        index = client.get_index(INDEX_NAME)

    # ================= 2. 核心配置 (Settings) =================
    print("⚙️ 正在更新索引配置，以支持字段级模糊搜索...")

    settings = {
        # A. 可搜索属性 (Searchable Attributes) - 决定了搜索算法的权重
        # 顺序极其重要：越靠前，匹配得分越高。
        # 这实现了你的需求：优先匹配指标名和地区，支持模糊搜索算法。
        'searchableAttributes': [
            'index.attribute_names',  # Top 1: 如果搜 "GRP"，优先匹配这里的 GRP
            'index.location',  # Top 2: 如果搜 "天津"，优先匹配这里的 天津
            'index.time',  # Top 3: 时间
            'index.entity_supp',  # Top 4: 补充实体 (如 "基础研究")
            'full_text_search',  # Fallback: 如果上面都没对上，再搜全文拼接串
            'meta.table_name'  # 最后: 表名
        ],

        # B. 可过滤属性 (Filterable Attributes) - 允许使用 filter 进行精确筛选
        # 智能体既可以用 q 模糊搜，也可以用 filter 精确锁死
        'filterableAttributes': [
            'index.location',
            'index.time',
            'index.attribute_names',  # 让智能体可以 filter="index.attribute_names = 'GRP'"
            'index.entity_supp',
            'meta.table_id'
        ],

        # C. 停用词 (Stop Words) - 过滤掉对统计搜索无意义的虚词
        'stopWords': [
            "的", "是", "多少", "什么", "在", "数据", "统计", "表", "一览", "情况"
        ],

        # D. 容错设置 (Typo Tolerance) - 默认开启，这里显式列出以供你参考
        # 你可以微调，比如禁止在这个字段上容错
        'typoTolerance': {
            'enabled': True,
            'minWordSizeForTypos': {
                'oneTypo': 5,  # 单词长度>=5才允许1个错字 (针对英文/拼音)
                'twoTypos': 9
            },
            'disableOnAttributes': []  # 如果不想让时间模糊，可以填 ['index.time']
        },

        # E. 分页最大限制 (避免深度分页性能问题)
        'pagination': {
            'maxTotalHits': 10000
        }
    }

    task = index.update_settings(settings)
    print(f"🚀 配置任务已提交 (Task UID: {task.task_uid})，正在等待生效...")
    client.wait_for_task(task.task_uid)
    print("✅ 索引配置更新完成！")

    # ================= 3. 数据上传 (Batch Upload) =================
    if not os.path.exists(DATA_FILE):
        print(f"❌ 错误：找不到数据文件 {DATA_FILE}")
        return

    print(f"📤 开始读取并上传数据: {DATA_FILE}")

    documents = []
    total_count = 0
    batch_count = 0

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue

            doc = json.loads(line)
            documents.append(doc)

            # 达到批次大小时上传
            if len(documents) >= BATCH_SIZE:
                index.add_documents(documents)
                total_count += len(documents)
                batch_count += 1
                print(f"   - 第 {batch_count} 批已提交 ({len(documents)} 条)...")
                documents = []  # 清空缓冲

        # 上传剩余的文档
        if documents:
            index.add_documents(documents)
            total_count += len(documents)
            print(f"   - 最后一批已提交 ({len(documents)} 条)...")

    print(f"🎉 全部上传完成！共上传 {total_count} 条文档。")
    print("索引正在后台构建中，请稍等几秒后尝试搜索。")


if __name__ == "__main__":
    setup_meilisearch()