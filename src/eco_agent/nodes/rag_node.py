from typing import List, Dict

from langchain_core.documents import Document
from langgraph.runtime import Runtime

from eco_agent.context import Context
from eco_agent.rag.pipeline import AdvancedRAGPipeline


# ============================================================
# 文本格式化工具函数
# ============================================================

def _format_table_docs(tables: List[Dict]) -> str:
    """将 Meilisearch 表格检索结果中的 data 转为 Markdown 表格"""
    if not tables:
        return ""

    parts: list[str] = []
    for t in tables:
        data = t.get("data", {})
        if not data:
            continue

        # 标题行: 表名 + 地点/时间
        meta = t.get("meta", {})
        index = t.get("index", {})
        title = meta.get("table_name", "")
        location = index.get("location", "")
        time = index.get("time", "")
        info = " | ".join(
            x for x in [f"地点: {location}", f"时间: {time}"] if x.split(": ")[1]
        )

        lines: list[str] = []
        lines.append(f"【{title}】{info}")

        # data dict → Markdown 表格（参考 table_to_markdown）
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        for attr, val in data.items():
            lines.append(f"| {attr} | {val} |")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _format_text_docs(docs: List[Document]) -> str:
    """将 Chroma 向量库 Document 列表转为纯文本"""
    if not docs:
        return ""

    return "\n".join(
        doc.page_content for doc in docs if doc.page_content
    )


def _format_tavily_docs(docs: List[Dict]) -> str:
    """将 Tavily 联网搜索结果转为纯文本"""
    if not docs:
        return ""

    parts: list[str] = []
    for d in docs:
        title = d.get("title", "")
        content = d.get("content", "")
        # source = d.get("source", "")
        parts.append(f"{title}\n{content}")

    return "\n".join(parts)


# ============================================================
# RAG Node
# ============================================================

async def rag_node(state, runtime: Runtime[Context]):

    question = state["question"]
    # 初始化 RAG pipeline
    rag_pipeline = AdvancedRAGPipeline(
        model_name=runtime.context.model,
        top_k=3
    )

    # 检索
    results = await rag_pipeline.retrieve(question)
    retrieved_origin_docs=_format_text_docs(results["docs"])
    retrieved_tavily_docs=_format_tavily_docs(results["tavily_docs"])
    retrieved_docs=retrieved_origin_docs+"\n"+retrieved_tavily_docs

    # 格式化为纯文本，供下游节点直接使用
    return {
        "retrieved_tables": _format_table_docs(results["tables"]),
        "retrieved_docs": retrieved_docs
    }