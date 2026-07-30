"""
表格推理评测脚本
评测智能体对表格数据的处理效果，只走 numerical_node 节点。

用法:
    cd src
    python3 -m eco_agent.table_eval --limit 2                          # 每个文件只测2条
    python3 -m eco_agent.table_eval --dataset-dir <path>               # 指定数据集目录
    python3 -m eco_agent.table_eval --model deepseek/DeepSeek-V3.2     # 指定模型
    python3 -m eco_agent.table_eval --workers 3                        # 3路并行（默认，充分利用服务器3张卡）
    python3 -m eco_agent.table_eval --workers 1                        # 串行模式
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from tqdm.asyncio import tqdm as tqdm_async

from eco_agent.nodes.numerical_node import numerical_node
from eco_agent.context import Context
from eco_agent.state import State
from eco_agent.utils import QueryExecutor


# ============================================================
# 答案提取
# ============================================================

def extract_final_answer(text: str) -> str:
    """从模型多步推理文本中提取最终答案.

    尝试匹配常见答案标记，取标记后的整段内容。
    找不到标记则返回原文。
    """
    if not text:
        return ""

    # 预处理：去掉 markdown 加粗/斜体标记，方便正则匹配
    cleaned = re.sub(r'\*{1,3}([^*]+?)\*{1,3}', r'\1', text)

    # 按优先级排列的答案标记
    markers = [
        "最终答案",
        "结论",
        "答案",
        "The answer is",
        "Answer",
        "Conclusion",
        "因此，",
        "因此,",
    ]

    best = None
    best_pos = len(cleaned)

    # 找到最早出现的答案标记
    for marker in markers:
        pos = cleaned.find(marker)
        if pos != -1 and pos < best_pos:
            # 确保标记后紧跟 :： 或换行等
            after_marker = cleaned[pos + len(marker):]
            # 跳过 :： 和空白
            m = re.match(r'\s*[:：]\s*', after_marker)
            if m:
                content_start = pos + len(marker) + m.end()
                content = cleaned[content_start:].strip()
                if content and len(content) >= 1:
                    best = content
                    best_pos = pos

    if best:
        # 清理残留的 markdown 符号
        best = best.strip("*").strip()
        return best

    # 没有显式标记，返回 cleaned 全文
    return cleaned.strip()


# ============================================================
# 数字提取
# ============================================================

def extract_numbers(text: str) -> list[float]:
    """从文本中提取所有数字（处理千分位逗号、百分号）。"""
    if not text:
        return []
    # 去掉中文逗号
    cleaned = text.replace(",", "").replace("，", "")
    # 匹配数字（含小数、负号）
    numbers = re.findall(r'-?\d+\.?\d*', cleaned)
    result = []
    for n in numbers:
        try:
            result.append(float(n))
        except ValueError:
            pass
    return result


# ============================================================
# 答案对比
# ============================================================

def compare_answers(
    model_answer: str,
    gold_answer: str,
    gold_unit: str = "",
    full_answer: str = "",
) -> tuple[bool, str, str]:
    """对比模型回答和标准答案。

    Args:
        model_answer: 提取后的答案（用于数值/文本匹配）
        gold_answer: 标准答案
        gold_unit: 期望单位
        full_answer: 完整模型输出（用于单位检查，因提取后可能丢失单位）

    Returns:
        (is_correct, extracted_answer, comparison_detail)
    """
    if not model_answer:
        return False, "", "Model answer is empty"

    gold = gold_answer.strip()
    model = model_answer.strip()
    unit = gold_unit.strip()

    def _check_unit(text: str) -> tuple[bool, str]:
        """检查文本中是否包含目标单位. gold_unit为空时跳过.

        优先在提取后的答案中查找，找不到则在完整模型输出中查找
        （因为提取过程可能丢失了单位信息）。
        """
        if not unit:
            return True, "unit check skipped (empty)"
        if unit in text:
            return True, f"unit '{unit}' found in extracted answer"
        if full_answer and unit in full_answer:
            return True, f"unit '{unit}' found in full answer"
        return False, f"unit '{unit}' not found in answer"

    # ---- 策略1: 直接包含 ----
    if gold in model:
        unit_ok, unit_msg = _check_unit(model)
        if not unit_ok:
            return False, model, f"Answer matched but {unit_msg}"
        return True, model, f"gold_answer found directly in model answer; {unit_msg}"

    # ---- 策略2: 数值比较 ----
    gold_nums = extract_numbers(gold)
    model_nums = extract_numbers(model)

    if gold_nums and model_nums:
        # 单个数值的 gold_answer
        if len(gold_nums) == 1:
            gold_val = gold_nums[0]
            best_rel_diff = float("inf")
            best_match = None
            for m_val in model_nums:
                if abs(gold_val) > 1e-9:
                    rel_diff = abs(m_val - gold_val) / abs(gold_val)
                else:
                    rel_diff = abs(m_val - gold_val)
                if rel_diff < best_rel_diff:
                    best_rel_diff = rel_diff
                    best_match = m_val
            if best_match is not None and best_rel_diff < 0.01:  # 1%容差
                unit_ok, unit_msg = _check_unit(model)
                if not unit_ok:
                    return False, str(best_match), f"Numerical match but {unit_msg}"
                return True, str(best_match), (
                    f"Numerical match: model={best_match}, gold={gold_val}, "
                    f"rel_diff={best_rel_diff:.6f}; {unit_msg}"
                )

        # 多个数值：检查所有gold数值是否在model中出现
        else:
            all_found = True
            for gv in gold_nums:
                found = False
                for mv in model_nums:
                    if abs(gv) > 1e-9:
                        if abs(mv - gv) / abs(gv) < 0.01:
                            found = True
                            break
                    elif abs(mv - gv) < 0.01:
                        found = True
                        break
                if not found:
                    all_found = False
                    break
            if all_found:
                unit_ok, unit_msg = _check_unit(model)
                if not unit_ok:
                    return False, model, f"All numbers matched but {unit_msg}"
                return True, model, f"All gold numbers found in model answer; {unit_msg}"

    # ---- 策略3: 逗号分隔文本 ----
    if "," in gold or "，" in gold:
        gold_parts = [p.strip() for p in re.split(r"[，,]", gold) if p.strip()]
        all_found = all(part in model for part in gold_parts)
        if all_found:
            unit_ok, unit_msg = _check_unit(model)
            if not unit_ok:
                return False, model, f"Parts matched but {unit_msg}"
            return True, model, f"All comma-separated gold parts found in model answer; {unit_msg}"

    # ---- 策略4: 清洗后直接比较 ----
    model_clean = re.sub(r"\s+", "", model)
    gold_clean = re.sub(r"\s+", "", gold)
    if gold_clean in model_clean:
        unit_ok, unit_msg = _check_unit(model)
        if not unit_ok:
            return False, model, f"Answer matched (cleaned) but {unit_msg}"
        return True, model, f"gold_answer found in model answer (whitespace cleaned); {unit_msg}"

    # ---- 策略5: 模糊匹配 ----
    if len(gold) > 5:
        for i in range(len(gold) - 4):
            sub = gold[i:i + 5]
            if sub in model:
                return False, model, (
                    f"Partial match only: found '{sub}' but gold='{gold}'"
                )

    return False, model, "No match found"


# ============================================================
# 构建评测图
# ============================================================

def build_eval_graph():
    """构建最小化评测图: START → numerical_node → END."""
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("numerical_node", numerical_node)
    builder.add_edge(START, "numerical_node")
    builder.add_edge("numerical_node", END)
    return builder.compile(name="Table Eval Graph")


# ============================================================
# 单条评测
# ============================================================

async def evaluate_single_item(
    item: dict,
    db: QueryExecutor,
    graph,
    model_name: str = "deepseek/DeepSeek-V3.2",
    verbose: bool = False,
) -> dict:
    """评测单条数据。

    Args:
        item: 评测数据项 (含 table_name, question, gold_answer 等)
        db: 数据库查询器
        graph: 编译好的 langgraph 图
        model_name: 模型名
        verbose: 是否打印详细日志

    Returns:
        评测结果dict (原始字段 + model_final_answer, is_correct 等)
    """
    table_name = item["table_name"]
    question = item["question"]
    gold_answer = item.get("gold_answer", "")
    gold_unit = item.get("gold_unit", "")

    base_result = {
        "table_id": item.get("table_id", ""),
        "table_name": table_name,
        "question": question,
        "gold_answer": gold_answer,
        "gold_unit": gold_unit,
        "error": None,
        "model_final_answer": "",
        "extracted_answer": "",
        "is_correct": False,
        "comparison_detail": "",
    }

    # 1. 从数据库获取表格内容
    try:
        headers, rows = db.get_table_data(table_name)
        if headers is None:
            base_result["error"] = f"Table '{table_name}' not found in database"
            return base_result
        md_table = db.table_to_markdown(headers, rows)
        if verbose:
            print(f"\n      Table: {table_name} ({len(headers)} cols, {len(rows) if rows else 0} rows)")
    except Exception as e:
        base_result["error"] = f"DB error fetching '{table_name}': {str(e)}"
        return base_result

    # 2. 构建初始状态
    initial_state = {
        "question": question,
        "retrieved_docs": "",
        "retrieved_tables": md_table,
        "messages": [HumanMessage(content=question)],
    }

    # 3. 运行图（带重试，处理 503 等临时错误）
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 5  # 秒

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await graph.ainvoke(
                initial_state,
                context=Context(model=model_name),
            )

            final_answer = result.get("final_answer", "")
            if not final_answer:
                # 尝试从 messages 中获取最后一条消息
                msgs = result.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, "content"):
                        final_answer = str(last_msg.content)

            base_result["model_final_answer"] = final_answer

            # 4. 提取答案并对比
            extracted = extract_final_answer(final_answer)
            base_result["extracted_answer"] = extracted

            is_correct, extracted_ans, detail = compare_answers(
                extracted, gold_answer, gold_unit, full_answer=final_answer
            )
            base_result["is_correct"] = is_correct
            base_result["extracted_answer"] = extracted_ans
            base_result["comparison_detail"] = detail
            break  # 成功，跳出重试循环

        except Exception as e:
            last_error = str(e)
            # 判断是否为可重试的错误（HTTP 503/502/504、连接错误、超时）
            retryable = any(kw in last_error.lower() for kw in (
                "503", "502", "504", "500",
                "connection", "timeout", "timed out",
                "service unavailable", "too many requests",
                "rate limit",
            ))
            if retryable and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 5, 10, 20s
                if verbose:
                    print(f"\n      ⚠ Retry {attempt}/{MAX_RETRIES} in {delay}s: {last_error}")
                await asyncio.sleep(delay)
            else:
                base_result["error"] = f"Model/invoke error: {last_error}"
                break

    return base_result


# ============================================================
# 逐文件评测
# ============================================================

async def evaluate_dataset(
    file_path: str,
    output_dir: str,
    model_name: str = "deepseek/DeepSeek-V3.2",
    limit: int = 0,
    workers: int = 3,
    verbose: bool = False,
) -> dict:
    """评测单个数据集文件。

    Args:
        file_path: JSON 数据集文件路径
        output_dir: 结果输出目录
        model_name: 模型名
        limit: 最多评测条数 (0 = 不限)
        workers: 并发数 (1 = 串行)
        verbose: 是否打印详细信息

    Returns:
        带统计信息的 summary dict
    """
    file_name = Path(file_path).stem
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {file_name}")
    print(f"  Workers: {workers}")
    print(f"{'=' * 70}")

    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 只评测 human_fix == 1 的数据
    skipped = sum(1 for it in items if it.get("human_fix") != 1)
    items = [it for it in items if it.get("human_fix") == 1]
    if skipped:
        print(f"  Skipped (human_fix!=1): {skipped}")

    if limit and limit > 0:
        items = items[:limit]

    total = len(items)
    print(f"  Items: {total}")

    graph = build_eval_graph()

    # ---- 串行模式：共享一个 DB 连接 ----
    if workers <= 1:
        db = QueryExecutor()
        try:
            results = []
            correct = 0
            errors = 0

            with tqdm_async(total=total, desc=f"  {file_name}", unit="item") as pbar:
                for i, item in enumerate(items):
                    result = await evaluate_single_item(
                        item, db, graph, model_name, verbose=verbose
                    )
                    results.append(result)

                    if result.get("error"):
                        errors += 1
                    elif result["is_correct"]:
                        correct += 1

                    pbar.set_postfix_str(
                        f"✅{correct} ✗{i + 1 - correct - errors} ❌{errors}"
                    )
                    pbar.update(1)
        finally:
            db.close()

    # ---- 并行模式：每个任务独立 DB 连接 ----
    else:
        sem = asyncio.Semaphore(workers)

        async def eval_one(idx: int, item: dict) -> tuple[int, dict]:
            """执行单条评测（并发控制 + 独立 DB 连接）。"""
            async with sem:
                db = QueryExecutor()
                try:
                    result = await evaluate_single_item(
                        item, db, graph, model_name, verbose=verbose
                    )
                    return idx, result
                finally:
                    db.close()

        tasks = [eval_one(i, item) for i, item in enumerate(items)]
        results = [None] * total
        correct = 0
        errors = 0
        wrong = 0

        with tqdm_async(total=total, desc=f"  {file_name}", unit="item") as pbar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results[idx] = result

                if result.get("error"):
                    errors += 1
                elif result["is_correct"]:
                    correct += 1
                else:
                    wrong += 1

                pbar.set_postfix_str(f"✅{correct} ✗{wrong} ❌{errors}")
                pbar.update(1)

    # ---- 统计 & 保存 ----
    valid = total - errors
    accuracy = correct / valid if valid > 0 else 0.0

    output_path = os.path.join(output_dir, f"{file_name}_results.json")
    summary = {
        "dataset": file_name,
        "file_path": file_path,
        "model": model_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "total": total,
        "correct": correct,
        "errors": errors,
        "valid": valid,
        "accuracy": round(accuracy, 6),
        "workers": workers,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  --- {file_name} ---")
    print(f"  Correct: {correct}/{valid}  Errors: {errors}  Accuracy: {accuracy:.2%}")
    print(f"  Saved: {output_path}")

    return summary


# ============================================================
# 主入口
# ============================================================

async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Table Reasoning Evaluation — 表格推理评测"
    )
    parser.add_argument(
        "--dataset-dir",
        default="/home/zhangruochi/eco_model/ssd/llfdata/table_data/create_from_ecodb/20260515-评测基准",
        help="数据集目录路径",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="结果输出目录（默认在数据集目录下创建 eval_results）",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", "deepseek/DeepSeek-V3.2"),
        help="模型名 (格式: provider/model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="每个文件最多测试 N 条 (0 = 不限)",
    )
    parser.add_argument(
        "--file",
        default="",
        help="只评测指定文件（文件名，不含路径），留空则评测所有文件",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印详细日志",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="并发 worker 数（默认 3，对应服务器 3 张 GPU；设为 1 则串行）",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        print(f"错误：数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    # 模型名中的 / 替换为 _ ，作为子目录名 (e.g. deepseek/DeepSeek-V3.2 → deepseek_DeepSeek-V3.2)
    model_dir = args.model.replace("/", "_").replace("\\", "_")
    base_output_dir = args.output_dir or os.path.join(dataset_dir, "eval_results")
    output_dir = os.path.join(base_output_dir, model_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Table Reasoning Evaluation")
    print("=" * 70)
    print(f"  Dataset dir : {dataset_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Model       : {args.model}")
    print(f"  Workers     : {args.workers}")
    if args.limit:
        print(f"  Limit       : {args.limit} items per file")
    if args.file:
        print(f"  File filter : {args.file}")

    # 收集数据集文件
    pattern = args.file if args.file else "*.json"
    dataset_files = sorted(Path(dataset_dir).glob(pattern))
    if not dataset_files:
        print(f"未找到匹配的数据集文件: {pattern}")
        sys.exit(1)

    print(f"\n  Found {len(dataset_files)} dataset file(s)")

    all_summaries = []

    # 文件级进度条
    with tqdm_async(
        total=len(dataset_files), desc="Overall", unit="file", position=0
    ) as overall_pbar:
        for file_path in dataset_files:
            file_name = Path(file_path).stem
            overall_pbar.set_postfix_str(f"current: {file_name}")
            overall_pbar.update(0)  # refresh display

            summary = await evaluate_dataset(
                str(file_path),
                output_dir,
                model_name=args.model,
                limit=args.limit,
                workers=args.workers,
                verbose=args.verbose,
            )
            all_summaries.append(summary)
            overall_pbar.update(1)

    # ========================
    # 汇总
    # ========================
    if len(all_summaries) <= 1:
        print("\nDone.")
        return

    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY")
    print("=" * 70)

    total_all = sum(s["total"] for s in all_summaries)
    correct_all = sum(s["correct"] for s in all_summaries)
    valid_all = sum(s["valid"] for s in all_summaries)
    overall_acc = correct_all / valid_all if valid_all > 0 else 0.0

    col_w = 32
    print(f"  {'Dataset':<{col_w}} {'Accuracy':>10}  {'Correct/Valid':>14}")
    print(f"  {'-' * col_w}  {'-' * 10}  {'-' * 14}")
    for s in all_summaries:
        name = s["dataset"]
        acc = s["accuracy"]
        c = s["correct"]
        v = s["valid"]
        print(f"  {name:<{col_w}} {acc:>9.2%}  {c:>5}/{v:<5}")

    print(f"  {'─' * col_w}  {'─' * 10}  {'─' * 14}")
    print(f"  {'OVERALL':<{col_w}} {overall_acc:>9.2%}  {correct_all:>5}/{valid_all:<5}")

    # 保存汇总
    overall_path = os.path.join(output_dir, "overall_summary.json")
    overall = {
        "model": args.model,
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_dir": dataset_dir,
        "workers": args.workers,
        "overall_accuracy": round(overall_acc, 6),
        "total_all": total_all,
        "correct_all": correct_all,
        "valid_all": valid_all,
        "error_all": total_all - valid_all,
        "datasets": [
            {k: v for k, v in s.items() if k != "results"}
            for s in all_summaries
        ],
    }
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"\n  Overall summary saved to: {overall_path}")


if __name__ == "__main__":
    asyncio.run(main())
