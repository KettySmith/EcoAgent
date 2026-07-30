"""
后处理评测结果：增加单位校验 + human_fix=1 过滤，重新统计准确率。

用法:
    cd src
    python3 -m eco_agent.post_eval

输出:
    eval_results/after_<模型名>/   — 后处理结果
"""

import json
import os
import re
from datetime import datetime, UTC
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
DATASET_DIR = "/home/zhangruochi/eco_model/ssd/llfdata/table_data/create_from_ecodb/20260515-评测基准"
EVAL_DIR = os.path.join(DATASET_DIR, "eval_results")


# ── 复用 table_eval 中的答案对比逻辑 ───────────────────────

def extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    cleaned = text.replace(",", "").replace("，", "")
    numbers = re.findall(r'-?\d+\.?\d*', cleaned)
    result = []
    for n in numbers:
        try:
            result.append(float(n))
        except ValueError:
            pass
    return result


def compare_answers(
    model_answer: str,
    gold_answer: str,
    gold_unit: str = "",
    full_answer: str = "",
) -> tuple[bool, str]:
    """后处理对比：返回 (is_correct, detail)。

    Args:
        model_answer: 提取后的答案
        gold_answer: 标准答案
        gold_unit: 期望单位
        full_answer: 完整模型输出（用于单位检查）
    """
    if not model_answer:
        return False, "Model answer is empty"

    gold = gold_answer.strip()
    model = model_answer.strip()
    unit = gold_unit.strip()

    def _check_unit(text: str) -> tuple[bool, str]:
        if not unit:
            return True, "unit check skipped (empty)"
        if unit in text:
            return True, f"unit '{unit}' found in extracted answer"
        if full_answer and unit in full_answer:
            return True, f"unit '{unit}' found in full answer"
        return False, f"unit '{unit}' not found"

    # 策略1: 直接包含
    if gold in model:
        unit_ok, unit_msg = _check_unit(model)
        if not unit_ok:
            return False, f"Answer matched but {unit_msg}"
        return True, f"gold_answer directly found; {unit_msg}"

    # 策略2: 数值比较
    gold_nums = extract_numbers(gold)
    model_nums = extract_numbers(model)

    if gold_nums and model_nums:
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
            if best_match is not None and best_rel_diff < 0.01:
                unit_ok, unit_msg = _check_unit(model)
                if not unit_ok:
                    return False, f"Numerical match but {unit_msg}"
                return True, f"Numerical match rel_diff={best_rel_diff:.6f}; {unit_msg}"

        else:
            all_found = True
            for gv in gold_nums:
                found = any(
                    abs(mv - gv) / max(abs(gv), 1e-9) < 0.01
                    for mv in model_nums
                )
                if not found:
                    all_found = False
                    break
            if all_found:
                unit_ok, unit_msg = _check_unit(model)
                if not unit_ok:
                    return False, f"All numbers matched but {unit_msg}"
                return True, f"All gold numbers found; {unit_msg}"

    # 策略3: 逗号分隔文本
    if "," in gold or "，" in gold:
        gold_parts = [p.strip() for p in re.split(r"[，,]", gold) if p.strip()]
        if all(part in model for part in gold_parts):
            unit_ok, unit_msg = _check_unit(model)
            if not unit_ok:
                return False, f"Parts matched but {unit_msg}"
            return True, f"All comma parts found; {unit_msg}"

    # 策略4: 清洗后比较
    model_clean = re.sub(r"\s+", "", model)
    gold_clean = re.sub(r"\s+", "", gold)
    if gold_clean in model_clean:
        unit_ok, unit_msg = _check_unit(model)
        if not unit_ok:
            return False, f"Answer matched (cleaned) but {unit_msg}"
        return True, f"gold_answer found (whitespace cleaned); {unit_msg}"

    # 策略5: 模糊匹配
    if len(gold) > 5:
        for i in range(len(gold) - 4):
            if gold[i:i + 5] in model:
                return False, f"Partial match only: '{gold[i:i+5]}' in '{gold}'"

    return False, "No match found"


# ── 加载原始数据集（获取 human_fix 字段）───────────────────

def load_dataset_index() -> dict[str, dict]:
    """加载所有原始数据集，构建 (table_id → item) 的索引."""
    index = {}
    for fname in sorted(os.listdir(DATASET_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(DATASET_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            tid = item.get("table_id", "")
            if tid:
                # 用 table_id 作为索引键
                index[str(tid)] = item
    return index


# ── 处理单个结果文件 ─────────────────────────────────────

def postprocess_results_file(results_path: str, dataset_index: dict) -> dict:
    """对单个 *_results.json 进行后处理."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_total = data["total"]
    old_correct = data["correct"]
    results = data["results"]

    new_results = []
    correct = 0
    skipped_no_human_fix = 0
    unit_corrected = 0  # 原本正确但因单位校验变错误的

    for r in results:
        tid = str(r.get("table_id", ""))
        orig = dataset_index.get(tid, {})

        # human_fix 过滤
        if orig.get("human_fix") != 1:
            skipped_no_human_fix += 1
            continue

        gold_answer = r.get("gold_answer", "")
        gold_unit = r.get("gold_unit", "")
        extracted = r.get("extracted_answer", "")
        full_answer = r.get("model_final_answer", "")
        had_error = bool(r.get("error"))

        if had_error:
            new_results.append(r)
            continue

        # 重新判断
        is_correct, detail = compare_answers(
            extracted, gold_answer, gold_unit, full_answer=full_answer
        )

        # 统计单位导致的变化
        if r.get("is_correct") and not is_correct:
            unit_corrected += 1

        r["is_correct"] = is_correct
        r["comparison_detail"] = detail
        new_results.append(r)

        if is_correct:
            correct += 1

    valid = len(new_results) - sum(1 for r in new_results if r.get("error"))
    accuracy = correct / valid if valid > 0 else 0.0

    return {
        **{k: v for k, v in data.items() if k != "results"},
        "total_original": old_total,
        "correct_original": old_correct,
        "total_after_filter": len(new_results),
        "skipped_human_fix": skipped_no_human_fix,
        "unit_corrected": unit_corrected,
        "correct": correct,
        "errors": sum(1 for r in new_results if r.get("error")),
        "valid": valid,
        "accuracy": round(accuracy, 6),
        "timestamp_post": datetime.now(UTC).isoformat(),
        "results": new_results,
    }


# ── 主入口 ────────────────────────────────────────────────

def main():
    dataset_index = load_dataset_index()
    print(f"Loaded dataset index: {len(dataset_index)} items")

    # 找到已有的模型目录（跳过 after_ 前缀的）
    model_dirs = []
    for name in sorted(os.listdir(EVAL_DIR)):
        full = os.path.join(EVAL_DIR, name)
        if os.path.isdir(full) and not name.startswith("after_"):
            model_dirs.append((name, full))

    if not model_dirs:
        print("No model directories found.")
        return

    print(f"Found {len(model_dirs)} model(s): {[m[0] for m in model_dirs]}")

    for model_name, model_path in model_dirs:
        after_dir = os.path.join(EVAL_DIR, f"after_{model_name}")
        os.makedirs(after_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Processing: {model_name} → after_{model_name}")
        print(f"{'=' * 60}")

        summaries = []

        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith("_results.json"):
                continue
            src_path = os.path.join(model_path, fname)
            dataset_name = fname.replace("_results.json", "")

            summary = postprocess_results_file(src_path, dataset_index)
            summaries.append(summary)

            # 保存
            dst_path = os.path.join(after_dir, fname)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            old_acc = summary["correct_original"] / summary["total_original"] \
                if summary["total_original"] else 0
            print(f"  {dataset_name:<35} "
                  f"{old_acc:.2%} → {summary['accuracy']:.2%}  "
                  f"({summary['correct']}/{summary['valid']})  "
                  f"skip:{summary['skipped_human_fix']}  "
                  f"unit_fix:{summary['unit_corrected']}")

        # 汇总
        total_all = sum(s["total_after_filter"] for s in summaries)
        correct_all = sum(s["correct"] for s in summaries)
        valid_all = sum(s["valid"] for s in summaries)
        overall_acc = correct_all / valid_all if valid_all > 0 else 0.0

        print(f"  {'─' * 50}")
        print(f"  {'OVERALL':<35} {overall_acc:.2%}  ({correct_all}/{valid_all})")

        # 保存 overall_summary
        overall_path = os.path.join(after_dir, "overall_summary.json")
        overall = {
            "model": model_name,
            "timestamp_post": datetime.now(UTC).isoformat(),
            "overall_accuracy": round(overall_acc, 6),
            "total_all": total_all,
            "correct_all": correct_all,
            "valid_all": valid_all,
            "datasets": [
                {k: v for k, v in s.items() if k != "results"}
                for s in summaries
            ],
        }
        with open(overall_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to: {after_dir}/")


if __name__ == "__main__":
    main()
