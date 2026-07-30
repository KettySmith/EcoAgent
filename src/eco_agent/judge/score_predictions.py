#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
HUMAN_FIXED_DIR = ROOT_DIR / "human_fixed"

TASK_TO_STANDARD = {
    "basic_arithmetic": HUMAN_FIXED_DIR / "Basic_Arithmetic_dataset.json",
    "average": HUMAN_FIXED_DIR / "average_dataset.json",
    "count": HUMAN_FIXED_DIR / "count_dataset.json",
    "extremum": HUMAN_FIXED_DIR / "extremum_dataset.json",
    "top_k": HUMAN_FIXED_DIR / "top-k_dataset.json",
}

TASK_FILE_HINTS = [
    ("Basic_Arithmetic_dataset", "basic_arithmetic"),
    ("basic_arithmetic_data", "basic_arithmetic"),
    ("average_dataset", "average"),
    ("average_data", "average"),
    ("count_dataset", "count"),
    ("count_data", "count"),
    ("extremum_dataset", "extremum"),
    ("extremum_data", "extremum"),
    ("top-k_dataset", "top_k"),
    ("top-k_data", "top_k"),
]

DETAIL_OUTPUT_FORMAT = "输出格式错误"
DETAIL_SORT_LOGIC = "排序/比较逻辑错误"
DETAIL_OPERATION = "运算类型错误"
DETAIL_MISSING = "参数缺失"
DETAIL_REDUNDANT = "参数冗余"
DETAIL_VALUE = "参数值错误"
DETAIL_RESULT_EXTRACTION = "结果提取错误"
DETAIL_CALC_EXECUTION = "计算执行错误"

REASON_STRUCTURE = "结构输出错误"
REASON_TABLE_LOOKUP = "表格取值错误"
REASON_LOGIC = "推理逻辑错误"
REASON_CALCULATION = "计算执行错误"
REASON_RESULT_ORGANIZATION = "结果组织错误"

REASON_OUTPUT_FORMAT = DETAIL_OUTPUT_FORMAT
REASON_SORT_LOGIC = DETAIL_SORT_LOGIC
REASON_OPERATION = DETAIL_OPERATION
REASON_MISSING = DETAIL_MISSING
REASON_REDUNDANT = DETAIL_REDUNDANT
REASON_VALUE = DETAIL_VALUE
REASON_RESULT_EXTRACTION = DETAIL_RESULT_EXTRACTION
REASON_CALC_EXECUTION = DETAIL_CALC_EXECUTION

TRANSLATION_TABLE = str.maketrans(
    {
        "，": ",",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "％": "%",
        "－": "-",
        "—": "-",
        "–": "-",
    }
)

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
NUMBER_RE = re.compile(NUMBER_PATTERN)
PERCENT_RE = re.compile(rf"^({NUMBER_PATTERN})%$")
RESULT_COMPARE_PLACES = 2


@dataclass
class ProgramScore:
    correct: bool
    reason: str = ""
    format_error: bool = False


def load_json_list(path: Path) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding) as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"{path} does not contain a JSON list.")
            return [item for item in data if isinstance(item, dict)]
        except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load JSON list from {path}: {last_error}")


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).translate(TRANSLATION_TABLE)
    return re.sub(r"\s+", "", text)


def normalize_question(value: Any) -> str:
    return normalize_space(value)


def normalize_label(value: Any) -> str:
    text = normalize_space(value)
    return text.strip(".,;:!?。；：")


def parse_decimal_text(value: Any) -> Decimal | None:
    text = normalize_space(value)
    if not text or text.endswith("%") or not re.fullmatch(NUMBER_PATTERN, text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_percent_text(value: Any) -> Decimal | None:
    text = normalize_space(value)
    match = PERCENT_RE.fullmatch(text)
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def decimal_equal(left: Decimal, right: Decimal) -> bool:
    return left == right


def round_decimal(value: Decimal, places: int) -> Decimal:
    quantizer = Decimal("1").scaleb(-places)
    return value.quantize(quantizer, rounding=ROUND_HALF_UP)


def decimal_equal_with_round(left: Decimal, right: Decimal, places: int) -> bool:
    return round_decimal(left, places) == round_decimal(right, places)


def decimal_equal_with_tolerance(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    return abs(left - right) <= tolerance


def split_top_level(text: str, delimiters: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth_round = 0
    depth_square = 0
    depth_curly = 0

    for char in text:
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly = max(0, depth_curly - 1)

        if (
            char in delimiters
            and depth_round == 0
            and depth_square == 0
            and depth_curly == 0
        ):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def find_last_top_level(text: str, target: str) -> int:
    depth_round = 0
    depth_square = 0
    depth_curly = 0
    last_index = -1
    for index, char in enumerate(text):
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly = max(0, depth_curly - 1)
        elif (
            char == target
            and depth_round == 0
            and depth_square == 0
            and depth_curly == 0
        ):
            last_index = index
    return last_index


def extract_numbers(text: Any) -> list[Decimal]:
    numbers: list[Decimal] = []
    for match in NUMBER_RE.findall(normalize_space(text)):
        try:
            numbers.append(Decimal(match))
        except InvalidOperation:
            continue
    return numbers


def split_csv_like(text: Any) -> list[str]:
    normalized = normalize_space(text)
    if not normalized:
        return []
    return [normalize_label(part) for part in split_top_level(normalized, ",")]


def detect_task_from_name(path: Path) -> str:
    file_name = path.name.lower()
    for hint, task in TASK_FILE_HINTS:
        if hint.lower() in file_name:
            return task
    raise ValueError(f"Cannot infer task type from file name: {path.name}")


def make_align_key(item: dict[str, Any]) -> tuple[str, str]:
    table_id = normalize_label(item.get("table_id", ""))
    question = normalize_question(item.get("question", ""))
    return table_id, question


def build_standard_index(items: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        if item.get("human_fix") != 1:
            continue
        key = make_align_key(item)
        if key[0] and key[1] and key not in index:
            index[key] = item
    return index


YEAR_ONLY_RE = re.compile(r"^(?P<year>\d{4})(?:年|年度)?$")
YEAR_CENSUS_RE = re.compile(r"^(?P<year>\d{4})(?:年|年度)?(?:全国)?人口普查$")
ADMIN_REGION_SUFFIXES = (
    "维吾尔自治区",
    "壮族自治区",
    "回族自治区",
    "特别行政区",
    "自治区",
    "省",
    "市",
)
TOP_K_OPTIONAL_SUFFIXES = ADMIN_REGION_SUFFIXES + ("废物",)
EXTREMUM_OPTIONAL_SUFFIXES = ADMIN_REGION_SUFFIXES


def canonicalize_year_like_label(value: Any) -> str:
    label = normalize_label(value)
    if not label:
        return ""

    year_match = YEAR_ONLY_RE.fullmatch(label)
    if year_match:
        return f"{year_match.group('year')}年"

    census_match = YEAR_CENSUS_RE.fullmatch(label)
    if census_match:
        return f"{census_match.group('year')}年人口普查"

    label = re.sub(r"(?<!\d)(\d{4})年度", r"\1年", label)
    label = re.sub(r"(?<!\d)(\d{4})(?=(?:全国)?人口普查$)", r"\1年", label)
    label = label.replace("全国人口普查", "人口普查")
    return label


def build_label_variants(value: Any, task: str) -> set[str]:
    base = canonicalize_year_like_label(value)
    if not base:
        return set()

    variants = {base}

    year_match = re.fullmatch(r"(\d{4})年", base)
    if year_match:
        variants.add(year_match.group(1))

    census_match = re.fullmatch(r"(\d{4})年人口普查", base)
    if census_match:
        year = census_match.group(1)
        variants.update(
            {
                f"{year}人口普查",
                f"{year}全国人口普查",
                f"{year}年全国人口普查",
            }
        )

    suffixes: tuple[str, ...] = ()
    if task == "top_k":
        suffixes = TOP_K_OPTIONAL_SUFFIXES
    elif task == "extremum":
        suffixes = EXTREMUM_OPTIONAL_SUFFIXES

    for suffix in suffixes:
        if base.endswith(suffix) and len(base) > len(suffix):
            variants.add(base[: -len(suffix)])

    return {item for item in variants if item}


def labels_equivalent(predicted: Any, standard: Any, task: str) -> bool:
    pred_variants = build_label_variants(predicted, task)
    std_variants = build_label_variants(standard, task)
    return bool(pred_variants and std_variants and pred_variants.intersection(std_variants))


def ordered_labels_equal(predicted: list[str], standard: list[str], task: str) -> bool:
    if not predicted or not standard or len(predicted) != len(standard):
        return False
    return all(labels_equivalent(pred_label, std_label, task) for pred_label, std_label in zip(predicted, standard))


def compare_basic_result(predicted: Any, standard: Any) -> bool:
    pred_text = normalize_space(predicted)
    std_text = normalize_space(standard)
    if not pred_text or not std_text:
        return False

    pred_parts = split_top_level(pred_text, ",")
    std_parts = split_top_level(std_text, ",")
    if len(pred_parts) == 2 and len(std_parts) == 2:
        if not labels_equivalent(pred_parts[0], std_parts[0], "basic_arithmetic"):
            return False
        pred_percent = parse_percent_text(pred_parts[1])
        std_percent = parse_percent_text(std_parts[1])
        if pred_percent is not None or std_percent is not None:
            return (
                pred_percent is not None
                and std_percent is not None
                and decimal_equal_with_round(pred_percent, std_percent, RESULT_COMPARE_PLACES)
            )
        pred_number = parse_decimal_text(pred_parts[1])
        std_number = parse_decimal_text(std_parts[1])
        return (
            pred_number is not None
            and std_number is not None
            and decimal_equal_with_round(pred_number, std_number, RESULT_COMPARE_PLACES)
        )

    pred_percent = parse_percent_text(pred_text)
    std_percent = parse_percent_text(std_text)
    if pred_percent is not None or std_percent is not None:
        return (
            pred_percent is not None
            and std_percent is not None
            and decimal_equal_with_round(pred_percent, std_percent, RESULT_COMPARE_PLACES)
        )

    pred_number = parse_decimal_text(pred_text)
    std_number = parse_decimal_text(std_text)
    if pred_number is not None and std_number is not None:
        return decimal_equal_with_round(pred_number, std_number, RESULT_COMPARE_PLACES)
    return normalize_label(pred_text) == normalize_label(std_text)


def compare_average_result(predicted: Any, standard: Any) -> bool:
    pred_parts = split_csv_like(predicted)
    std_parts = split_csv_like(standard)
    if not pred_parts or not std_parts or len(pred_parts) != len(std_parts):
        return False
    for pred_part, std_part in zip(pred_parts, std_parts):
        pred_number = parse_decimal_text(pred_part)
        std_number = parse_decimal_text(std_part)
        if (
            pred_number is None
            or std_number is None
            or not decimal_equal_with_round(pred_number, std_number, RESULT_COMPARE_PLACES)
        ):
            return False
    return True


def compare_count_result(predicted: Any, standard: Any) -> bool:
    pred_number = parse_decimal_text(predicted)
    std_number = parse_decimal_text(standard)
    if pred_number is None or std_number is None:
        return normalize_label(predicted) == normalize_label(standard)
    return decimal_equal(pred_number, std_number)


def compare_extremum_result(predicted: Any, standard: Any) -> bool:
    return labels_equivalent(predicted, standard, "extremum")


def compare_top_k_result(predicted: Any, standard: Any) -> bool:
    pred_parts = split_csv_like(predicted)
    std_parts = split_csv_like(standard)
    return ordered_labels_equal(pred_parts, std_parts, "top_k")


def compare_unit(predicted: Any, standard: Any) -> bool:
    return normalize_label(predicted) == normalize_label(standard)


def compare_result(task: str, predicted: Any, standard: Any) -> bool:
    if task == "basic_arithmetic":
        return compare_basic_result(predicted, standard)
    if task == "average":
        return compare_average_result(predicted, standard)
    if task == "count":
        return compare_count_result(predicted, standard)
    if task == "extremum":
        return compare_extremum_result(predicted, standard)
    if task == "top_k":
        return compare_top_k_result(predicted, standard)
    raise ValueError(f"Unsupported task type: {task}")


def replace_numbers_with_placeholder(text: str) -> str:
    return re.sub(NUMBER_PATTERN, "N", text)


def classify_number_count(predicted: list[Any], standard: list[Any]) -> str:
    if len(predicted) < len(standard):
        return DETAIL_MISSING
    if len(predicted) > len(standard):
        return DETAIL_REDUNDANT
    return DETAIL_VALUE


def basic_operator_family(program: Any) -> str:
    normalized = normalize_space(program)
    if "差值:" in normalized and any(symbol in normalized for symbol in ("<", ">", "<=", ">=")):
        return "compare_diff"
    placeholder = replace_numbers_with_placeholder(normalized)
    if "/" in placeholder and "*N" in placeholder:
        return "percent_formula"
    operators = "".join(char for char in placeholder if char in "+-*/<>")
    unique = set(operators)
    if unique == {"+"}:
        return "addition"
    if unique == {"-"}:
        return "subtraction"
    if unique == {"*"}:
        return "multiplication"
    if unique == {"/"}:
        return "division"
    if "/" in unique:
        return "compound_division"
    if any(char in unique for char in "<>"):
        return "comparison"
    return "mixed"


def parse_basic_compare_diff(program: Any) -> dict[str, Any] | None:
    normalized = normalize_space(program)
    pattern = re.compile(
        rf"^(?P<a>{NUMBER_PATTERN})(?P<cmp>>=|<=|>|<)(?P<b>{NUMBER_PATTERN});?差值:(?P<x>{NUMBER_PATTERN})(?P<op>[+\-*/])(?P<y>{NUMBER_PATTERN})$"
    )
    match = pattern.fullmatch(normalized)
    if not match:
        return None
    return {
        "cmp": match.group("cmp"),
        "compare_numbers": [Decimal(match.group("a")), Decimal(match.group("b"))],
        "diff_operator": match.group("op"),
        "diff_numbers": [Decimal(match.group("x")), Decimal(match.group("y"))],
    }


def compare_basic_program(predicted: Any, standard: Any) -> ProgramScore:
    pred_text = normalize_space(predicted)
    std_text = normalize_space(standard)
    if not pred_text:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if pred_text == std_text:
        return ProgramScore(True)

    pred_compare = parse_basic_compare_diff(pred_text)
    std_compare = parse_basic_compare_diff(std_text)
    if pred_compare or std_compare:
        if pred_compare is None or std_compare is None:
            return ProgramScore(False, REASON_OPERATION)
        if pred_compare["cmp"] != std_compare["cmp"]:
            return ProgramScore(False, REASON_SORT_LOGIC)
        if pred_compare["diff_operator"] != std_compare["diff_operator"]:
            return ProgramScore(False, REASON_OPERATION)
        if pred_compare["compare_numbers"] != std_compare["compare_numbers"]:
            return ProgramScore(False, REASON_VALUE)
        if pred_compare["diff_numbers"] != std_compare["diff_numbers"]:
            return ProgramScore(False, REASON_VALUE)
        return ProgramScore(True)

    pred_family = basic_operator_family(pred_text)
    std_family = basic_operator_family(std_text)
    if pred_family != std_family:
        return ProgramScore(False, REASON_OPERATION)

    pred_numbers = extract_numbers(pred_text)
    std_numbers = extract_numbers(std_text)
    if pred_numbers == std_numbers:
        return ProgramScore(True)
    return ProgramScore(False, classify_number_count(pred_numbers, std_numbers))


def parse_average_program(program: Any) -> list[dict[str, Any]] | None:
    normalized = normalize_space(program)
    if not normalized:
        return None
    blocks = split_top_level(normalized, ",")
    formulas: list[dict[str, Any]] = []
    for block in blocks:
        slash_index = find_last_top_level(block, "/")
        if slash_index == -1:
            return None
        numerator = block[:slash_index]
        denominator = parse_decimal_text(block[slash_index + 1 :])
        if denominator is None:
            return None
        terms = extract_numbers(numerator)
        if not terms:
            return None
        formulas.append({"terms": terms, "denominator": denominator})
    return formulas


def compare_average_program(predicted: Any, standard: Any) -> ProgramScore:
    pred_text = normalize_space(predicted)
    std_text = normalize_space(standard)
    if not pred_text:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if pred_text == std_text:
        return ProgramScore(True)

    pred_formulas = parse_average_program(pred_text)
    std_formulas = parse_average_program(std_text)
    if pred_formulas is None:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if std_formulas is None:
        return ProgramScore(pred_text == std_text, "" if pred_text == std_text else REASON_VALUE)

    if len(pred_formulas) < len(std_formulas):
        return ProgramScore(False, REASON_MISSING)
    if len(pred_formulas) > len(std_formulas):
        return ProgramScore(False, REASON_REDUNDANT)

    for pred_formula, std_formula in zip(pred_formulas, std_formulas):
        if pred_formula["denominator"] != std_formula["denominator"]:
            return ProgramScore(False, REASON_VALUE)
        pred_terms = sorted(pred_formula["terms"])
        std_terms = sorted(std_formula["terms"])
        if len(pred_terms) < len(std_terms):
            return ProgramScore(False, REASON_MISSING)
        if len(pred_terms) > len(std_terms):
            return ProgramScore(False, REASON_REDUNDANT)
        if pred_terms != std_terms:
            return ProgramScore(False, REASON_VALUE)
    return ProgramScore(True)


def parse_count_term(term: str) -> dict[str, Any] | None:
    comparators = list(re.finditer(r">=|<=|>|<|=", term))
    if not comparators:
        return None
    comparator = comparators[-1]
    left = term[: comparator.start()]
    right = term[comparator.end() :]
    right_numbers = extract_numbers(right)
    if not right_numbers:
        return None
    return {
        "operator": comparator.group(),
        "left_numbers": extract_numbers(left),
        "right_numbers": right_numbers,
    }


def parse_count_program(program: Any) -> dict[str, Any] | None:
    normalized = normalize_space(program)
    if not normalized or "->" not in normalized:
        return None
    parts = [part for part in normalized.split("->") if part]
    if len(parts) < 2:
        return None
    terms = [part for part in split_top_level(parts[0], "+") if part]
    parsed_terms: list[dict[str, Any]] = []
    for term in terms:
        parsed = parse_count_term(term)
        if parsed is None:
            return None
        parsed_terms.append(parsed)
    bool_values = [int(match) for match in re.findall(r"(?<!\d)(?:0|1)(?!\d)", parts[-1])]
    return {"terms": parsed_terms, "bool_values": bool_values}


def compare_count_program(predicted: Any, standard: Any) -> ProgramScore:
    pred_text = normalize_space(predicted)
    std_text = normalize_space(standard)
    if not pred_text:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if pred_text == std_text:
        return ProgramScore(True)

    pred_program = parse_count_program(pred_text)
    std_program = parse_count_program(std_text)
    if pred_program is None:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if std_program is None:
        return ProgramScore(pred_text == std_text, "" if pred_text == std_text else REASON_VALUE)

    pred_terms = pred_program["terms"]
    std_terms = std_program["terms"]
    if len(pred_terms) < len(std_terms):
        return ProgramScore(False, REASON_MISSING)
    if len(pred_terms) > len(std_terms):
        return ProgramScore(False, REASON_REDUNDANT)

    for pred_term, std_term in zip(pred_terms, std_terms):
        if pred_term["operator"] != std_term["operator"]:
            return ProgramScore(False, REASON_SORT_LOGIC)
        if pred_term["right_numbers"] != std_term["right_numbers"]:
            return ProgramScore(False, REASON_SORT_LOGIC)
        pred_numbers = pred_term["left_numbers"]
        std_numbers = std_term["left_numbers"]
        if len(pred_numbers) < len(std_numbers):
            return ProgramScore(False, REASON_MISSING)
        if len(pred_numbers) > len(std_numbers):
            return ProgramScore(False, REASON_REDUNDANT)
        if pred_numbers != std_numbers:
            return ProgramScore(False, REASON_VALUE)

    pred_bools = pred_program["bool_values"]
    std_bools = std_program["bool_values"]
    if std_bools and pred_bools != std_bools:
        return ProgramScore(False, REASON_SORT_LOGIC)
    return ProgramScore(True)


def split_label_value(text: str) -> tuple[str, Decimal | None] | None:
    if ":" not in text:
        return None
    label, value = text.split(":", 1)
    return normalize_label(label), parse_decimal_text(value)


def pair_candidates_by_label(
    predicted: list[tuple[str, Decimal | None]],
    standard: list[tuple[str, Decimal | None]],
    task: str,
) -> list[tuple[tuple[str, Decimal | None], tuple[str, Decimal | None]]] | None:
    remaining = predicted.copy()
    pairs: list[tuple[tuple[str, Decimal | None], tuple[str, Decimal | None]]] = []
    for std_item in standard:
        matched_index: int | None = None
        for index, pred_item in enumerate(remaining):
            if labels_equivalent(pred_item[0], std_item[0], task):
                matched_index = index
                break
        if matched_index is None:
            return None
        pairs.append((remaining.pop(matched_index), std_item))
    return pairs


def parse_extremum_program(program: Any) -> dict[str, Any] | None:
    normalized = normalize_space(program)
    if not normalized or "->" not in normalized:
        return None
    parts = [part for part in normalized.split("->") if part]
    match = re.fullmatch(r"(MAX|MIN)\((.*)\)", parts[0])
    if not match:
        return None
    candidates: list[tuple[str, Decimal | None]] = []
    for raw_pair in split_top_level(match.group(2), ","):
        parsed_pair = split_label_value(raw_pair)
        if parsed_pair is None:
            return None
        candidates.append(parsed_pair)
    winner = split_label_value(parts[-1])
    if winner is None:
        return None
    return {"mode": match.group(1), "candidates": candidates, "winner": winner}


def compare_extremum_program(predicted: Any, standard: Any) -> ProgramScore:
    pred_text = normalize_space(predicted)
    std_text = normalize_space(standard)
    if not pred_text:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if pred_text == std_text:
        return ProgramScore(True)

    pred_program = parse_extremum_program(pred_text)
    std_program = parse_extremum_program(std_text)
    if pred_program is not None and std_program is not None:
        if pred_program["mode"] != std_program["mode"]:
            return ProgramScore(False, REASON_SORT_LOGIC)
        pred_candidates = pred_program["candidates"]
        std_candidates = std_program["candidates"]
        if len(pred_candidates) < len(std_candidates):
            return ProgramScore(False, REASON_MISSING)
        if len(pred_candidates) > len(std_candidates):
            return ProgramScore(False, REASON_REDUNDANT)
        candidate_pairs = pair_candidates_by_label(pred_candidates, std_candidates, "extremum")
        if candidate_pairs is None:
            return ProgramScore(False, REASON_VALUE)
        for pred_pair, std_pair in candidate_pairs:
            if pred_pair[1] != std_pair[1]:
                return ProgramScore(False, REASON_VALUE)
        pred_winner_label, pred_winner_value = pred_program["winner"]
        std_winner_label, std_winner_value = std_program["winner"]
        if not labels_equivalent(pred_winner_label, std_winner_label, "extremum"):
            return ProgramScore(False, REASON_SORT_LOGIC)
        if pred_winner_value != std_winner_value:
            return ProgramScore(False, REASON_VALUE)
        return ProgramScore(True)

    pred_parts = [part for part in pred_text.split("->") if part]
    std_parts = [part for part in std_text.split("->") if part]
    if not pred_parts:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if pred_parts and std_parts:
        pred_tail = split_label_value(pred_parts[-1])
        std_tail = split_label_value(std_parts[-1])
        if pred_tail and std_tail and not labels_equivalent(pred_tail[0], std_tail[0], "extremum"):
            return ProgramScore(False, REASON_SORT_LOGIC)
    pred_numbers = extract_numbers(pred_text)
    std_numbers = extract_numbers(std_text)
    if pred_numbers == std_numbers:
        return ProgramScore(True)
    return ProgramScore(False, classify_number_count(pred_numbers, std_numbers))


def parse_top_k_program(program: Any) -> dict[str, Any] | None:
    normalized = normalize_space(program)
    if not normalized or "->" not in normalized:
        return None
    parts = [part for part in normalized.split("->") if part]
    if len(parts) < 3:
        return None
    match = re.fullmatch(r"Top(?P<k>\d+)-(?P<mode>High|Low)\((?P<body>.*)\)", parts[0])
    if not match:
        return None
    candidates: list[tuple[str, Decimal | None]] = []
    for raw_pair in split_top_level(match.group("body"), ","):
        parsed_pair = split_label_value(raw_pair)
        if parsed_pair is None:
            return None
        candidates.append(parsed_pair)
    ranking_block = parts[1].strip("[]")
    result_block = parts[-1].strip("[]")
    rank_separator = ">" if ">" in ranking_block else "<"
    ranking = [normalize_label(part) for part in ranking_block.split(rank_separator) if part]
    results = [normalize_label(part) for part in split_top_level(result_block, ",")]
    return {
        "k": int(match.group("k")),
        "mode": match.group("mode"),
        "candidates": candidates,
        "ranking": ranking,
        "results": results,
    }


def build_candidate_lookup(
    candidates: list[tuple[str, Decimal | None]],
    task: str,
) -> dict[str, Decimal | None]:
    lookup: dict[str, Decimal | None] = {}
    for label, value in candidates:
        for variant in build_label_variants(label, task) or {normalize_label(label)}:
            lookup.setdefault(variant, value)
    return lookup


def resolve_token_values(
    tokens: list[str],
    candidates: list[tuple[str, Decimal | None]],
    task: str,
) -> list[Decimal | None] | None:
    lookup = build_candidate_lookup(candidates, task)
    resolved: list[Decimal | None] = []
    for token in tokens:
        token_variants = build_label_variants(token, task) or {normalize_label(token)}
        matched_value: Decimal | None = None
        found = False
        for variant in token_variants:
            if variant in lookup:
                matched_value = lookup[variant]
                found = True
                break
        if not found:
            return None
        resolved.append(matched_value)
    return resolved


def compare_top_k_program(predicted: Any, standard: Any, standard_gold: Any = "") -> ProgramScore:
    pred_text = normalize_space(predicted)
    std_text = normalize_space(standard)
    if not pred_text:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if pred_text == std_text:
        return ProgramScore(True)

    pred_program = parse_top_k_program(pred_text)
    std_program = parse_top_k_program(std_text)
    if pred_program is None:
        return ProgramScore(False, REASON_OUTPUT_FORMAT, True)
    if std_program is None:
        return ProgramScore(pred_text == std_text, "" if pred_text == std_text else REASON_VALUE)

    if pred_program["k"] != std_program["k"] or pred_program["mode"] != std_program["mode"]:
        return ProgramScore(False, REASON_SORT_LOGIC)

    pred_candidates = pred_program["candidates"]
    std_candidates = std_program["candidates"]
    if len(pred_candidates) < len(std_candidates):
        return ProgramScore(False, REASON_MISSING)
    if len(pred_candidates) > len(std_candidates):
        return ProgramScore(False, REASON_REDUNDANT)

    pred_candidate_values = sorted(value for _, value in pred_candidates if value is not None)
    std_candidate_values = sorted(value for _, value in std_candidates if value is not None)
    if len(pred_candidate_values) != len(pred_candidates):
        return ProgramScore(False, REASON_VALUE)
    if len(std_candidate_values) != len(std_candidates):
        return ProgramScore(False, REASON_VALUE)
    if pred_candidate_values != std_candidate_values:
        return ProgramScore(False, REASON_VALUE)

    pred_ranking_values = resolve_token_values(pred_program["ranking"], pred_candidates, "top_k")
    std_ranking_values = resolve_token_values(std_program["ranking"], std_candidates, "top_k")
    pred_result_values = resolve_token_values(pred_program["results"], pred_candidates, "top_k")
    std_result_values = resolve_token_values(std_program["results"], std_candidates, "top_k")

    if (
        pred_ranking_values is None
        or std_ranking_values is None
        or std_result_values is None
    ):
        return ProgramScore(False, REASON_SORT_LOGIC)

    if len(pred_ranking_values) != len(pred_candidates):
        return ProgramScore(False, REASON_SORT_LOGIC)

    if pred_ranking_values != std_ranking_values:
        return ProgramScore(False, REASON_SORT_LOGIC)

    if pred_result_values is not None:
        if len(pred_result_values) != pred_program["k"]:
            return ProgramScore(False, REASON_SORT_LOGIC)
        if pred_result_values != std_result_values:
            return ProgramScore(False, REASON_SORT_LOGIC)
        return ProgramScore(True)

    standard_gold_parts = split_csv_like(standard_gold)
    if (
        len(pred_program["results"]) == pred_program["k"]
        and standard_gold_parts
        and ordered_labels_equal(pred_program["results"], standard_gold_parts, "top_k")
    ):
        return ProgramScore(True)

    if len(pred_program["results"]) != pred_program["k"]:
        return ProgramScore(False, REASON_SORT_LOGIC)
    if pred_result_values != std_result_values:
        return ProgramScore(False, REASON_SORT_LOGIC)
    return ProgramScore(True)


def compare_program(
    task: str,
    predicted: Any,
    standard: Any,
    standard_gold: Any = "",
) -> ProgramScore:
    if task == "basic_arithmetic":
        return compare_basic_program(predicted, standard)
    if task == "average":
        return compare_average_program(predicted, standard)
    if task == "count":
        return compare_count_program(predicted, standard)
    if task == "extremum":
        return compare_extremum_program(predicted, standard)
    if task == "top_k":
        return compare_top_k_program(predicted, standard, standard_gold)
    raise ValueError(f"Unsupported task type: {task}")


def classify_basic_result_error(predicted: Any, standard: Any) -> str:
    pred_text = normalize_space(predicted)
    std_text = normalize_space(standard)
    if not pred_text:
        return REASON_RESULT_EXTRACTION

    pred_parts = split_top_level(pred_text, ",")
    std_parts = split_top_level(std_text, ",")
    if len(std_parts) == 2:
        if len(pred_parts) != 2:
            return REASON_RESULT_EXTRACTION
        if not labels_equivalent(pred_parts[0], std_parts[0], "basic_arithmetic"):
            return REASON_RESULT_EXTRACTION
        std_percent = parse_percent_text(std_parts[1])
        if std_percent is not None:
            return REASON_CALC_EXECUTION if parse_percent_text(pred_parts[1]) is not None else REASON_RESULT_EXTRACTION
        std_number = parse_decimal_text(std_parts[1])
        if std_number is not None:
            return REASON_CALC_EXECUTION if parse_decimal_text(pred_parts[1]) is not None else REASON_RESULT_EXTRACTION
        return REASON_RESULT_EXTRACTION

    std_percent = parse_percent_text(std_text)
    if std_percent is not None:
        return REASON_CALC_EXECUTION if parse_percent_text(pred_text) is not None else REASON_RESULT_EXTRACTION

    std_number = parse_decimal_text(std_text)
    if std_number is not None:
        return REASON_CALC_EXECUTION if parse_decimal_text(pred_text) is not None else REASON_RESULT_EXTRACTION
    return REASON_RESULT_EXTRACTION


def classify_average_result_error(predicted: Any, standard: Any) -> str:
    pred_parts = split_csv_like(predicted)
    std_parts = split_csv_like(standard)
    if not pred_parts or len(pred_parts) != len(std_parts):
        return REASON_RESULT_EXTRACTION
    for pred_part in pred_parts:
        if parse_decimal_text(pred_part) is None:
            return REASON_RESULT_EXTRACTION
    return REASON_CALC_EXECUTION


def classify_count_result_error(predicted: Any) -> str:
    return REASON_CALC_EXECUTION if parse_decimal_text(predicted) is not None else REASON_RESULT_EXTRACTION


def classify_extremum_result_error(predicted: Any) -> str:
    return REASON_RESULT_EXTRACTION


def classify_top_k_result_error(predicted: Any, standard: Any) -> str:
    pred_parts = split_csv_like(predicted)
    std_parts = split_csv_like(standard)
    if not pred_parts or len(pred_parts) != len(std_parts):
        return REASON_RESULT_EXTRACTION
    return REASON_RESULT_EXTRACTION


def compress_program_reason(detail_reason: str, format_error: bool = False) -> str:
    if format_error or detail_reason == REASON_OUTPUT_FORMAT:
        return REASON_STRUCTURE
    if detail_reason in {REASON_MISSING, REASON_REDUNDANT, REASON_VALUE}:
        return REASON_TABLE_LOOKUP
    if detail_reason in {REASON_SORT_LOGIC, REASON_OPERATION}:
        return REASON_LOGIC
    return REASON_TABLE_LOOKUP


def compress_result_reason(detail_reason: str) -> str:
    if detail_reason == REASON_CALC_EXECUTION:
        return REASON_CALCULATION
    return REASON_RESULT_ORGANIZATION


def determine_reason(
    task: str,
    prediction: dict[str, Any],
    standard_item: dict[str, Any],
    result_correct: int,
    step3_correct: bool,
    program_score: ProgramScore,
) -> str:
    if not program_score.correct:
        return compress_program_reason(program_score.reason, program_score.format_error)
    if not step3_correct:
        return REASON_CALCULATION
    if result_correct == 1:
        return ""
    return REASON_RESULT_ORGANIZATION


def clear_score_fields(prediction: dict[str, Any]) -> None:
    prediction["result_correct"] = ""
    prediction["program_correct"] = ""
    prediction["reason"] = ""


def write_score_fields(
    prediction: dict[str, Any],
    result_correct: int,
    program_score: ProgramScore,
    reason: str,
) -> None:
    prediction["result_correct"] = result_correct
    prediction["program_correct"] = int(program_score.correct)
    prediction["reason"] = reason


def score_prediction_file(
    prediction_path: Path,
    task: str | None = None,
    standard_path: Path | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    resolved_task = task or detect_task_from_name(prediction_path)
    resolved_standard = standard_path or TASK_TO_STANDARD[resolved_task]

    predictions = load_json_list(prediction_path)
    standard_items = load_json_list(resolved_standard)
    standard_index = build_standard_index(standard_items)
    standard_valid_count = len(standard_index)

    evaluated_count = min(len(predictions), max_items) if max_items is not None else len(predictions)
    skipped = 0
    aligned = 0
    result_correct_count = 0
    program_correct_count = 0
    reason_counter: Counter[str] = Counter()

    for index, prediction in enumerate(predictions):
        if max_items is not None and index >= max_items:
            break

        key = make_align_key(prediction)
        standard_item = standard_index.get(key)
        if standard_item is None:
            skipped += 1
            clear_score_fields(prediction)
            continue

        pred_table_name = normalize_label(prediction.get("table_name", ""))
        std_table_name = normalize_label(standard_item.get("table_name", ""))
        if pred_table_name and std_table_name and pred_table_name != std_table_name:
            skipped += 1
            clear_score_fields(prediction)
            continue

        aligned += 1
        step3_correct = compare_result(
            resolved_task,
            prediction.get("gold_answer", ""),
            standard_item.get("gold_answer", ""),
        )
        answer_value_correct = compare_result(
            resolved_task,
            prediction.get("answer", ""),
            standard_item.get("gold_answer", ""),
        )
        unit_correct = compare_unit(
            prediction.get("unit", ""),
            standard_item.get("gold_unit", ""),
        )
        result_correct = int(answer_value_correct and unit_correct)
        program_score = compare_program(
            resolved_task,
            prediction.get("program", ""),
            standard_item.get("program", ""),
            standard_item.get("gold_answer", ""),
        )
        reason = determine_reason(
            resolved_task,
            prediction,
            standard_item,
            result_correct,
            step3_correct,
            program_score,
        )
        write_score_fields(prediction, result_correct, program_score, reason)

        if result_correct == 1:
            result_correct_count += 1
        if program_score.correct:
            program_correct_count += 1
        if reason:
            reason_counter[reason] += 1

    with prediction_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)

    return {
        "task": resolved_task,
        "prediction_file": str(prediction_path),
        "standard_file": str(resolved_standard),
        "standard_valid_samples": standard_valid_count,
        "prediction_samples_total": len(predictions),
        "prediction_samples_evaluated": evaluated_count,
        "aligned": aligned,
        "skipped": skipped,
        "result_correct_count": result_correct_count,
        "program_correct_count": program_correct_count,
        "reason_distribution": dict(reason_counter),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score prediction files against human_fixed standard datasets.")
    parser.add_argument("prediction_files", nargs="+", help="One or more prediction result files to score in place.")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_TO_STANDARD.keys()),
        help="Override task type. Recommended only when a single prediction file is passed.",
    )
    parser.add_argument(
        "--standard",
        help="Override the standard dataset path. Recommended only when a single prediction file is passed.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Only score the first N prediction samples in each file for quick testing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.task or args.standard) and len(args.prediction_files) != 1:
        parser.error("--task and --standard can only be used when a single prediction file is provided.")

    overall_result_correct = 0
    overall_program_correct = 0
    overall_aligned = 0
    overall_skipped = 0
    overall_reason_counter: Counter[str] = Counter()

    for raw_prediction_path in args.prediction_files:
        prediction_path = Path(raw_prediction_path)
        if not prediction_path.is_absolute():
            prediction_path = (ROOT_DIR / prediction_path).resolve()
        if not prediction_path.exists():
            print(f"[ERROR] Prediction file not found: {prediction_path}", file=sys.stderr)
            return 1

        standard_path = None
        if args.standard:
            standard_path = Path(args.standard)
            if not standard_path.is_absolute():
                standard_path = (ROOT_DIR / standard_path).resolve()

        try:
            summary = score_prediction_file(
                prediction_path=prediction_path,
                task=args.task,
                standard_path=standard_path,
                max_items=args.max_items,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to score {prediction_path}: {exc}", file=sys.stderr)
            return 1

        overall_result_correct += summary["result_correct_count"]
        overall_program_correct += summary["program_correct_count"]
        overall_aligned += summary["aligned"]
        overall_skipped += summary["skipped"]
        overall_reason_counter.update(summary["reason_distribution"])

        print(f"[Summary] {summary['prediction_file']}")
        print(f"  task: {summary['task']}")
        print(f"  standard_valid_samples: {summary['standard_valid_samples']}")
        print(f"  prediction_samples_total: {summary['prediction_samples_total']}")
        print(f"  prediction_samples_evaluated: {summary['prediction_samples_evaluated']}")
        print(f"  aligned: {summary['aligned']}")
        print(f"  skipped: {summary['skipped']}")
        print(f"  result_correct_count: {summary['result_correct_count']}")
        print(f"  program_correct_count: {summary['program_correct_count']}")
        print(f"  reason_distribution: {summary['reason_distribution']}")
        print(f"  updated_file: {summary['prediction_file']}")

    print("[Overall]")
    print(f"  aligned: {overall_aligned}")
    print(f"  skipped: {overall_skipped}")
    print(f"  result_correct_count: {overall_result_correct}")
    print(f"  program_correct_count: {overall_program_correct}")
    print(f"  reason_distribution: {dict(overall_reason_counter)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
