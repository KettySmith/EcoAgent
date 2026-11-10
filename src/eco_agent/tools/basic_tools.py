"""Small, deterministic tools for calculations commonly needed by an agent."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from typing import Any, Callable, List

from langchain_core.tools import tool


# Only the AST nodes listed here can be evaluated.  In particular, names, function
# calls, attributes and subscripts are deliberately not supported.
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_COMPARISON_OPERATORS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_MAX_EXPRESSION_LENGTH = 500
_MAX_ABSOLUTE_NUMBER = 1e100
_MAX_POWER_EXPONENT = 1000

# Convert a postfix percentage such as ``13%`` to ``(13 / 100)``.  The percent
# sign must touch the number; spaced ``5 % 2`` therefore remains modulo.
_PERCENT_LITERAL = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"
    r"(?=\s*(?:$|[+*/<>=!&|,)\]-]))"
)


def _normalise_expression(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression is too long (maximum {_MAX_EXPRESSION_LENGTH} characters)"
        )
    return _PERCENT_LITERAL.sub(r"(\1 / 100)", expression)


def _validate_number(value: Any) -> int | float:
    # bool is a subclass of int, but accepting True/False would be surprising in
    # a mathematical calculator.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only real numbers are supported")
    if not math.isfinite(value) or abs(value) > _MAX_ABSOLUTE_NUMBER:
        raise ValueError("Numeric result is too large or is not finite")
    return value


def safe_eval(node: ast.AST) -> int | float | bool:
    """Evaluate the safe numeric/comparison subset of a parsed Python expression."""
    if isinstance(node, ast.Constant):
        return _validate_number(node.value)

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _validate_number(safe_eval(node.left))
        right = _validate_number(safe_eval(node.right))
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise ValueError(
                f"Power exponent is too large (maximum {_MAX_POWER_EXPONENT})"
            )
        return _validate_number(operation(left, right))

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return _validate_number(operation(_validate_number(safe_eval(node.operand))))

    if isinstance(node, ast.Compare):
        # Python comparisons may be chained: 1 < 2 <= 3.
        left = _validate_number(safe_eval(node.left))
        for operator_node, comparator_node in zip(node.ops, node.comparators):
            operation = _COMPARISON_OPERATORS.get(type(operator_node))
            if operation is None:
                raise ValueError(
                    f"Unsupported comparison: {type(operator_node).__name__}"
                )
            right = _validate_number(safe_eval(comparator_node))
            if not operation(left, right):
                return False
            left = right
        return True

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def _success(**values: Any) -> str:
    return json.dumps({"success": True, **values}, ensure_ascii=False)


def _failure(error: Exception) -> str:
    return json.dumps({"success": False, "error": str(error)}, ensure_ascii=False)


@tool
def calculate(expression: str) -> str:
    """安全地计算基础算术表达式或数值比较表达式。

    支持的运算：
    - 算术运算：+、-、*、/、%、** 和括号；
    - 数值比较：<、<=、>、>=、==、!=，并支持连续比较；
    - 百分数字面量：例如 ``13% * 50`` 表示计算 50 的 13%。

    示例：``(100 + 200) * 3``、``74 > 76``、``5 % 2``、
    ``(120 - 100) / 100 * 100``、``15% * 240``。
    返回 JSON 字符串，其中包含 ``success``，以及 ``result`` 或 ``error``。
    """
    try:
        tree = ast.parse(_normalise_expression(expression), mode="eval")
        return _success(result=safe_eval(tree.body))
    except Exception as error:
        return _failure(error)


@tool
def calculate_ratio(value: float, reference: float) -> str:
    """计算一个数值相对于基准值的比率，即 ``value / reference``。

    适用于“30 与 120 的比率是多少”等问题，结果为 0.25。
    ``value`` 是待比较数值，``reference`` 是作为分母的基准值；基准值不能为 0。
    """
    try:
        numerator = _validate_number(value)
        denominator = _validate_number(reference)
        if denominator == 0:
            raise ValueError("Reference value must not be zero")
        return _success(result=_validate_number(numerator / denominator))
    except Exception as error:
        return _failure(error)


@tool
def calculate_percentage(part: float, total: float) -> str:
    """计算部分数值 ``part`` 占总数 ``total`` 的百分比。

    计算公式为 ``part / total * 100``。例如 part=30、total=120 时，
    返回 25，表示占比为 25%。总数 ``total`` 不能为 0。
    """
    try:
        numerator = _validate_number(part)
        denominator = _validate_number(total)
        if denominator == 0:
            raise ValueError("Total must not be zero")
        return _success(
            result=_validate_number(numerator / denominator * 100), unit="percent"
        )
    except Exception as error:
        return _failure(error)


@tool
def calculate_percentage_change(old_value: float, new_value: float) -> str:
    """计算数值从旧值变化到新值时的百分比增长率或下降率。

    计算公式为 ``(new_value - old_value) / abs(old_value) * 100``。
    正数表示增长，负数表示下降。例如从 80 变为 100 时返回 25，表示增长 25%。
    旧值 ``old_value`` 不能为 0。
    """
    try:
        old = _validate_number(old_value)
        new = _validate_number(new_value)
        if old == 0:
            raise ValueError("Old value must not be zero")
        return _success(
            result=_validate_number((new - old) / abs(old) * 100), unit="percent"
        )
    except Exception as error:
        return _failure(error)


BASIC_TOOLS: List[Callable[..., Any]] = [
    calculate,
    calculate_ratio,
    calculate_percentage,
    calculate_percentage_change,
]
