"""Tool-calling ReAct node for numerical reasoning."""

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from eco_agent.context import Context
from eco_agent.prompts import NUMERICAL_AGENT_PROMPT
from eco_agent.state import State
from eco_agent.tools import TOOLS
from eco_agent.utils import load_chat_model

MAX_REACT_ITERATIONS = 10
MAX_MODEL_ATTEMPTS = 3
MAX_TRANSIENT_TOOL_RETRIES = 2
MAX_TOOL_FAILURES = 3
MAX_SAME_CALL_FAILURES = 2
_RETRY_BASE_DELAY_SECONDS = 0.5

_ERROR_RECOVERY_PROMPT = """

【工具错误恢复规则】
- 工具失败时会返回包含 success、error_type、retryable 和 retry_strategy 的 JSON。
- retryable=false 时，禁止用相同方式重试；应切换工具、使用已有证据回答或明确说明数据不足。
- retryable=true 时，必须先根据 error 和 retry_strategy 修正参数，禁止原样重复失败的调用。
- SQL 执行失败时，先核对准确表名和列名，再调用对应的 SQL 生成工具，并把上一次错误完整传入 sql_error。
- table_not_found 时，SQLite 数据先调用 create_table_from_markdown/get_temp_table_info；不要猜测表名。
- 达到某工具的失败上限后，停止调用该工具，不得编造工具结果。
"""

# 工具名 → 工具对象快速查找
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def _json_dumps(value: Any) -> str:
    """稳定序列化工具参数和错误消息。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _parse_tool_result(result: Any) -> tuple[bool, Any, str]:
    """将不同工具的返回值统一为（是否成功、解析值、原始文本）。"""
    if isinstance(result, str):
        raw_content = result
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            # generate_sql 等工具成功时直接返回 SQL 字符串。
            return True, result, raw_content
    else:
        parsed = result
        raw_content = _json_dumps(result)

    if isinstance(parsed, dict) and parsed.get("success") is False:
        return False, parsed, raw_content
    return True, parsed, raw_content


def _classify_error(error: str, *, unknown_tool: bool = False) -> dict[str, Any]:
    """把工具错误分类，以便选择自动重试、模型纠错或直接熔断。"""
    lowered = error.lower()

    if unknown_tool:
        return {
            "error_type": "unknown_tool",
            "retryable": False,
            "transient": False,
            "retry_strategy": "从已提供的工具列表中重新选择工具。",
        }
    if any(word in lowered for word in ("禁止", "forbidden", "not allowed")):
        return {
            "error_type": "forbidden_operation",
            "retryable": False,
            "transient": False,
            "retry_strategy": "不要重复危险操作，改用只读查询或其他安全方案。",
        }
    if any(
        word in lowered
        for word in (
            "timeout",
            "timed out",
            "429",
            "rate limit",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "database is locked",
            "服务暂时不可用",
            "连接失败",
        )
    ):
        return {
            "error_type": "transient_error",
            "retryable": True,
            "transient": True,
            "retry_strategy": "保持参数不变，短暂等待后重试。",
        }
    if any(word in lowered for word in ("sql", "column", "列名", "syntax")):
        return {
            "error_type": "sql_error",
            "retryable": True,
            "transient": False,
            "retry_strategy": (
                "检查表结构和列名；调用 SQL 生成工具时，将本次错误通过 "
                "sql_error 参数传入后重新生成 SQL。"
            ),
        }
    if any(word in lowered for word in ("表 '", "table ", "表名", "不存在")):
        return {
            "error_type": "table_not_found",
            "retryable": True,
            "transient": False,
            "retry_strategy": "先创建或查询可用表，再使用工具返回的准确表名。",
        }
    if any(
        word in lowered
        for word in (
            "validation",
            "argument",
            "参数",
            "expression",
            "unsupported",
            "除数",
            "zero",
            "must not be zero",
            "解析",
        )
    ):
        return {
            "error_type": "invalid_arguments",
            "retryable": True,
            "transient": False,
            "retry_strategy": "根据错误信息修正参数，禁止使用完全相同的参数重试。",
        }
    return {
        "error_type": "tool_error",
        "retryable": True,
        "transient": False,
        "retry_strategy": "检查工具输入并调整方案；不要原样重复失败的调用。",
    }


def _error_message(
    *,
    tool_name: str,
    error: str,
    classification: dict[str, Any],
    attempt: int,
    max_attempts: int,
) -> str:
    """创建供 ReAct 模型读取的统一结构化错误。"""
    return _json_dumps(
        {
            "success": False,
            "tool": tool_name,
            "error_type": classification["error_type"],
            "error": error,
            "retryable": classification["retryable"],
            "retry_strategy": classification["retry_strategy"],
            "attempt": attempt,
            "max_attempts": max_attempts,
        }
    )


async def _invoke_model_with_retry(model: Any, messages: list[Any]) -> Any:
    """对模型服务的短暂故障执行有限次数的指数退避重试。"""
    last_error: Exception | None = None
    for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
        try:
            return await model.ainvoke(messages)
        except Exception as error:
            last_error = error
            error_text = f"{type(error).__name__}: {error}"
            is_transient = isinstance(error, (TimeoutError, ConnectionError)) or (
                _classify_error(error_text)["transient"]
            )
            if is_transient and attempt < MAX_MODEL_ATTEMPTS:
                await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
                continue
            raise
    assert last_error is not None
    raise last_error


async def _invoke_tool_with_retry(tool: Any, args: dict[str, Any]) -> dict[str, Any]:
    """执行工具；仅对明确的瞬时错误自动使用原参数重试。"""
    max_attempts = MAX_TRANSIENT_TOOL_RETRIES + 1

    for attempt in range(1, max_attempts + 1):
        try:
            result = await tool.ainvoke(args)
            success, parsed, raw_content = _parse_tool_result(result)
            if success:
                return {
                    "success": True,
                    "content": raw_content,
                    "parsed": parsed,
                    "attempt": attempt,
                }
            error = str(parsed.get("error", "工具返回失败，但未提供错误信息"))
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"

        classification = _classify_error(error)
        if classification["transient"] and attempt < max_attempts:
            await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
            continue

        return {
            "success": False,
            "content": _error_message(
                tool_name=tool.name,
                error=error,
                classification=classification,
                attempt=attempt,
                max_attempts=max_attempts if classification["transient"] else 1,
            ),
            "error": error,
            "classification": classification,
            "attempt": attempt,
        }

    raise RuntimeError("工具重试循环异常结束")


async def numerical_node(state: State, runtime: Runtime[Context]):
    """ReAct 子智能体：LLM 自主选择工具，逐步推理，直到得出最终答案。"""

    model = load_chat_model(runtime.context.model)
    model_with_tools = model.bind_tools(TOOLS)

    # 构建初始消息
    question = state["question"]
    docs = state.get("retrieved_docs", "")
    tables = state.get("retrieved_tables", "")
    table_source = state.get("table_source", "")

    # 根据 table_source 选择 SQL 工具提示
    if table_source == "meilisearch":
        sql_tool_hint = (
            "\n【表格数据来源：Meilisearch 检索 → Markdown 格式】\n"
            "这些表格是内存中的临时数据，**不是 MySQL 数据库表**。\n"
            "查询这些表格时，必须使用以下 SQLite 工具：\n"
            "1. create_table_from_markdown —— 将 Markdown 表格加载到内存\n"
            "2. get_temp_table_info —— 查看表结构（列名、类型）\n"
            "3. generate_sqlite_sql —— 生成 SQLite 查询语句\n"
            "4. execute_sqlite —— 执行 SQLite 查询\n"
            "**禁止使用 generate_sql / execute_sql（那是 MySQL 工具，仅用于真实数据库表）。**"
        )
    else:
        sql_tool_hint = (
            "\n【表格数据来源：MySQL 数据库表】\n"
            "查询这些表时，使用以下工具：\n"
            "1. generate_sql —— 生成 MySQL 查询语句\n"
            "2. execute_sql —— 执行 MySQL 查询\n"
            "**禁止使用 create_table_from_markdown / execute_sqlite / execute_sqlite / get_temp_table_info（那是 SQLite 内存表工具）。**"
        )

    messages = [
        SystemMessage(content=NUMERICAL_AGENT_PROMPT + _ERROR_RECOVERY_PROMPT),
        HumanMessage(content=f"""用户问题：{question}

检索到的参考文档：
{docs}

检索到的参考表格：
{tables}

表格数据来源（决定使用哪种SQL工具）：
{table_source}
{sql_tool_hint}

请逐步分析并回答。"""),
    ]

    tool_trace: list[dict[str, Any]] = list(state.get("tool_trace") or [])
    tool_failure_counts: dict[str, int] = {}
    same_call_failures: dict[str, int] = {}
    total_tool_failures = int(state.get("tool_retry_count") or 0)

    # ReAct 循环
    for react_round in range(1, MAX_REACT_ITERATIONS + 1):
        try:
            response = await _invoke_model_with_retry(model_with_tools, messages)
        except Exception as error:
            final_answer = f"数值推理模型调用失败：{type(error).__name__}: {error}"
            return {
                "messages": messages,
                "final_answer": final_answer,
                "tool_retry_count": total_tool_failures,
                "tool_trace": tool_trace,
            }
        messages.append(response)

        if not response.tool_calls:
            # 无工具调用 → 最终答案
            return {
                "messages": messages,
                "final_answer": response.content,
                "tool_retry_count": total_tool_failures,
                "tool_trace": tool_trace,
            }

        # 执行每一个工具调用
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", f"missing-id-{react_round}")
            call_signature = f"{tool_name}:{_json_dumps(tool_args)}"
            tool = TOOLS_BY_NAME.get(tool_name)

            if tool is None:
                classification = _classify_error("未知工具", unknown_tool=True)
                content = _error_message(
                    tool_name=tool_name,
                    error=f"未知工具 {tool_name}。可用工具：{sorted(TOOLS_BY_NAME)}",
                    classification=classification,
                    attempt=1,
                    max_attempts=1,
                )
                outcome = {"success": False, "error_type": "unknown_tool"}
            elif tool_failure_counts.get(tool_name, 0) >= MAX_TOOL_FAILURES:
                classification = {
                    "error_type": "tool_circuit_open",
                    "retryable": False,
                    "transient": False,
                    "retry_strategy": "该工具本轮已多次失败，请改用其他工具或基于已有证据回答。",
                }
                content = _error_message(
                    tool_name=tool_name,
                    error=f"工具已累计失败 {MAX_TOOL_FAILURES} 次，已停止继续调用。",
                    classification=classification,
                    attempt=MAX_TOOL_FAILURES,
                    max_attempts=MAX_TOOL_FAILURES,
                )
                outcome = {"success": False, "error_type": "tool_circuit_open"}
            elif same_call_failures.get(call_signature, 0) >= MAX_SAME_CALL_FAILURES:
                classification = {
                    "error_type": "repeated_failed_call",
                    "retryable": False,
                    "transient": False,
                    "retry_strategy": "相同参数已经重复失败，请修改参数、切换工具或结束推理。",
                }
                content = _error_message(
                    tool_name=tool_name,
                    error="完全相同的工具参数已经连续失败，已阻止再次执行。",
                    classification=classification,
                    attempt=MAX_SAME_CALL_FAILURES,
                    max_attempts=MAX_SAME_CALL_FAILURES,
                )
                outcome = {"success": False, "error_type": "repeated_failed_call"}
            else:
                execution = await _invoke_tool_with_retry(tool, tool_args)
                content = execution["content"]
                outcome = {
                    "success": execution["success"],
                    "attempt": execution["attempt"],
                }
                if not execution["success"]:
                    classification = execution["classification"]
                    outcome["error_type"] = classification["error_type"]
                    outcome["error"] = execution["error"]
                    tool_failure_counts[tool_name] = (
                        tool_failure_counts.get(tool_name, 0) + 1
                    )
                    same_call_failures[call_signature] = (
                        same_call_failures.get(call_signature, 0) + 1
                    )
                    total_tool_failures += 1
                else:
                    # 成功后清除该调用的历史失败，避免陈旧错误触发熔断。
                    same_call_failures.pop(call_signature, None)

            messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
            tool_trace.append(
                {
                    "round": react_round,
                    "tool": tool_name,
                    "args": tool_args,
                    **outcome,
                }
            )

    # 达到最大迭代次数仍未结束
    fallback_instruction = HumanMessage(
        content=(
            "工具调用轮数已达到上限。请停止调用工具，根据以上已经成功获得的结果，"
            "给出尽可能可靠的最终答案；若证据不足，请明确说明已确认的信息、失败步骤"
            "以及还缺少什么，不要编造数值。"
        )
    )
    try:
        fallback_response = await _invoke_model_with_retry(
            model, [*messages, fallback_instruction]
        )
        final_answer = fallback_response.content
        messages.extend([fallback_instruction, fallback_response])
    except Exception:
        final_answer = (
            "已达到最大推理步数限制，且无法生成降级答案。请检查工具执行记录后重试。"
        )

    return {
        "messages": messages,
        "final_answer": final_answer,
        "tool_retry_count": total_tool_failures,
        "tool_trace": tool_trace,
    }
