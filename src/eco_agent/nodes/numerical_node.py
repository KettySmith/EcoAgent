
# Tool Calling ReAct 
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from eco_agent.context import Context
from eco_agent.prompts import NUMERICAL_AGENT_PROMPT
from eco_agent.state import State
from eco_agent.tools import TOOLS
from eco_agent.utils import load_chat_model

MAX_REACT_ITERATIONS = 10

# 工具名 → 工具对象快速查找
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


async def numerical_node(state: State, runtime: Runtime[Context]):
    """ReAct 子智能体：LLM 自主选择工具，逐步推理，直到得出最终答案。"""

    model = load_chat_model(runtime.context.model)
    model_with_tools = model.bind_tools(TOOLS)

    # 构建初始消息
    question = state["question"]
    docs = state.get("retrieved_docs", "")
    tables = state.get("retrieved_tables", "")

    messages = [
        SystemMessage(content=NUMERICAL_AGENT_PROMPT),
        HumanMessage(content=f"""用户问题：{question}

检索到的参考文档：
{docs}

检索到的参考表格：
{tables}

请逐步分析并回答。"""),
    ]

    # ReAct 循环
    for _ in range(MAX_REACT_ITERATIONS):
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # 无工具调用 → 最终答案
            return {
                "messages": messages,
                "final_answer": response.content,
            }

        # 执行每一个工具调用
        for tc in response.tool_calls:
            tool = TOOLS_BY_NAME.get(tc["name"])
            if tool is not None:
                result = await tool.ainvoke(tc["args"])
            else:
                result = f"错误：未知工具 {tc['name']}"
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    # 达到最大迭代次数仍未结束
    return {
        "messages": messages,
        "final_answer": "达到最大推理步数限制，请简化问题重试。",
    }
