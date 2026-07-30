"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
import re
import json
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from eco_agent.context import Context
from eco_agent.state import InputState, State,QueryPlan
from eco_agent.tools import TOOLS
from eco_agent.utils import load_chat_model, QueryExecutor
from eco_agent.prompts import QUERY_PLAN_PROMPT, SQL_GENERATION_PROMPT, ROUTER_PROMPT, VERIFY_PROMPT
from eco_agent.nodes.rag_node import rag_node
from eco_agent.nodes.router_node import router_node
from eco_agent.nodes.cot_node import cot_node
from eco_agent.nodes.numerical_node import numerical_node
SQL_RETRY_LIMIT=1
TOOL_RETRY_LIMIT=1

### 暂时弃用)
def validate_query_plan(plan: QueryPlan):

    assert plan.query_type in [
        "lookup", "compare", "aggregation", "topk", "trend","other"
    ], "invalid query_type"

    assert plan.metric is not None, "metric required"

    # -------------------
    # aggregation 规则
    # -------------------
    if plan.query_type == "aggregation":
        assert plan.aggregation in ["avg", "sum", "max", "min"], \
            "aggregation type required for aggregation query"
        assert plan.limit is None, "aggregation should not have limit"

    # -------------------
    # topk 规则
    # -------------------
    if plan.query_type == "topk":
        assert plan.limit is not None, "topk requires limit"
        assert plan.order_by.get("direction") in ["asc", "desc"], \
            "order_by direction required"

    # -------------------
    # compare 规则
    # -------------------
    if plan.query_type == "compare":
        assert len(plan.compare_keys) > 0 or plan.filters.get("location"), \
            "compare requires compare_keys or location filters"
        
### 计划节点(暂时弃用)
async def plan_node(state: State, runtime: Runtime[Context]):

    model = load_chat_model(runtime.context.model)
    question = state["question"]
    prompt = QUERY_PLAN_PROMPT.format(question=question)

    response = await model.ainvoke(prompt)
    text = response.content


    try:
        json_text = re.search(r"\{.*\}", text, re.S).group()
        data = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"QueryPlan parse failed: {text}") from e

    # 2. 转为 dataclass
    plan = QueryPlan(**data)

    # 3. 校验（关键）
    validate_query_plan(plan)

    return {
        "query_plan": plan
    }


    


### 验证节点
async def verify_node(state: State, runtime: Runtime[Context]):


    # 1. SQL失败处理
    if not state.get("sql_success") and state["sql_retry_count"] > SQL_RETRY_LIMIT:
        return {
            "verify": "answer"
        }
    # 避免无需sql，success默认为true
    if (not state.get("sql_success") or len(state.get("sql_result", [])) == 0) and state["sql_retry_count"] <= SQL_RETRY_LIMIT:
        return {
            "verify": "retry_sql",
            "sql_retry_count": state["sql_retry_count"] + 1
        }
    
    # 2. Tool失败处理
    tool_result = ""
    tool_data={}
    last_msg = state["messages"][-1]
    if isinstance(last_msg, ToolMessage):
        tool_result = last_msg.content
        try:
            tool_data = json.loads(tool_result)

            # tool调用失败
            if not tool_data.get("success", False):

                # 超过重试次数
                if state.get("tool_retry_count", 0) > TOOL_RETRY_LIMIT:
                    return {
                        "verify": "answer"
                    }

                # 继续重试tool
                return {
                    "verify": "tool",
                    "tool_retry_count":
                        state.get("tool_retry_count", 0) + 1
                }

        except Exception:
            # tool返回不是合法json
            return {
                "verify": "answer"
            }

    
    

    model = load_chat_model(runtime.context.model)

    prompt = VERIFY_PROMPT.format(
        question=state["question"],
        sql_result=state.get("sql_result", []),
        tool_result=tool_data.get("result", "")
        )

    response = await model.ainvoke(prompt)

    if not isinstance(response, AIMessage):
        raise ValueError("router output must be AIMessage")
    text = response.content
    json_text = re.search(r"\{.*\}", text, re.S).group()
    data = json.loads(json_text)
    return {
        "verify": data["router"]
    }

async def answer_node(state: State, runtime: Runtime[Context]):

    model = load_chat_model(runtime.context.model)


    question = state["question"]
    sql_result = state.get("sql_result", [])
    sql_headers = state.get("sql_headers", [])

    context = f"""
用户问题：
{question}

SQL表头：
{sql_headers}

SQL结果：
{sql_result}

回答：
"""
    
    # Tool结果
    tool_result = ""
    last_msg = state["messages"][-1]

    if isinstance(last_msg, ToolMessage):
        tool_result = last_msg.content

    context += f"\nTool执行结果：\n{tool_result}\n"
    # if hasattr(state, "query_plan"):
    #     context += f"QueryPlan：{state.get('query_plan', [])}\n"


    response = await model.ainvoke([
        {"role": "system", "content": runtime.context.system_prompt},
        {"role": "user", "content": context}
        
    ])

    return {"messages": [response]}


## Build the graph
builder = StateGraph(State,context_schema=Context)

def router_output(state: State):
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        raise ValueError("router output must be AIMessage")
    text = last_msg.content
    json_text = re.search(r"\{.*\}", text, re.S).group()
    data = json.loads(json_text)
    r=data["router"]

    if r == "Semantic":
        return "cot_node"
    else:  
        return "numerical_node"
    
def subagent_router_output(state: State):
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        raise ValueError("router output must be AIMessage")
    text = last_msg.content
    json_text = re.search(r"\{.*\}", text, re.S).group()
    data = json.loads(json_text)
    r=data["router"]

    if r == "sql":
        return "sql_generate_node"
    elif r == "tool":
        return "tool_node"
    else:
        return "answer_node"


def verify_output(state: State):
    r=state.get("verify", "")

    if r == "retry_sql":
        return "sql_generate_node"
    elif r == "tool":
        return "tool_node"  
    else:
        return "answer_node"

# origin
builder.add_node("rag_node", rag_node)
builder.add_node("router_node", router_node)
builder.add_conditional_edges(
    "router_node",
    router_output
)
builder.add_node("cot_node", cot_node)
builder.add_node("numerical_node", numerical_node)

# builder.add_node("verify_node",  verify_node)
# builder.add_node("tool_node", ToolNode(TOOLS))
# builder.add_node("answer_node", answer_node)

builder.add_edge(START, "rag_node")
builder.add_edge("rag_node", "router_node")
builder.add_edge("cot_node", END)
builder.add_edge("numerical_node", END)




# builder.add_edge("sql_generate_node", "sql_exec_node")
# builder.add_edge("sql_exec_node", "verify_node")
# builder.add_edge("tool_node", "verify_node")
# builder.add_conditional_edges(
#     "verify_node",
#     verify_output
# )
# builder.add_edge("answer_node", END)



#table numerical_node test
# builder.add_node("numerical_node", numerical_node)
# builder.add_node("answer_node", answer_node)

# builder.add_edge(START, "numerical_node")
# builder.add_edge("numerical_node", "answer_node")
# builder.add_edge("answer_node", END)


# 进度： 已调通： rag_node  router_node cot_node



graph = builder.compile(name="Economic Agent")
