from langgraph.runtime import Runtime

from eco_agent.context import Context
from eco_agent.state import  State
from eco_agent.utils import load_chat_model
from eco_agent.prompts import  COT_PROMPT

async def cot_node(state: State, runtime: Runtime[Context]):
    model = load_chat_model(runtime.context.model)

    prompt = COT_PROMPT.format(
        question=state["question"],
        document=state.get("retrieved_docs", ""),
        table=state.get("retrieved_tables", "")
    )

    response = await model.ainvoke(prompt)

    return {
        "messages": [response],
        "final_answer": response.content
    }