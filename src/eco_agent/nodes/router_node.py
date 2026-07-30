
from langgraph.runtime import Runtime

from eco_agent.context import Context
from eco_agent.state import  State
from eco_agent.utils import load_chat_model
from eco_agent.prompts import  ROUTER_PROMPT

async def router_node(state: State, runtime: Runtime[Context]): 
    model = load_chat_model(runtime.context.model) 
    document = state.get("retrieved_docs", "") + "\n" + state.get("retrieved_tavily_docs", "")
    prompt = ROUTER_PROMPT.format( 
        question=state["question"], 
        document=document,
        table=state.get("retrieved_tables", "")
        ) 
    # print("router节点，state的table_name:", state["table_name"])
    response = await model.ainvoke(prompt) 
    return {
        "messages": [response]
            }