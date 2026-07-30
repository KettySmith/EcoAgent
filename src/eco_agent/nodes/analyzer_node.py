import json
import re
from eco_agent.prompts import ANALYZER_PROMPT
from eco_agent.utils import load_chat_model
from langgraph.runtime import Runtime
from eco_agent.context import Context

##弃用，用于retriever内部
async def analyzer_node(state, runtime: Runtime[Context]):

    model = load_chat_model(runtime.context.model)
    question = state["question"]
    prompt = ANALYZER_PROMPT.format(
        question=question
    )

    response = await model.ainvoke(prompt)
    content = response.content
    output = parse_output(content)

    metrics = output["metrics"]
    times = output["time"]
    locations = output["location"]

    # 如果为空列表，给一个 None 占位，保证生成的组合
    if not metrics:
        metrics = [None]
    else:
        metrics=list(dict.fromkeys(metrics))
    if not times:
        times = [None]
    else:
        times=list(dict.fromkeys(times))
    if not locations:
        locations = [None]
    else:
        locations=list(dict.fromkeys(locations))

    # 2. 生成扁平化组合
    output_list = []
    for me in metrics:
        for time in times:
            for location in locations:
                output_list.append({
                    "metrics": me,
                    "time": time,
                    "location": location
                })


    return {
        "keyquery_list": output_list
    }


def parse_output(text: str) -> dict:
    """
    防御性解析，防止模型输出多余文本
    """
    try:
        json_text = re.search(r"\{.*\}", text, re.S).group()
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Analyzer 输出解析失败: {text}")


   


