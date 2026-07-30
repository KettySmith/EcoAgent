import asyncio
from langchain_core.messages import HumanMessage
from eco_agent.graph import graph
from eco_agent.context import Context
from eco_agent.utils import load_chat_model, QueryExecutor
from dotenv import load_dotenv

load_dotenv()


def pretty_print(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


async def main():

    print("🚀 Economic Agent 启动成功")

    db = QueryExecutor()
    # table_name = "china_statistical_yearbook_2021c03_12"
    # # table_name="china_statistical_yearbook_2021c06_01_1020131340894"

    # headers, rows = db.get_table_data(table_name)
    # md_table = db.table_to_markdown(headers, rows)
    # db.close()
    # user_input = "影响中国GDP的因素有哪些"
    user_input = "2020年12月合肥的相对湿度是否高于武汉？"
    # user_input="在2020年，全国居民人均消费支出中，哪三项支出的金额最高？"

    print("\n🔥 STREAM TRACE START\n")

    initial_state =  {
            "question": user_input,
            "table": "",
            "table_name": "",
            "sql_result": [],
            "sql_headers": [],
            "sql_retry_count":0,
            "sql_success":True, # 避免无需sql时，默认执行失败
            "tool_retry_count":0, # 避免无需tool时，默认执行失败
            "messages": [HumanMessage(content=user_input)]
        }

    async for event in graph.astream(initial_state,context=Context(),stream_mode="updates"):
        print("\n" + "-" * 60)
        print(event)


if __name__ == "__main__":
    # react-agent/src$ python3 -m eco_agent.main
    asyncio.run(main())
