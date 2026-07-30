"""简单模型调用测试 —— 用于排查模型连接/名称问题."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = "qwen/qwe3-8b"  # 修改这里测试不同模型

_provider, model = MODEL_NAME.split("/", maxsplit=1)

client = ChatOpenAI(
    model=model,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
    temperature=0,
)

print(f"Provider : {_provider}")
print(f"Model    : {model}")
print(f"Base URL : {os.environ.get('OPENAI_BASE_URL')}")
print(f"API Key  : {os.environ.get('OPENAI_API_KEY')[:20]}...")
print("-" * 50)

try:
    response = client.invoke("你好，请回复'OK'即可，不要多说。")
    print(f"✅ 成功!")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ 失败: {e}")
