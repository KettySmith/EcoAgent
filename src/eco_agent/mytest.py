from dotenv import load_dotenv

from eco_agent.rag.retriever import Retriever

load_dotenv()

retriever = Retriever()

results = retriever.search_by_string("GDP 北京 2023", limit=5)

print(results)