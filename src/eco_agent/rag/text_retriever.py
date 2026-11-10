from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from eco_agent.rag.bge_embedder import BGEEmbedder


class TextRetriever:
    """
    本地文本知识库检索器
    """

    def __init__(
        self,
        persist_directory: str = "rag_db/book_rag_db",
        embedding_model: str = "rag_model/bge-small-zh-v1.5",
        top_k: int = 3
    ):

        self.top_k = top_k
        # 相对路径以当前文件所在目录为基准；传入绝对路径时保留原路径。
        rag_directory = Path(__file__).resolve().parent
        persist_directory = str(rag_directory / persist_directory)
        embedding_model = str(rag_directory / embedding_model)

        embedding = BGEEmbedder(
            embedding_model
        )

        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )

    def retrieve(
        self,
        query_list: List[str]
    ) -> List[Document]:

        all_docs = []
        seen = set()

        for query in query_list:

            docs = self.db.similarity_search(
                query,
                k=self.top_k
            )

            for doc in docs:

                content = doc.page_content

                if content not in seen:
                    seen.add(content)
                    all_docs.append(doc)

        return all_docs
