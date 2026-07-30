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
        persist_directory: str = "/home/zhangruochi/LYC/book_rag_db",
        embedding_model: str = "/home/zhangruochi/LYC/bge-small-zh-v1.5",
        top_k: int = 3
    ):

        self.top_k = top_k

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