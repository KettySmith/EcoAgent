from typing import List
import os

from tavily import TavilyClient


class TavilyRetriever:
    """
    基于 Tavily 的 Web 检索器
    """

    def __init__(
        self,
        top_k: int = 3
    ):
        self.top_k = top_k

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not found"
            )

        self.client = TavilyClient(
            api_key=api_key
        )

    async def retrieve(
        self,
        query: str
    ) -> List[dict]:

        try:

            response = self.client.search(
                query=query,
                max_results=self.top_k
            )

            docs = []

            for item in response.get(
                "results",
                []
            ):

                docs.append(
                    {
                        "title": item.get(
                            "title",
                            ""
                        ),
                        "content": item.get(
                            "content",
                            ""
                        ),
                        "source": item.get(
                            "url",
                            ""
                        )
                    }
                )

            return docs

        except Exception as e:

            print(
                f"Tavily search failed: {e}"
            )

            return []