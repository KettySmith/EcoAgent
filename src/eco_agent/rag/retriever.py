from typing import List, Dict
import meilisearch
import os


class Retriever:
    def __init__(
        self,
        meilisearch_url: str = None,
        api_key: str = None,
        index_name: str = None
    ):
        # 延迟读取环境变量（避免模块导入时 load_dotenv 尚未执行）
        meilisearch_url = meilisearch_url or os.environ.get("MEILISEARCH_URL")
        api_key = api_key or os.environ.get("MEILISEARCH_API_KEY")
        index_name = index_name or os.environ.get("MEILISEARCH_INDEX")

        self.ms_client = meilisearch.Client(meilisearch_url, api_key)
        self.ms_index = self.ms_client.index(index_name)

    # 单条检索
    def search_by_string(self,search_query: str,limit: int = 100) -> List[Dict]: #输入查询字符串，查询条数
        # 清洗结果：去除多余空格、换行，确保仅空格分隔
        search_terms = search_query.strip().replace("\n", " ").replace("\t", " ")
        # 去重关键词
        search_terms = " ".join(list(dict.fromkeys(search_terms.split())))
        search_result = self.ms_index.search(
            query=search_terms,
            opt_params={'limit': limit}
        )
        return search_result.get("hits",[])
    
    def retrieve(self, keyquery_list: List[Dict]) -> List[Dict]:
        retrieved_results=[]
        # 不同关键词检索结果最终拼接入同一个总列表中
        for query in keyquery_list:
            parts = [query.get("metrics") or "", query.get("time") or "", query.get("location") or ""]
            search_string = ",".join(p for p in parts if p)
            retri_result=self.search_by_string(search_string,2)
            retrieved_results.extend(retri_result)
        return retrieved_results


 


