import os
from typing import Any, Dict, List
from langchain_core.documents import Document
from eco_agent.rag.query_analyzer import QueryAnalyzer
from eco_agent.rag.retriever import Retriever
from eco_agent.rag.text_retriever import TextRetriever
from eco_agent.rag.tavily_retriever import TavilyRetriever
# from app.rag.query_optimizer import AdvancedQueryOptimizer
# from app.rag.retriever import AdvancedHybridRetriever
# from app.rag.reranker import LLMReranker, LongContextReorder
# from app.rag.text_splitter import AdvancedParentDocumentSplitter
# from app.utils.logger import app_logger



class AdvancedRAGPipeline:
    """
    高级 RAG 管道

    完整流程：
    查询优化 → 混合检索 → 重排序 → 上下文优化
    """

    def __init__(
            self,
            model_name: str,
            # vectorstore: Chroma,
            # all_documents: List[Document],
            # parent_splitter: AdvancedParentDocumentSplitter,
            # query_strategy: str = "multi_query",
            # use_llm_reranker: bool = False,
            top_k: int = 3
            #enable_cache: bool = True
    ):
        """
        Args:
            vectorstore: 向量数据库（包含子文档）
            all_documents: 所有子文档列表
            parent_splitter: 父文档切分器（用于映射）
            query_strategy: 查询优化策略
            use_llm_reranker: 是否使用 LLM 重排序
            top_k: 最终返回的文档数量
        """

        self.top_k = top_k

        # 1. 查询优化器
        self.query_analyzer = QueryAnalyzer(model_name=model_name)

        # 2. 多路检索

        #本地文本库检索
        self.text_retriever = TextRetriever(top_k=3)

        #表格检索
        self.table_retriever =Retriever()

        #Tavily搜索引擎 文本
        self.tavily_retriever = TavilyRetriever(top_k=3)

        # # 3. 重排序器
        # self.reranker = LLMReranker(top_k=top_k * 2)

        # # 4. 父文档映射器
        # self.parent_splitter = parent_splitter

        # # 5. 长上下文重排序
        # self.context_reorder = LongContextReorder()

        # 6. 缓存层
        #self.cache = RAGCache(enabled=enable_cache)

    async def retrieve(self, query: str) -> Dict[str, Any]:
        """
        完整检索流程

        Args:
            query: 用户查询

        Returns:
            优化后的上下文文档列表
        """

        # 尝试从缓存获取
        # cached_result = self.cache.get(query, self.top_k)
        # if cached_result:
        #     return cached_result

        # app_logger.info(f"开始 Advanced RAG 检索: {query}")

        # ========== 阶段 1：查询优化 ==========
        analyzer_result = await self.query_analyzer.query_analyze(query)
        keyquery_list = analyzer_result["keyquery_list"]
        print(f"1️. 查询优化完成，生成 {keyquery_list} ")
        # app_logger.info(f"1️. 查询优化完成，生成 {len(optimized_queries)} 个查询")

        # ========== 阶段 2：混合检索 ==========

        #本地文本检索  用的原始query
        text_docs = self.text_retriever.retrieve(query)
        #表格检索
        table_docs = self.table_retriever.retrieve(keyquery_list)
    #     table_docs=[
    #     {
    #     "id":"1213635_12",
    #     "meta":{
    #         "table_id":"1213635",
    #         "table_name":"8-6 主要城市平均相对湿度 (2020年) 单位：% ",
    #         "original_text":"合肥"
    #         },
    #     "index":{
    #         "location":"合肥",
    #         "time":"2020年",
    #         "entity_supp":"",
    #         "attribute_names":[
    #         "主要城市平均相对湿度（1月，%）",
    #         "主要城市平均相对湿度（2月，%）",
    #         "主要城市平均相对湿度（3月，%）",
    #         "主要城市平均相对湿度（4月，%）",
    #         "主要城市平均相对湿度（5月，%）",
    #         "主要城市平均相对湿度（6月，%）",
    #         "主要城市平均相对湿度（7月，%）",
    #         "主要城市平均相对湿度（8月，%）",
    #         "主要城市平均相对湿度（9月，%）",
    #         "主要城市平均相对湿度（10月，%）",
    #         "主要城市平均相对湿度（11月，%）",
    #         "主要城市平均相对湿度（12月，%）",
    #         "主要城市年度平均相对湿度（%）"
    #         ]
    #         },
    #     "data":{
    #         "主要城市平均相对湿度（1月，%）":"90",
    #         "主要城市平均相对湿度（2月，%）":"80",
    #         "主要城市平均相对湿度（3月，%）":"76",
    #         "主要城市平均相对湿度（4月，%）":"71",
    #         "主要城市平均相对湿度（5月，%）":"75",
    #         "主要城市平均相对湿度（6月，%）":"87",
    #         "主要城市平均相对湿度（7月，%）":"93",
    #         "主要城市平均相对湿度（8月，%）":"85",
    #         "主要城市平均相对湿度（9月，%）":"84",
    #         "主要城市平均相对湿度（10月，%）":"85",
    #         "主要城市平均相对湿度（11月，%）":"82",
    #         "主要城市平均相对湿度（12月，%）":"74",
    #         "主要城市年度平均相对湿度（%）":"82"
    #         },
    #     "full_text_search":"合肥 2020年 主要城市平均相对湿度（1月，%） 主要城市平均相对湿度（2月，%） 主要城市平均相对湿度（3月，%） 主要城市平均相对湿度（4月，%） 主要城市平均相对湿度（5月，%） 主要城市平均相对湿度（6月，%） 主要城市平均相对湿度（7月，%） 主要城市平均相对湿度（8月，%） 主要城市平均相对湿度（9月，%） 主要城市平均相对湿度（10月，%） 主要城市平均相对湿度（11月，%） 主要城市平均相对湿度（12月，%） 主要城市年度平均相对湿度（%） 8-6 主要城市平均相对湿度 (2020年) 单位：%"        
    #     },
    #     {
    #     "id":"1213635_18",
    #     "meta":{
    #         "table_id":"1213635",
    #         "table_name":"8-6 主要城市平均相对湿度 (2020年) 单位：% ",
    #         "original_text":"武汉"
    #         },
    #     "index":{
    #         "location":"武汉",
    #         "time":"2020年",
    #         "entity_supp":"",
    #         "attribute_names":[
    #         "主要城市平均相对湿度（1月，%）",
    #         "主要城市平均相对湿度（2月，%）",
    #         "主要城市平均相对湿度（3月，%）",
    #         "主要城市平均相对湿度（4月，%）",
    #         "主要城市平均相对湿度（5月，%）",
    #         "主要城市平均相对湿度（6月，%）",
    #         "主要城市平均相对湿度（7月，%）",
    #         "主要城市平均相对湿度（8月，%）",
    #         "主要城市平均相对湿度（9月，%）",
    #         "主要城市平均相对湿度（10月，%）",
    #         "主要城市平均相对湿度（11月，%）",
    #         "主要城市平均相对湿度（12月，%）",
    #         "主要城市年度平均相对湿度（%）"
    #         ]
    #         },
    #     "data":{
    #         "主要城市平均相对湿度（1月，%）":"86",
    #         "主要城市平均相对湿度（2月，%）":"81",
    #         "主要城市平均相对湿度（3月，%）":"79",
    #         "主要城市平均相对湿度（4月，%）":"71",
    #         "主要城市平均相对湿度（5月，%）":"77",
    #         "主要城市平均相对湿度（6月，%）":"86",
    #         "主要城市平均相对湿度（7月，%）":"88",
    #         "主要城市平均相对湿度（8月，%）":"76",
    #         "主要城市平均相对湿度（9月，%）":"85",
    #         "主要城市平均相对湿度（10月，%）":"85",
    #         "主要城市平均相对湿度（11月，%）":"82",
    #         "主要城市平均相对湿度（12月，%）":"76",
    #         "主要城市年度平均相对湿度（%）":"81"
    #         },
    #     "full_text_search":"武汉 2020年 主要城市平均相对湿度（1月，%） 主要城市平均相对湿度（2月，%） 主要城市平均相对湿度（3月，%） 主要城市平均相对湿度（4月，%） 主要城市平均相对湿度（5月，%） 主要城市平均相对湿度（6月，%） 主要城市平均相对湿度（7月，%） 主要城市平均相对湿度（8月，%） 主要城市平均相对湿度（9月，%） 主要城市平均相对湿度（10月，%） 主要城市平均相对湿度（11月，%） 主要城市平均相对湿度（12月，%） 主要城市年度平均相对湿度（%） 8-6 主要城市平均相对湿度 (2020年) 单位：%"    
    #     },
    #     {
    #     "id":"1213641_11",
    #     "meta":{
    #     "table_id":"1213641",
    #     "table_name":"8-14 主要城市废气中主要污染物排放情况 (2020年) 单位：吨 ",
    #     "original_text":" 合肥"
    #     },
    #     "index":{
    #         "location":"合肥",
    #         "time":"2020年",
    #         "entity_supp":"",
    #         "attribute_names":[
    #         "工业源二氧化硫排放量（单位：吨）",
    #         "工业源氮氧化物排放量（单位：吨）",
    #         "工业源颗粒物排放量（单位：吨）",
    #         "生活及其他源二氧化硫排放量（单位：吨）",
    #         "生活及其他源氮氧化物排放量（单位：吨）",
    #         "生活及其他源颗粒物排放量（单位：吨）"
    #         ]
    #     },
    #     "data":{
    #         "工业源二氧化硫排放量（单位：吨）":"4742.063",
    #         "工业源氮氧化物排放量（单位：吨）":"11627.567",
    #         "工业源颗粒物排放量（单位：吨）":"4518.88",
    #         "生活及其他源二氧化硫排放量（单位：吨）":"599.16",
    #         "生活及其他源氮氧化物排放量（单位：吨）":"1678.57",
    #         "生活及其他源颗粒物排放量（单位：吨）":"6080.5"
    #     },
    #     "full_text_search":"合肥 2020年 工业源二氧化硫排放量（单位：吨） 工业源氮氧化物排放量（单位：吨） 工业源颗粒物排放量（单位：吨） 生活及其他源二氧化硫排放量（单位：吨） 生活及其他源氮氧化物排放量（单位：吨） 生活及其他源颗粒物排放量（单位：吨） 8-14 主要城市废气中主要污染物排放情况 (2020年) 单位：吨"    
    # },
    # {
    #     "id":"1213981_6",
    #     "meta":{
    #         "table_id":"1213981",
    #         "table_name":"28-11 能源平衡表 单位：亿升标准油 注：①再生能源包括太阳光电、风力发电和太阳热能。 ②非能源消费仅含润滑油、柏油、溶剂油。 ",
    #         "original_text":" 再生能源①"
    #     },
    #     "index":{
    #         "location":"中国",
    #         "time":"2016-2020年分年度数据（能源平衡表，单位：亿升标准油）",
    #         "entity_supp":"再生能源①",
    #         "attribute_names":[
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2016年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2017年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2018年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2019年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2020年）"
    #         ]
    #     },
    #     "data":{
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2016年）":"3.57324",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2017年）":"4.3704",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2018年）":"5.26942",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2019年）":"6.65464",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2020年）":"8.95659"
    #     },
    #     "full_text_search":"中国 2016-2020年分年度数据（能源平衡表，单位：亿升标准油） 再生能源① 中国能源平衡表年度能源总供给量（亿升标准油，2016年） 中国能源平衡表年度能源总供给量（亿升标准油，2017年） 中国能源平衡表年度能源总供给量（亿升标准油，2018年） 中国能源平衡表年度能源总供给量（亿升标准油，2019年） 中国能源平衡表年度能源总供给量（亿升标准油，2020年） 28-11 能源平衡表 单位：亿升标准油 注：①再生能源包括太阳光电、风力发电和太阳热能。 ②非能源消费仅含润滑油、柏油、溶剂油。"
    # },
    # {
    #     "id":"1213981_13",
    #     "meta":{
    #         "table_id":"1213981",
    #         "table_name":"28-11 能源平衡表 单位：亿升标准油 注：①再生能源包括太阳光电、风力发电和太阳热能。 ②非能源消费仅含润滑油、柏油、溶剂油。 ",
    #         "original_text":" 本地能源消费按部门分"
    #     },
    #     "index":{
    #         "location":"中国",
    #         "time":"2016-2020年分年度数据（能源平衡表，单位：亿升标准油）",
    #         "entity_supp":"本地能源消费按部门分",
    #         "attribute_names":[
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2016年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2017年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2018年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2019年）",
    #         "中国能源平衡表年度能源总供给量（亿升标准油，2020年）"
    #         ]
    #     },
    #     "data":{
    #     "中国能源平衡表年度能源总供给量（亿升标准油，2016年）":"865.89314",
    #     "中国能源平衡表年度能源总供给量（亿升标准油，2017年）":"860.38292",
    #     "中国能源平衡表年度能源总供给量（亿升标准油，2018年）":"875.69",
    #     "中国能源平衡表年度能源总供给量（亿升标准油，2019年）":"849.07982",
    #     "中国能源平衡表年度能源总供给量（亿升标准油，2020年）":"853.63765"
    #     },
    #     "full_text_search":"中国 2016-2020年分年度数据（能源平衡表，单位：亿升标准油） 本地能源消费按部门分 中国能源平衡表年度能源总供给量（亿升标准油，2016年） 中国能源平衡表年度能源总供给量（亿升标准油，2017年） 中国能源平衡表年度能源总供给量（亿升标准油，2018年） 中国能源平衡表年度能源总供给量（亿升标准油，2019年） 中国能源平衡表年度能源总供给量（亿升标准油，2020年） 28-11 能源平衡表 单位：亿升标准油 注：①再生能源包括太阳光电、风力发电和太阳热能。 ②非能源消费仅含润滑油、柏油、溶剂油。"    
    # },

    # ]
    #     #tavily联网搜索
        # tavily_docs = await self.tavily_retriever.retrieve(query)
        tavily_docs=[]


        # app_logger.info(f"2️. 混合检索完成，获得 {len(child_docs)} 个候选文档")

        # # ========== 阶段 3：重排序 ==========
        # reranked_child_docs = self.reranker.rerank(
        #     query=query,
        #     documents=child_docs,
        #     top_k=self.top_k * 2
        # )
        # app_logger.info(f"3️. 重排序完成，保留 {len(reranked_child_docs)} 个文档")

        # # ========== 阶段 4：上下文优化 ==========
        # # 4.1 映射到父文档
        # parent_docs = self.parent_splitter.get_parent_context(reranked_child_docs)
        # app_logger.info(f"4️. 父文档映射完成，获得 {len(parent_docs)} 个完整上下文")

        # # 4.2 长上下文重排序
        # final_docs = self.context_reorder.reorder(parent_docs[:self.top_k])
        # app_logger.info(f"✅ RAG 检索完成，最终返回 {len(final_docs)} 个文档")

        # 缓存结果
        #self.cache.set(query, self.top_k, final_docs)

        return {
            "tables": table_docs,
            "docs": text_docs,
            "tavily_docs":tavily_docs
        }