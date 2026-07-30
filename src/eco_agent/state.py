"""Define the state structures for the agent."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated, TypedDict


class InputState(TypedDict):

    messages: Annotated[Sequence[AnyMessage], add_messages]


QueryType = Literal[
    "lookup",
    "compare",
    "aggregation",
    "topk",
    "trend",
    "other"
]


class QueryPlan:
    query_type: QueryType
    metric: str

    filters: Dict[str, Union[str, List[str], None]]
    # e.g. {"time": "2020", "location": "北京"}

    group_by: List[str] 

    order_by: Dict[str, Any] 
    # {"field": "price", "direction": "desc"}

    limit: Optional[int] 
    aggregation: Optional[Literal["avg", "sum", "max", "min"]] 
    
    compare_keys: List[str]
    # compare_keys = ["location"]



class State(InputState):

    question: str
    table: str
    table_name: str

    # Candidate tables returned by retrieval. Each item should retain the
    # physical database table name separately from its display title.
    candidate_tables: List[Dict[str, Any]]
    selected_table_name: str
    table_schema: List[Dict[str, str]]

    # 经济三要素
    # keyquery_list: List[Any]

    #rag
    retrieved_tables: str #List[Any]
    retrieved_docs: str #List[Any]



    # schema linking
    matched_columns: Dict[str, str]

    # ========== Query Decomposition ==========
    sub_questions: List[str]
    current_sub_index: int
    sub_evidence: List[Dict]  # [{sub_q, evidence, source}]

    # ========== Complexity Router ==========
    complexity: str  # "simple" | "complex"

    # ========== CoT ==========
    cot_result: str



    # SQL
    query_plan: QueryPlan | None
    sql: str | None = None
    sql_result: Any = None
    sql_headers: list[str]
    sql_success: bool  # 无需sql时默认为true
    sql_error: str
    sql_retry_count: int

    # Tool execution
    tool_retry_count: int
    tool_trace: List[Dict[str, Any]]

    # Verify
    verify: str
    verify_retry_count: int  # per-sub-question retry limit

    # Calculation
    requires_calculation: bool
    calculation_expression: str
    calculation_executed: bool
    formula_template: str
    formula: str
    calculation_result: str

    # Final
    answer: str
    final_answer: str

    is_last_step: IsLastStep
