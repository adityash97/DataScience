import operator
from typing import Annotated, TypedDict

MAX_RETRIES = 2


class WorkflowState(TypedDict):
    # Identity
    workflow_id: int
    execution_id: int

    # Input
    user_input: str

    # Execution tracking
    current_agent: str
    current_step: str
    next_step: str
    retry_count: int

    # Agent outputs
    execution_plan: dict
    execution_result: dict
    output: dict

    # Message log — appended by each node via LangGraph reducer
    messages: Annotated[list[str], operator.add]

    # Status and error
    status: str
    error: str
