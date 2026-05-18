import operator
from typing import Annotated, TypedDict


class WorkflowState(TypedDict):
    workflow_id: int
    execution_id: int
    user_input: str
    current_step: str
    messages: Annotated[list[str], operator.add]
    output: dict
    error: str
    status: str
