from abc import ABC, abstractmethod

from ai.state.workflow_state import WorkflowState


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, state: WorkflowState) -> dict:
        """Receive current workflow state, return partial state update."""
