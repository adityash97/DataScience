import logging

from ai.prompts.response_prompt import RESPONSE_PROMPT
from ai.state.workflow_state import WorkflowState

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ResponseAgent(BaseAgent):
    name = 'response'

    def run(self, state: WorkflowState) -> dict:
        logger.info('[ResponseAgent] Formatting final response')
        return {
            'current_agent': 'response',
            'current_step': 'responding',
            'output': {
                'result': f'Orchestrated response for: "{state["user_input"]}"',
                'agents_executed': ['Planner', 'Executor', 'Response'],
                'execution_plan': state.get('execution_plan', {}),
                'execution_result': state.get('execution_result', {}),
                'execution_log': state['messages'],
                'retry_count': state.get('retry_count', 0),
                'note': 'Mock multi-agent execution — LLM integration pending.',
            },
            'messages': ['[Response] Final answer formatted successfully.'],
        }
