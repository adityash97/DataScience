import logging

from ai.prompts.response_prompt import RESPONSE_PROMPT
from ai.state.workflow_state import WorkflowState

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ResponseAgent(BaseAgent):
    name = 'response'

    def run(self, state: WorkflowState) -> dict:
        tool_used = state.get('tool_used', '')
        logger.info('[ResponseAgent] Formatting final response | tool=%s', tool_used or 'none')

        return {
            'current_agent': 'response',
            'current_step': 'responding',
            'output': {
                'result': f'Orchestrated response for: "{state["user_input"]}"',
                'agents_executed': ['Planner', 'Tool', 'Executor', 'Response'] if tool_used else ['Planner', 'Executor', 'Response'],
                'execution_plan': state.get('execution_plan', {}),
                'execution_result': state.get('execution_result', {}),
                'tool_used': tool_used or None,
                'tool_result': state.get('tool_result', {}) or None,
                'execution_log': state['messages'],
                'execution_steps': state.get('execution_steps', []),
                'retry_count': state.get('retry_count', 0),
                'note': 'Multi-agent execution with tool calling — LLM integration pending.',
            },
            'messages': ['[Response] Final answer formatted successfully.'],
            'execution_steps': [{'agent': 'response', 'action': 'response_formatted'}],
        }
