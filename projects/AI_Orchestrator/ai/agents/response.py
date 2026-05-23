import logging

from ai.prompts.response_prompt import RESPONSE_PROMPT
from ai.state.workflow_state import WorkflowState

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ResponseAgent(BaseAgent):
    name = 'response'

    def run(self, state: WorkflowState) -> dict:
        # import pdb;pdb.set_trace()
        tool_used = state.get('tool_used', '')
        logger.info('[ResponseAgent] Fnormatting final respose | tool=%s', tool_used or 'none')
        if state["tool_result"]['tool'] == 'web_search':

            tool_result = state.get('tool_result', {})
            results = tool_result.get('results', [])

            formatted_results = '\n'.join(
                [
                    '\n'+f'- {item.get("title", "")}. Source: {item.get("url", "")}'
                    for item in results
                ]
            )

            final_result = (
                f'Orchestrated response for query "{state["user_input"]}":\n\n'
                f'{formatted_results}'
            )
        else:
            final_result = f'Orchestrated response for query "{state["user_input"]}": {state["tool_result"]}'
        return {
            'current_agent': 'response',
            'current_step': 'responding',
            'output': {
                'result': final_result,
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
