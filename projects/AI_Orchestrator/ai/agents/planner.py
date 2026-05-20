import logging
import re

from ai.prompts.planner_prompt import PLANNER_PROMPT
from ai.state.workflow_state import WorkflowState

from .base import BaseAgent

logger = logging.getLogger(__name__)

_PLAN_STEPS = ['retrieve_context', 'analyze_data', 'synthesize_response']

_SEARCH_KEYWORDS = ('search', 'find', 'lookup', 'news', 'latest', 'current', 'who is', 'what is')
_DB_KEYWORDS = ('workflow', 'execution', 'history', 'database', 'records', 'previous runs')
_CALC_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*/x])\s*(-?\d+(?:\.\d+)?)')
_OP_MAP = {'+': 'add', '-': 'subtract', '*': 'multiply', 'x': 'multiply', '/': 'divide'}


def _detect_tool(user_input: str) -> tuple[str, dict]:
    """Return (tool_name, tool_input). Empty string means no tool needed."""
    text = user_input.lower()

    match = _CALC_PATTERN.search(user_input)
    if match:
        a, op, b = match.groups()
        return 'calculator', {'op': _OP_MAP[op], 'a': float(a), 'b': float(b)}

    if any(k in text for k in _DB_KEYWORDS):
        return 'database', {'resource': 'recent_executions', 'limit': 5}

    if any(k in text for k in _SEARCH_KEYWORDS):
        return 'web_search', {'query': user_input}

    return '', {}


class PlannerAgent(BaseAgent):
    name = 'planner'

    def run(self, state: WorkflowState) -> dict:
        user_input = state['user_input']
        tool_name, tool_input = _detect_tool(user_input)

        logger.info('[PlannerAgent] Plan for: %.60s | tool=%s', user_input, tool_name or 'none')

        tools_required = [tool_name] if tool_name else []
        messages = [
            f'[Planner] Analyzing: "{user_input}"',
            f'[Planner] Plan: {" → ".join(_PLAN_STEPS)}',
        ]
        if tool_name:
            messages.append(f'[Planner] Tool required: {tool_name} (input: {tool_input})')
        else:
            messages.append('[Planner] No tool required — proceeding to executor.')

        return {
            'current_agent': 'planner',
            'current_step': 'planning',
            'execution_plan': {
                'steps': _PLAN_STEPS,
                'query': user_input,
                'tool': tool_name,
                'tool_input': tool_input,
            },
            'tools_required': tools_required,
            'tool_input': tool_input,
            'messages': messages,
            'execution_steps': [{
                'agent': 'planner',
                'action': 'plan_created',
                'tool_required': tool_name or None,
            }],
        }
