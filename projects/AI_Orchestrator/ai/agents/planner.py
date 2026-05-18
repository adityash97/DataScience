import logging

from ai.prompts.planner_prompt import PLANNER_PROMPT
from ai.state.workflow_state import WorkflowState

from .base import BaseAgent

logger = logging.getLogger(__name__)

_PLAN_STEPS = ['retrieve_context', 'analyze_data', 'synthesize_response']


class PlannerAgent(BaseAgent):
    name = 'planner'

    def run(self, state: WorkflowState) -> dict:
        logger.info('[PlannerAgent] Building execution plan for: %.60s', state['user_input'])
        return {
            'current_agent': 'planner',
            'current_step': 'planning',
            'execution_plan': {
                'steps': _PLAN_STEPS,
                'query': state['user_input'],
            },
            'messages': [
                f'[Planner] Analyzing: "{state["user_input"]}"',
                f'[Planner] Plan: {" → ".join(_PLAN_STEPS)}',
            ],
        }
