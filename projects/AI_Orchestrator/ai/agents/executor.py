import logging

from ai.prompts.executor_prompt import EXECUTOR_PROMPT
from ai.state.workflow_state import WorkflowState

from .base import BaseAgent

logger = logging.getLogger(__name__)


def _summarize_tool_result(tool_result: dict) -> str:
    if not tool_result:
        return 'No tool result available.'
    tool = tool_result.get('tool', 'unknown')
    if not tool_result.get('success', False):
        return f"Tool '{tool}' failed: {tool_result.get('error', 'unknown error')}"
    if tool == 'calculator':
        return f"Calculator: {tool_result.get('a')} {tool_result.get('op')} {tool_result.get('b')} = {tool_result.get('result')}"
    if tool == 'web_search':
        results = tool_result.get('results', [])
        titles = [r.get('title', '') for r in results[:3]]
        return f"Web search returned {len(results)} hits: {titles}"
    if tool == 'database':
        resource = tool_result.get('resource', '')
        count = tool_result.get('count', None)
        return f"Database '{resource}' fetched ({count if count is not None else 'ok'})"
    return f"Tool '{tool}' completed."


class ExecutorAgent(BaseAgent):
    name = 'executor'

    def run(self, state: WorkflowState) -> dict:
        plan = state.get('execution_plan', {})
        tool_used = state.get('tool_used', '')
        tool_result = state.get('tool_result', {})
        tool_summary = _summarize_tool_result(tool_result) if tool_used else 'No tool used.'

        logger.info('[ExecutorAgent] Executing plan: %s | tool=%s', plan, tool_used or 'none')

        messages = [
            '[Executor] Retrieving relevant context...',
            '[Executor] Analyzing retrieved data...',
        ]
        if tool_used:
            messages.append(f'[Executor] Incorporating tool result — {tool_summary}')
        messages.append('[Executor] Synthesizing intermediate result...')
        # import pdb;pdb.set_trace()
        return {
            'current_agent': 'executor',
            'current_step': 'executing',
            'execution_result': {
                'data': f'Processed query: "{state["user_input"]}"',
                'plan_executed': plan.get('steps', []),
                'tool_used': tool_used or None,
                'tool_summary': tool_summary,
                'tool_result': tool_result or None,
                'success': True,
            },
            'messages': messages,
            'execution_steps': [{
                'agent': 'executor',
                'action': 'executed_plan',
                'consumed_tool': tool_used or None,
            }],
        }
