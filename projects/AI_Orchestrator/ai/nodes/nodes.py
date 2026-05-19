import logging

from ai.registry.registry import get_agent
from ai.state.workflow_state import MAX_RETRIES, WorkflowState
from ai.tools import run_tool

logger = logging.getLogger(__name__)


def input_node(state: WorkflowState) -> dict:
    logger.info('[input_node] Received input: %.60s', state['user_input'])
    return {
        'current_step': 'input',
        'current_agent': '',
        'status': 'running',
        'retry_count': 0,
        'tools_required': [],
        'tool_used': '',
        'tool_input': {},
        'tool_result': {},
        'messages': [f"Input received: {state['user_input']}"],
        'execution_steps': [{'agent': 'input', 'action': 'received_input'}],
    }


def planner_node(state: WorkflowState) -> dict:
    return get_agent('planner').run(state)


def tool_node(state: WorkflowState) -> dict:
    """
    Execute the tool requested by the planner.
    Reads tools_required + tool_input from state, writes tool_used + tool_result.
    """
    tools = state.get('tools_required') or []
    if not tools:
        logger.info('[tool_node] No tool to run — skipping.')
        return {
            'current_step': 'tool_skipped',
            'messages': ['[ToolNode] No tool required.'],
        }

    tool_name = tools[0]
    tool_input = state.get('tool_input') or {}
    logger.info('[tool_node] Calling tool=%s input=%s', tool_name, tool_input)

    try:
        result = run_tool(tool_name, tool_input)
        success = result.get('success', False)
        logger.info('[tool_node] Tool %s completed — success=%s', tool_name, success)
        return {
            'current_agent': 'tool_runner',
            'current_step': 'tool_executed',
            'tool_used': tool_name,
            'tool_result': result,
            'messages': [
                f'[ToolNode] Calling tool: {tool_name}',
                f'[ToolNode] Tool result: {"ok" if success else "failed"}',
            ],
            'execution_steps': [{
                'agent': 'tool_runner',
                'action': 'tool_executed',
                'tool': tool_name,
                'success': success,
            }],
        }
    except Exception as exc:
        logger.exception('[tool_node] Tool execution crashed')
        return {
            'current_step': 'tool_failed',
            'tool_used': tool_name,
            'tool_result': {'tool': tool_name, 'success': False, 'error': str(exc)},
            'messages': [f'[ToolNode] Tool {tool_name} crashed: {exc}'],
            'execution_steps': [{
                'agent': 'tool_runner',
                'action': 'tool_failed',
                'tool': tool_name,
                'error': str(exc),
            }],
        }


def executor_node(state: WorkflowState) -> dict:
    return get_agent('executor').run(state)


def response_node(state: WorkflowState) -> dict:
    return get_agent('response').run(state)


def retry_handler(state: WorkflowState) -> dict:
    retry_count = state.get('retry_count', 0) + 1
    logger.warning('[retry_handler] Retrying execution — attempt %d/%d', retry_count, MAX_RETRIES)
    return {
        'current_step': 'retrying',
        'status': 'retrying',
        'retry_count': retry_count,
        'messages': [f'[RetryHandler] Retrying (attempt {retry_count}/{MAX_RETRIES})'],
    }


def error_handler(state: WorkflowState) -> dict:
    retry_count = state.get('retry_count', 0)
    logger.error('[error_handler] Workflow failed after %d retries', retry_count)
    return {
        'current_step': 'failed',
        'status': 'failed',
        'output': {
            'result': None,
            'error': state.get('error') or 'Workflow execution failed after max retries.',
            'retry_count': retry_count,
            'agents_executed': ['Planner', 'Executor'],
        },
        'messages': [f'[ErrorHandler] Workflow failed — max retries ({MAX_RETRIES}) exceeded.'],
    }


def output_node(state: WorkflowState) -> dict:
    logger.info(
        '[output_node] Finalizing — agent: %s tool: %s retries: %s',
        state.get('current_agent'), state.get('tool_used') or 'none', state.get('retry_count', 0),
    )
    return {
        'current_step': 'output',
        'status': 'completed',
        'messages': ['Workflow completed.'],
        'execution_steps': [{'agent': 'output', 'action': 'workflow_completed'}],
    }
