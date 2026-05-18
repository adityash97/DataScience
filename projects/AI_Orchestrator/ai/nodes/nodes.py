import logging

from ai.registry.registry import get_agent
from ai.state.workflow_state import MAX_RETRIES, WorkflowState

logger = logging.getLogger(__name__)


def input_node(state: WorkflowState) -> dict:
    logger.info('[input_node] Received input: %.60s', state['user_input'])
    return {
        'current_step': 'input',
        'current_agent': '',
        'status': 'running',
        'retry_count': 0,
        'messages': [f"Input received: {state['user_input']}"],
    }


def planner_node(state: WorkflowState) -> dict:
    return get_agent('planner').run(state)


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
    logger.info('[output_node] Finalizing — agent: %s retries: %s', state.get('current_agent'), state.get('retry_count', 0))
    return {
        'current_step': 'output',
        'status': 'completed',
        'messages': ['Workflow completed.'],
    }
