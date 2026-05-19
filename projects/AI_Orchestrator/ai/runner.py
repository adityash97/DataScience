import logging

from ai.graphs.workflow_graph import build_workflow_graph
from ai.state.workflow_state import WorkflowState

logger = logging.getLogger(__name__)


def run_workflow(workflow_id: int, execution_id: int, user_input: str) -> dict:
    logger.info('Workflow started — workflow_id=%s execution_id=%s', workflow_id, execution_id)

    initial_state: WorkflowState = {
        'workflow_id': workflow_id,
        'execution_id': execution_id,
        'user_input': user_input,
        'current_agent': '',
        'current_step': '',
        'next_step': '',
        'retry_count': 0,
        'execution_plan': {},
        'execution_result': {},
        'tools_required': [],
        'tool_used': '',
        'tool_input': {},
        'tool_result': {},
        'execution_steps': [],
        'messages': [],
        'output': {},
        'error': '',
        'status': 'running',
    }

    try:
        graph = build_workflow_graph()
        final_state = graph.invoke(initial_state)
        status = final_state.get('status', 'completed')
        logger.info(
            'Workflow finished — execution_id=%s status=%s retries=%s tool=%s',
            execution_id,
            status,
            final_state.get('retry_count', 0),
            final_state.get('tool_used') or 'none',
        )
        return {
            'success': status != 'failed',
            'status': status,
            'output': final_state.get('output', {}),
            'messages': final_state.get('messages', []),
            'retry_count': final_state.get('retry_count', 0),
            'current_step': final_state.get('current_step', ''),
            'tool_used': final_state.get('tool_used', ''),
            'tool_result': final_state.get('tool_result', {}),
            'execution_steps': final_state.get('execution_steps', []),
        }
    except Exception as exc:
        logger.exception('Workflow crashed — execution_id=%s', execution_id)
        return {
            'success': False,
            'status': 'failed',
            'output': {},
            'messages': [],
            'retry_count': 0,
            'current_step': 'error',
            'tool_used': '',
            'tool_result': {},
            'execution_steps': [],
            'error': str(exc),
        }
