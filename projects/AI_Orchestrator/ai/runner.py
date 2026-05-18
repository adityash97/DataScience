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
        'current_step': '',
        'messages': [],
        'output': {},
        'error': '',
        'status': 'running',
    }

    try:
        graph = build_workflow_graph()
        final_state = graph.invoke(initial_state)
        logger.info('Workflow completed — execution_id=%s', execution_id)
        return {
            'success': True,
            'output': final_state['output'],
            'messages': final_state['messages'],
            'status': final_state['status'],
        }
    except Exception as exc:
        logger.exception('Workflow failed — execution_id=%s', execution_id)
        return {
            'success': False,
            'output': {},
            'messages': [],
            'status': 'failed',
            'error': str(exc),
        }
