import logging

from ai.state.workflow_state import MAX_RETRIES, WorkflowState

logger = logging.getLogger(__name__)


def route_after_executor(state: WorkflowState) -> str:
    """
    Conditional edge after executor_node.

    Returns:
        'retry'    — execution failed, retries remaining
        'error'    — execution failed, no retries left
        'response' — execution succeeded, continue to response agent
    """
    if state.get('status') == 'failed':
        retry_count = state.get('retry_count', 0)
        if retry_count < MAX_RETRIES:
            logger.warning('[Router] Executor failed — routing to retry (attempt %d/%d)', retry_count + 1, MAX_RETRIES)
            return 'retry'
        logger.error('[Router] Executor failed — max retries exceeded, routing to error handler')
        return 'error'

    logger.info('[Router] Executor succeeded — routing to response agent')
    return 'response'
