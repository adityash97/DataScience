import logging

logger = logging.getLogger(__name__)

name = 'database'
description = (
    'Fetch stored workflow or execution records from the local database. '
    'Input: {"resource": "workflow"|"execution"|"recent_executions", "id": Optional[int], "limit": Optional[int]}.'
)


def run(payload: dict) -> dict:
    from workflows.models import Workflow, WorkflowExecution

    payload = payload or {}
    resource = payload.get('resource', 'recent_executions')
    logger.info('[database_tool] Resource: %s payload: %s', resource, payload)

    try:
        if resource == 'workflow':
            wf_id = payload.get('id')
            if wf_id is None:
                return {'tool': name, 'success': False, 'error': 'Missing id for workflow lookup.'}
            wf = Workflow.objects.filter(pk=wf_id).values('id', 'name', 'workflow_type', 'is_active').first()
            return {'tool': name, 'success': bool(wf), 'resource': resource, 'result': wf or {}}

        if resource == 'execution':
            ex_id = payload.get('id')
            if ex_id is None:
                return {'tool': name, 'success': False, 'error': 'Missing id for execution lookup.'}
            ex = WorkflowExecution.objects.filter(pk=ex_id).values(
                'id', 'workflow_id', 'status', 'created_at', 'completed_at'
            ).first()
            return {'tool': name, 'success': bool(ex), 'resource': resource, 'result': ex or {}}

        # default: recent executions
        limit = int(payload.get('limit', 5))
        rows = list(
            WorkflowExecution.objects
            .order_by('-created_at')
            .values('id', 'workflow_id', 'status', 'created_at')[:limit]
        )
        return {'tool': name, 'success': True, 'resource': 'recent_executions', 'result': rows, 'count': len(rows)}
    except Exception as exc:
        logger.exception('[database_tool] Query failed')
        return {'tool': name, 'success': False, 'error': str(exc)}
