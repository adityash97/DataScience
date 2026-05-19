import logging
from datetime import datetime, timezone

from rest_framework.decorators import api_view
from rest_framework.request import Request

from ai.runner import run_workflow
from utils.response import error_response, success_response

from .models import Workflow, WorkflowExecution
from .serializers import WorkflowExecutionSerializer, WorkflowSerializer

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    'completed': WorkflowExecution.Status.COMPLETED,
    'retrying':  WorkflowExecution.Status.RETRYING,
    'failed':    WorkflowExecution.Status.FAILED,
}


# ---------------------------------------------------------------------------
# Workflow APIs
# ---------------------------------------------------------------------------

@api_view(['GET'])
def workflow_list(request: Request):
    workflows = Workflow.objects.filter(is_active=True)
    serializer = WorkflowSerializer(workflows, many=True)
    return success_response(data=serializer.data)


@api_view(['POST'])
def workflow_create(request: Request):
    serializer = WorkflowSerializer(data=request.data)
    if not serializer.is_valid():
        return error_response(message='Invalid data', errors=serializer.errors, status=400)
    serializer.save()
    logger.info('Workflow created: %s', serializer.data['name'])
    return success_response(data=serializer.data, message='Workflow created', status=201)


@api_view(['GET'])
def workflow_detail(request: Request, pk: int):
    try:
        workflow = Workflow.objects.get(pk=pk)
    except Workflow.DoesNotExist:
        return error_response(message='Workflow not found', status=404)
    serializer = WorkflowSerializer(workflow)
    return success_response(data=serializer.data)


# ---------------------------------------------------------------------------
# Execution APIs
# ---------------------------------------------------------------------------

@api_view(['POST'])
def workflow_run(request: Request, pk: int):
    try:
        workflow = Workflow.objects.get(pk=pk, is_active=True)
    except Workflow.DoesNotExist:
        return error_response(message='Workflow not found', status=404)

    input_payload = request.data.get('input_payload', {})
    user_input = input_payload.get('query', '')

    execution = WorkflowExecution.objects.create(
        workflow=workflow,
        status=WorkflowExecution.Status.RUNNING,
        input_payload=input_payload,
        started_at=datetime.now(timezone.utc),
    )

    result = run_workflow(
        workflow_id=workflow.id,
        execution_id=execution.id,
        user_input=user_input,
    )

    execution.status = _STATUS_MAP.get(result['status'], WorkflowExecution.Status.FAILED)
    execution.output_payload = result['output']
    execution.error_message = result.get('error', '')
    execution.completed_at = datetime.now(timezone.utc)
    execution.save()

    serializer = WorkflowExecutionSerializer(execution)
    response_data = {
        **serializer.data,
        'routing': {
            'retry_count': result['retry_count'],
            'final_step': result['current_step'],
            'status': result['status'],
        },
    }

    if result['success']:
        return success_response(data=response_data, message='Workflow executed successfully', status=201)
    return error_response(
        message='Workflow execution failed',
        errors={'detail': result.get('error'), 'retry_count': result['retry_count']},
        status=500,
    )


@api_view(['GET'])
def execution_list(request: Request):
    executions = WorkflowExecution.objects.select_related('workflow').all()
    serializer = WorkflowExecutionSerializer(executions, many=True)
    return success_response(data=serializer.data)


@api_view(['GET'])
def execution_detail(request: Request, pk: int):
    try:
        execution = WorkflowExecution.objects.select_related('workflow').get(pk=pk)
    except WorkflowExecution.DoesNotExist:
        return error_response(message='Execution not found', status=404)
    serializer = WorkflowExecutionSerializer(execution)
    return success_response(data=serializer.data)
