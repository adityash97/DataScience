import logging
from datetime import datetime, timezone

from rest_framework.decorators import api_view
from rest_framework.request import Request

from utils.response import error_response, success_response

from .models import Workflow, WorkflowExecution
from .serializers import WorkflowExecutionSerializer, WorkflowSerializer

logger = logging.getLogger(__name__)


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

    execution = WorkflowExecution.objects.create(
        workflow=workflow,
        status=WorkflowExecution.Status.COMPLETED,
        input_payload=input_payload,
        output_payload=_mock_execution_output(workflow, input_payload),
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )

    logger.info('Mock execution completed: workflow=%s execution=%s', workflow.name, execution.id)
    serializer = WorkflowExecutionSerializer(execution)
    return success_response(data=serializer.data, message='Workflow executed', status=201)


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


# ---------------------------------------------------------------------------
# Mock execution helper (replaced by LangGraph in a future milestone)
# ---------------------------------------------------------------------------

def _mock_execution_output(workflow: Workflow, input_payload: dict) -> dict:
    query = input_payload.get('query', '')
    return {
        'summary': f'Mock execution of "{workflow.name}" completed successfully.',
        'query': query,
        'agents': ['Planner', 'Retriever', 'Analyst', 'Critic', 'Final Response'],
        'result': 'This is a placeholder result. LangGraph integration is pending.',
    }
