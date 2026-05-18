from rest_framework import serializers

from .models import Workflow, WorkflowExecution


class WorkflowSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workflow
        fields = [
            'id',
            'name',
            'description',
            'workflow_type',
            'configuration',
            'is_active',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class WorkflowExecutionSerializer(serializers.ModelSerializer):
    workflow_name = serializers.CharField(source='workflow.name', read_only=True)

    class Meta:
        model = WorkflowExecution
        fields = [
            'id',
            'workflow',
            'workflow_name',
            'status',
            'input_payload',
            'output_payload',
            'error_message',
            'started_at',
            'completed_at',
            'created_at',
        ]
        read_only_fields = [
            'id',
            'workflow_name',
            'status',
            'output_payload',
            'error_message',
            'started_at',
            'completed_at',
            'created_at',
        ]
