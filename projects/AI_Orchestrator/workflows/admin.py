from django.contrib import admin

from .models import Workflow, WorkflowExecution


@admin.register(Workflow)
class WorkflowAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'workflow_type', 'is_active', 'created_at']
    list_filter = ['workflow_type', 'is_active']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(WorkflowExecution)
class WorkflowExecutionAdmin(admin.ModelAdmin):
    list_display = ['id', 'workflow', 'status', 'started_at', 'completed_at', 'created_at']
    list_filter = ['status', 'workflow']
    search_fields = ['workflow__name']
    readonly_fields = ['created_at', 'started_at', 'completed_at']
