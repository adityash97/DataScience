from django.db import models


class Workflow(models.Model):
    class WorkflowType(models.TextChoices):
        RESEARCH = 'research', 'Research'
        ANALYSIS = 'analysis', 'Analysis'
        SUMMARIZATION = 'summarization', 'Summarization'
        CUSTOM = 'custom', 'Custom'

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    workflow_type = models.CharField(
        max_length=50,
        choices=WorkflowType.choices,
        default=WorkflowType.CUSTOM,
    )
    configuration = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class WorkflowExecution(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        RETRYING = 'retrying', 'Retrying'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    workflow = models.ForeignKey(
        Workflow,
        on_delete=models.CASCADE,
        related_name='executions',
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    input_payload = models.JSONField(default=dict)
    output_payload = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.workflow.name} — {self.status} ({self.id})'
