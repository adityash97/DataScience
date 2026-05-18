from django.urls import path

from . import views

urlpatterns = [
    # Workflow endpoints
    path('', views.workflow_list, name='workflow-list'),
    path('create/', views.workflow_create, name='workflow-create'),
    path('<int:pk>/', views.workflow_detail, name='workflow-detail'),
    path('<int:pk>/run/', views.workflow_run, name='workflow-run'),

    # Execution endpoints
    path('executions/', views.execution_list, name='execution-list'),
    path('executions/<int:pk>/', views.execution_detail, name='execution-detail'),
]
