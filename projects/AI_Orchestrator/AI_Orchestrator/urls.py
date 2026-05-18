from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/health/', include('health.urls')),
    path('api/workflows/', include('workflows.urls')),
]
