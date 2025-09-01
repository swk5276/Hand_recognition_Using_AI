# visualization_project/visualization_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("visualize/", include("visualizer.urls")),
]
