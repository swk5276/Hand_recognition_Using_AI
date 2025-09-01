from django.urls import path
from . import views

urlpatterns = [
    path("mlp/", views.visualize_mlp, name="visualize_mlp"),
    path("cnn/", views.visualize_cnn, name="visualize_cnn"),
    path("single/", views.visualize_single, name="visualize_single"),  # ✅ 추가
]
