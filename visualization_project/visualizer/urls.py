from django.urls import path
from . import views

urlpatterns = [
    path('', views.visualize_predictions, name='visualize_predictions'),
]
