from django.urls import path
from . import views
urlpatterns = [ path('', views.visualize_predictions, name='visualize_predictions'),path('draw/', views.draw_and_predict, name='draw_and_predict'), ]
