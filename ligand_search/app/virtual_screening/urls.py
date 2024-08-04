from django.urls import path
from virtual_screening import views


urlpatterns = [
    path('', views.index, name='index')
]
