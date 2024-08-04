from django.urls import path
from virtual_screening import views


urlpatterns = [
    path('', views.search, name='search'),
    path('manage-ligand/', views.manageLigand, name='manage-ligand')
]
