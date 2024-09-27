from django.urls import path
from virtual_screening import views

urlpatterns = [
    path('', views.search, name='search'),
    path('manage-ligand/', views.manageLigand, name='manage-ligand'),
    path('results_list/', views.results_list, name="results_list"),
    path('results_list/<int:result_id>/', views.result_detail, name="result_detail"),
    path('demo/', views.demo, name="demo"),
]
