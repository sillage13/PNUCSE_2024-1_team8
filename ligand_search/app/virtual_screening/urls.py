from django.urls import path
from virtual_screening import views

urlpatterns = [
    path('', views.search, name='search'),
    path('manage-ligand/', views.manageLigand, name='manage-ligand'),
    path('results-list/', views.results_list, name='results-list'),
    path('results-list/<int:result_id>/', views.result_detail, name='result-detail'),
    path('demo/', views.demo, name='demo'),
    path('processing/', views.processing, name='processing'),
    path('perform-task/', views.performTask, name='perform-task'),
    path('get-task-status/', views.getTaskStatus, name='get-task-status'),
    path('manage-ligand/import-from-db/', views.import_from_db, name='import-from-db'),
]