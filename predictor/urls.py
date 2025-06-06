from django.urls import path
from . import views

urlpatterns = [

    path('plot/', views.stock_plot, name='stock_plot'),
    path('predict/<str:company>/', views.stock_predict, name='stock_predict'),
    path('chart/<str:company>/', views.stock_chart, name='stock_chart'),
    path('dashboard/<str:company>/', views.stock_dashboard, name='stock_dashboard'),
    path('predict/', views.redirect_to_company, name='redirect_to_company'),

]
