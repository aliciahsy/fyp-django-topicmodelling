from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('videodatabase/', views.view_video, name='view_video'),
    path('create/', views.create_video, name='create_video'),
    path('update/<int:pk>/', views.update_video, name='update_video'),
    path('delete/<int:pk>/', views.delete_video, name='delete_video'),
    path('details/<int:pk>/', views.video_details, name='video_details'),
    path('video_cards/', views.video_cards, name='video_cards'),
    path('user_videodetails/<int:video_id>/', views.user_videodetails, name='user_videodetails'),
]