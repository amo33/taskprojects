from curses.ascii import SP
from django.urls import path, re_path
from .views import UserView

urlpatterns = [
    path('users', UserView.as_view()),
    path('users/<int:userid>', UserView.as_view()), # userid이 언제는 들어오고 언제는 안 들어오게 할순 없나
]
