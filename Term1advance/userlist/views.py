from curses.ascii import US
from django.shortcuts import render
from itsdangerous import Serializer
from rest_framework.response import Response
from .serializers import UserSerializer
from .models import User
from django.views.generic import TemplateView

from rest_framework.views import APIView
# Create your views here.

class UserView(APIView):
    serializer_class = UserSerializer

    def get(self, request):
        detail = [{"name": detail.name, "detail":detail.detail} for detail in React.objects.all()]
        return Response(detail)
    def post(self, request):

        serializer = UserSerializer(data=request.data)
        if serializer.is_valid(raise_exception= True):
            serializer.save()
            return Response(serializer.data)
