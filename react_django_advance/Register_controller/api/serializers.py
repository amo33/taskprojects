from rest_framework import serializers
from .models import USer
class Userserializer(serializers.ModelSerializer):
    class Meta:
        model = USer
        fields = ('user_id','username', 'age', 'image_path')

class createUserSerializer(serializers.ModelSerializer): # post 다룬다. 

    class Meta:
        model = USer 
        fields = ('username','age') # post로 받고
        