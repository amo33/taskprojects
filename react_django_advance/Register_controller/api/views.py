from unicodedata import category
from itsdangerous import Serializer
from rest_framework import generics, status
from .serializers import Userserializer, createUserSerializer
from .models import USer
from rest_framework.response import Response
from pathlib import Path
import pandas as pd 
import uuid
from PIL import Image
import os 
from rest_framework.decorators import api_view

class UserView(generics.CreateAPIView):
    serializer_class = createUserSerializer
    def post(self, request): 
        serializer = createUserSerializer(data = request.data)
        if os.stat("data.tsv").st_size == 0: # text가 비었다면 column 명이 없다면 
            f = open("data.tsv", "a")
            line =['{}\t{}\t{}\t{}\t{}\n'.format('id', 'user_id','username','age','image_path')]
            f.writelines(line)
            f.close()
        if serializer.is_valid():
            username = serializer.data.get('username')
            age = serializer.data.get('age')
            if request.FILES.get('image')!= None:
                image = request.FILES.get('image')
                image_for_thumb = Image.open(image) 
                size = (128, 128)
                image_for_thumb.thumbnail(size)
                image_path = "/thumbnailed/"+str(uuid.uuid4())+str(image)
                image_for_thumb.save(Path('user_register/public'+image_path))
            else:
                image_path = None
            user = USer(username = username, age = age, image_path = image_path)
                
            user.save()
            f = open("data.tsv", "a") #저장
            userline=['{}\t{}\t{}\t{}\t{}\n'.format(user.id,user.user_id,user.username,user.age,user.image_path)]
            f.writelines(userline)
            f.close()
                
            return Response(Userserializer(user).data, status= status.HTTP_201_CREATED)

        return Response({'Bad Request': 'Invalid data...'}, status= status.HTTP_400_BAD_REQUEST)
    serializer_class = Userserializer
    def get(self, request ,userid=None): 
        method = request.GET.get('method',None)
        print(method)
        print(userid)
        if method == 'db': # DB일때
            queryset = USer.objects.all() if userid == None else USer.objects.filter(id=userid)
            user = []
            for element in queryset:
                user.append({
                    "id" : element.id,
                    "username": element.username,
                    "age": element.age,
                    "user_id": element.user_id,
                    "Image_flag" : 1 if element.image_path != None else 0,
                    "image_path" : element.image_path if id != None else None,
                })
            return Response(user , status=status.HTTP_200_OK)
        elif method == 'text': # text 일때
            df = pd.read_csv('data.tsv', sep='\t')
            df_copy = df.copy()
            if userid ==None:
                df_copy['Image_flag'] = df_copy['image_path'].apply(lambda x :1 if x != 'None' else 0)
                df_copy.drop('image_path', axis=1, inplace=True)
                user = df_copy.to_dict('records')
            elif userid != None:
                    
                user = []
                idx =df_copy.index[df_copy['id'] == int(userid)].tolist()[0]
                user_info = df_copy.loc[idx]
                user.append({
                    "user_id" : user_info.user_id,
                    "username": user_info.username,
                    "age":user_info.age,
                    "image_path":user_info.image_path if user_info.image_path != 'None' else None,
                })
            return Response(user, status = status.HTTP_200_OK)
        else: 
            return Response({'No request': 'Invalid parameter'}, status= status.HTTP_204_NO_CONTENT)
    '''
@api_view(['GET'])
def show_info_api_view(request): 
    method = request.GET.get('method', None)
    id = request.GET.get('user_id',None) # detail인지 목록페이지인지 판단하는 기준 
    if method == 'showdb': # DB일때
        queryset = USer.objects.all() if id == None else USer.objects.filter(id=id)
    elif method == 'showlist': # text 일때
        df = pd.read_csv('data.tsv', sep='\t')
        df_copy = df.copy()
        if id == None:
            df_copy['Image_flag'] = df_copy['image_path'].apply(lambda x :1 if x != 'None' else 0)
            df_copy.drop('image_path', axis=1, inplace=True)
            user = df_copy.to_dict('records')
            return Response(user , status=status.HTTP_200_OK)
        elif id != None:
            queryset = []
            idx =df_copy.index[df_copy['id'] == int(id)].tolist()[0]
            queryset.append(df_copy.loc[idx])
    user = []
    for element in queryset:
        user.append({
            "id" : element.id,
            "username": element.username,
            "age": element.age,
            "user_id": element.user_id,
            "Image_flag" : 1 if element.image_path != None else 0,
            "image_path" : element.image_path if id != None else None,
        })
        return Response(user, status = status.HTTP_200_OK)
    else: 
        return Response({'No request': 'Invalid parameter'}, status= status.HTTP_204_NO_CONTENT)
''' 