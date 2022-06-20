import cv2 
import numpy as np

top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"trousers"] 
# 경고 -> 위에서 쓰인 category list와 다른 목적으로 상의와 아우터 그리고 하의가 추가된거.

def mask_images(boxes,key,territory): #이 과정은 따른 곳에서 미리 해줄 생각 95~107코드는 추천세트 이미지 만드는 곳     
    #한번에 저장하고 싶으면 이 위치에 넣어줘야한다.
    #img = cv2.imread("1542.jpg")
    #im = np.zeros(img.shape, dtype = np.uint8)
    # i.e. 3 or 4 depending on your img
    for j in range(len(key)):
      
      img = cv2.imread("testman1.jpg")
      im = np.zeros(img.shape, dtype = np.uint8)
      channel_count = img.shape[2] 

      #x_y_keys = list(box_data[key[j]][0].keys())

      #if len(x_y_keys)<=20: # --------------------- 무식한 방식. 이미지의 면적이 적으면 뺀다. threshold 같은 역할 
      #  continue
      h=np.int32( territory[key[j]][0]['세로'])
      w = np.int32( territory[key[j]][0]['가로'])
      if (float(w)/h < 0.29):
        continue
      #box_loc = [0 for _ in range(len(x_y_keys))]
      print(key[j])
      ignore_mask_color = (255,)*channel_count
      
      l= (np.array(boxes[key[j]]).astype(np.int32))
      #print(l)
      cv2.fillPoly(im, [l], ignore_mask_color)
      masked_img = cv2.bitwise_not(img, im) # bitwise 연산 

      y_low = np.int32(territory[key[j]][0]['Y좌표'])
      y_high = y_low +np.int32( territory[key[j]][0]['세로'])
      x_low = np.int32(territory[key[j]][0]['X좌표'])
      x_high = x_low + np.int32(territory[key[j]][0]['가로'])
      
      transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
      transparent[:,:,0:3] = img
      im = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
      print(im.shape)
      transparent[:, :, 3] = im
      transparent = transparent[y_low:y_high, x_low:x_high]
      #print(masked_img)
      if key[j] in top_category:
        print(key[j])
        cv2.imwrite("top/result_test"+str(1)+".png",transparent)
      else:
        print(key[j])
        cv2.imwrite("bottom/result_test"+str(1)+".png",transparent)


with open('testman.json')  as json_file:
    data =json.load(json_file)
    box_data = {}
    territory = {}
    key_data = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].keys())

    key = []
    for i in range(1, len(key_data)):
        if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][key_data[i]] != [{}]:
            #print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'])
            if (data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]] != [{}]): 
              key.append(key_data[i])
              box_data[key_data[i]] = []
              box_data[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]][0]['좌표'])
              territory[key_data[i]] = []
              territory[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'][key_data[i]])
    print(box_data)
    print(territory)
    if (box_data != {}):
      mask_images(box_data,key, territory)
    