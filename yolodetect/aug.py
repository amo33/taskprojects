from cProfile import label
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd 
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
path = "./labelImg/data/train/"

file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".jpg")]
file_list_box = [file for file in file_list if file.endswith(".txt")]
file_list_img = sorted(file_list_img)
file_list_box = sorted(file_list_box)
print(file_list_box)
print(file_list_img)
IMAGE_SIZE= 600
train_transforms = A.Compose(
    [
        #A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.ShiftScaleRotate(rotate_limit=50, p=1,
        border_mode=cv2.BORDER_CONSTANT),
    ],
    bbox_params=A.BboxParams(format='yolo',
    min_visibility=0.4, label_fields=[],),
)
def load_image_into_numpy_array(image):
    (img_width, img_height) = image.size
    return np.array(image.getdata()).reshape((img_height, img_width, 3)).astype(np.uint8)

def myFig(img, label):
    fig ,ax = plt.subplots()
    ax.imshow(img)

    if len(label) > 0:
        dw = img.shape[1]
        dh = img.shape[0]
        print(dw, dh)
        x1 = (label[0][0] - label[0][2]/2 )  * dw
        y1 = (label[0][1] - label[0][3]/2 ) * dh 
        w = label[0][2] * dw 
        h = label[0][3]* dh 
        
        #x1 = label[0][0]
        #y1 = label[0][1]
        #w = label[0][2]
        #h = label[0][3]
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor = 'r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
for i in range(9):
    image = Image.open(path+file_list_img[i])
    img = load_image_into_numpy_array(image)
    gt = np.loadtxt(path + file_list_box[i], delimiter = ' ', ndmin=2)
    dt = np.roll(gt, 4, axis=1).tolist()
    #gt = gt.tolist()
    print(gt)
    transformed = train_transforms(image=img, bboxes=dt)
    print(transformed['bboxes'])
    lst = transformed['bboxes']
    res_image = transformed['image']
    myFig(res_image, lst)
    newpath = file_list_img[i][:-4]
    print(newpath)
    #cv2.imwrite(path+str(i)+newpath+'.jpg', res_image)
    #np.savetxt(path+str(i)+newpath+'.txt',gt,delimiter=' ')
    #data_df.to_csv(path+str(i)+newpath+'.txt', sep=' ', index=False, header=False)


'''
seq=iaa.Sequential([
    iaa.Affine(
        rotate = 45)])

def load_image_into_numpy_array(image):
    (img_width, img_height) = image.size
    return np.array(image.getdata()).reshape((img_height, img_width, 3)).astype(np.uint8)

for j in range(1): #len(file_list_img)
    for i in range(1):
        # img = cv2.imread(path+file_list_img[i])
        image = Image.open(path+file_list_img[i])
        img = load_image_into_numpy_array(image)
        gt = np.loadtxt(path + file_list_box[i], delimiter = ' ').reshape(-1,5)
        cv2.imshow("show", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #index = gt[:0].astype(np.uint8)
        #input_img = img[np.newaxis, :, :, :]
        input_img = img
        bbox = []
        label_true = []
        box = []
        label_true.append(gt[0][0])
        x_1 = (gt[0][1] - gt[0][3]/2) * 600
        y_1 = (gt[0][2] - gt[0][4]/2) * 400
        x_2 = (gt[0][1] + gt[0][3]/2) * 600
        y_2 = (gt[0][2] + gt[0][4]/2) * 400
        box.append(BoundingBox(x1 = x_1, y1 = y_1, x2 = x_2, y2 = y_2, label=int(gt[0][0])))
        
        bbs = BoundingBoxesOnImage(
            box, 
            shape=img.shape
        )
        draw2 = box[0].draw_on_image(input_img, color=[0, 0, 255])
        cv2.imshow("show", draw2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        aug_img , aug_bbox = seq(image = img, bounding_boxes = bbs)
        
        bbox.append(aug_bbox[0])
        print(gt[0])
        print(gt[0][1])
        print(aug_bbox[0])
        print(aug_bbox)
        draw1 = aug_bbox.draw_on_image(aug_img, color=[0, 0, 255])
        #res = np.hstack((draw2, draw1))
        cv2.imshow("show", draw1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        for k in range(1, gt.shape[0]):
            input_img = img
            label_true.append(gt[k][0])
            print(gt[k])
            box= [BoundingBox(x1 = gt[k][1], y1 = gt[k][2], x2 = gt[k][3], y2 = gt[k][4], label=int(gt[k][0]))]
            bbs = BoundingBoxesOnImage(
                box,
                shape = input_img.shape
            )
            aug_img, aug_bbox = seq(images = input_img, bounding_boxes = box[k])
            bbox.append(aug_bbox[0])
        
        print(bbox[0])
        print(bbox[0][0:2])

        data_df = pd.DataFrame(bbox[0][0:2].reshape(1,-1))
        for m in range(1,gt.shape[0]):
            data_df2 = pd.DataFrame(bbox[m][0:2].reshape(1,-1))
            data_df = pd.concat([data_df, data_df2])
        label_df = pd.DataFrame(label_true, dtype=np.uint8)
        data_df.insert(0,-1,label_df)
        print(data_df)
        newpath = file_list_img[i][:-4]
        print(newpath)
        
        #cv2.imwrite(path+str(j)+newpath+'.jpg', aug_img)
        #np.savetxt(path+str(j)+newpath+'.txt',gt,delimiter=' ')
        #data_df.to_csv(path+str(j)+newpath+'.txt', sep=' ', index=False, header=False)
'''
'''

tree = 0

val = tree.findall("object/bndbox")
val_xmin = [x.findtext("xmin")for x in val]
val_ymin = [x.findtext("ymin")for x in val]
val_xmax = [x.findtext("xmax")for x in val]
val_ymax = [x.findtext("ymax")for x in val]
# augmentation 하고자 하는 사진에, xml 파싱데이터로 bounding box 생성
input_img = img[np.newaxis, :, :, :]
bbox = [ia.BoundingBox(x1=float(val_xmin[0]), y1=float(val_ymin[0]),
                       x2=float(val_xmax[0]), y2=float(val_ymax[0]),label=label_true[0])]
bbox1 = [ia.BoundingBox(x1=float(val_xmin[1]), y1=float(val_ymin[1]),
                       x2=float(val_xmax[1]), y2=float(val_ymax[1]),label=label_true[1])]
print(input_img.shape)
# Aug 연산 예시
seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)),
    iaa.Affine(
        scale=(0.5, 0.7),
        rotate=45)])
# Aug연산을 이용해서 bounding box도 같이 aug

org_img, org_bbox, org_bbox1 = input_img, bbox, bbox1
print('label1= ',  org_bbox[0])
print('label2= ', org_bbox1[0])
org = org_bbox[0].draw_on_image(org_img[0], size=5, color=[0, 255, 0])
input_img1 = org[np.newaxis, :, :, :]
org1 = org_bbox1[0].draw_on_image(input_img1[0],size=5, color=[0, 255, 0])
# 연산 이미지 처리
aug_img1, aug_bbox0 = seq(images=input_img, bounding_boxes=bbox)
aug_img1, aug_bbox1 = seq(images=input_img1, bounding_boxes=bbox1)
draw1 = aug_bbox1[0].draw_on_image(aug_img1[0], size=5, color=[0, 255, 0])
# 연산되어 변형된 labeling data를 표시해준다.
print('aug_label1 = ', aug_bbox0[0])
print('aug_label2 = ', aug_bbox1[0])
res = np.hstack((org1, draw1))
cv2.imshow('res', res)
cv2.imwrite('result.jpg',org1)
'''