from glob import glob 


train_img_list = glob('/home/triplet_tank/yolodetection/yolov5/custom_datasets/images/train/*.jpg') 
valid_img_list = glob('/home/triplet_tank/yolodetection/yolov5/custom_datasets/images/valid/*.jpg')

with open('./train.txt','w') as f:
    f.write('\n'.join(train_img_list)+ '\n')
with open('./valid.txt','w') as f:
    f.write('\n'.join(valid_img_list)+'\n')