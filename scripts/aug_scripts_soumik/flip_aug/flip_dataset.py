# final SU seq reverse augmentation prepatation 
#  image path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/frame066.png


import os
import cv2

dataset_dir = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
flip_dataset_dir = dataset_dir+"_flip"

# os.makedirs(flip_dataset_dir, exist_ok=True)
# for img_file in os.listdir(dataset_dir):
#     #load the image file
#     img_file_path = os.path.join(dataset_dir, img_file)
#     #read the image file
#     img = cv2.imread(img_file_path)
#     #flip the image
#     img_flipv = cv2.flip(img, 0)
#     img_fliph = cv2.flip(img, 1)
#     img_flipvh = cv2.flip(img, -1)
#     #save the flipped image
#     img_flipv_file_path = os.path.join(flip_dataset_dir, img_file.split(".")[0]+"_flipv.png")
#     img_fliph_file_path = os.path.join(flip_dataset_dir, img_file.split(".")[0]+"_fliph.png")
#     img_flipvh_file_path = os.path.join(flip_dataset_dir, img_file.split(".")[0]+"_flipvh.png")
#     cv2.imwrite(img_flipv_file_path, img_flipv)
#     cv2.imwrite(img_fliph_file_path, img_fliph)
#     cv2.imwrite(img_flipvh_file_path, img_flipvh)




os.makedirs(flip_dataset_dir+'v', exist_ok=True)
os.makedirs(flip_dataset_dir+'h', exist_ok=True)
os.makedirs(flip_dataset_dir+'vh', exist_ok=True)
for img_file in os.listdir(dataset_dir):
    #load the image file
    img_file_path = os.path.join(dataset_dir, img_file)
    #read the image file
    img = cv2.imread(img_file_path, 0)
    #flip the image
    print(img.shape)
    img_flipv = cv2.flip(img, 0)
    print(img_flipv.shape)
    img_fliph = cv2.flip(img, 1)
    img_flipvh = cv2.flip(img, -1)
    #save the flipped image
    img_flipv_file_path = os.path.join(flip_dataset_dir+'v', img_file.split(".")[0]+".png")
    img_fliph_file_path = os.path.join(flip_dataset_dir+'h', img_file.split(".")[0]+".png")
    img_flipvh_file_path = os.path.join(flip_dataset_dir+'vh', img_file.split(".")[0]+".png")
    cv2.imwrite(img_flipv_file_path, img_flipv)
    cv2.imwrite(img_fliph_file_path, img_fliph)
    cv2.imwrite(img_flipvh_file_path, img_flipvh)
    

    
