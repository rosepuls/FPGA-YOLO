import os
import json
import cv2
import numpy as np

# ================= 配置区 =================
# 切换这里即可：'train' 或 'val'
SET_TYPE = 'val' 

SOURCE_DIR = 'hard_negatives'    # 存放误判原图的文件夹
BASE_DIR = './'                  # 你的数据集根目录

if SET_TYPE == 'train':
    JSON_PATH = os.path.join(BASE_DIR, 'annotations/instances_train2017.json')
    SAVE_IMG_DIR = os.path.join(BASE_DIR, 'train2017')
    PREFIX = 'train_hard_neg'
else:
    JSON_PATH = os.path.join(BASE_DIR, 'annotations/instances_val2017.json')
    SAVE_IMG_DIR = os.path.join(BASE_DIR, 'val2017')
    PREFIX = 'val_hard_neg'
# ==========================================

def augment(img, mode):
    if mode == 0: return cv2.flip(img, 1) # 水平翻转
    if mode == 1: return cv2.GaussianBlur(img, (5, 5), 0) # 模糊
    if mode == 2: return cv2.convertScaleAbs(img, alpha=0.8, beta=-20) # 调暗
    return img

with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

current_img_id = max([img['id'] for img in coco_data['images']]) + 1

for img_name in os.listdir(SOURCE_DIR):
    if not img_name.lower().endswith(('.jpg', '.png')): continue
    
    img = cv2.imread(os.path.join(SOURCE_DIR, img_name))
    
    # 训练集生成 8 张增强图，验证集生成 2 张即可
    num_aug = 8 if SET_TYPE == 'train' else 2
    
    for i in range(num_aug):
        aug_img = augment(img, i % 3)
        new_name = f"{PREFIX}_{i}_{img_name}"
        cv2.imwrite(os.path.join(SAVE_IMG_DIR, new_name), aug_img)
        
        coco_data['images'].append({
            "file_name": new_name,
            "height": aug_img.shape[0],
            "width": aug_img.shape[1],
            "id": current_img_id
        })
        current_img_id += 1

with open(JSON_PATH, 'w') as f:
    json.dump(coco_data, f)

print(f"已完成 {SET_TYPE} 集的负样本添加！")