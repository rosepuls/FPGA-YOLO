import os
import shutil
import random
import json
import xml.etree.ElementTree as ET

# --- 基础路径配置 ---
BASE_DIR = r"D:\desktop\FPGA\Project\PaddleDetection\dataset\coco"
IMG_DIR = os.path.join(BASE_DIR, "single")
XML_DIR = os.path.join(BASE_DIR, "singleannotations")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train2017")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val2017")
ANNO_DIR = os.path.join(BASE_DIR, "annotations")

# 你已经有的 JSON 文件路径
TRAIN_JSON = os.path.join(ANNO_DIR, "instances_train2017.json")
VAL_JSON = os.path.join(ANNO_DIR, "instances_val2017.json")

def append_to_existing_coco(img_list, target_img_dir, target_json_path):
    """安全地将新数据追加到现有的 COCO JSON 文件中"""
    
    # 1. 检查目标 JSON 是否存在，如果存在则读取，否则报错退出（确保不覆盖）
    if not os.path.exists(target_json_path):
        print(f"❌ 错误: 找不到原有的 JSON 文件 {target_json_path}，请检查路径！")
        return
        
    print(f"📂 正在加载现有 JSON: {os.path.basename(target_json_path)}")
    with open(target_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
        
    # 2. 从现有的 JSON 中提取类别映射 (确保新旧 ID 绝对一致)
    category_map = {cat['name']: cat['id'] for cat in coco_data.get('categories', [])}
    if not category_map:
        print("⚠️ 警告: 你的原 JSON 中没有 categories 字段！请确认文件是否完整。")
        return
        
    # 3. 寻找当前最大的 image_id 和 annotation_id，防止 ID 冲突！
    current_img_ids = [img['id'] for img in coco_data.get('images', [])]
    current_anno_ids = [ann['id'] for ann in coco_data.get('annotations', [])]
    
    next_img_id = max(current_img_ids) + 1 if current_img_ids else 1
    next_anno_id = max(current_anno_ids) + 1 if current_anno_ids else 1
    
    print(f"   -> 原有图片数: {len(current_img_ids)}，原有标注数: {len(current_anno_ids)}")
    print(f"   -> 新图片 ID 将从 {next_img_id} 开始，新标注 ID 将从 {next_anno_id} 开始。")

    added_img_count = 0
    
    # 4. 开始追加新数据
    for img_name in img_list:
        base_name, _ = os.path.splitext(img_name)
        src_img_path = os.path.join(IMG_DIR, img_name)
        src_xml_path = os.path.join(XML_DIR, base_name + '.xml')
        
        if not os.path.exists(src_xml_path):
            continue
            
        # 复制图片到目标文件夹 (train2017 / val2017)
        dst_img_path = os.path.join(target_img_dir, img_name)
        shutil.copy(src_img_path, dst_img_path)
        
        # 解析 XML
        tree = ET.parse(src_xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 追加图片信息到 existing JSON
        coco_data['images'].append({
            "id": next_img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        # 处理 Bounding Box 并追加
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in category_map:
                print(f"⚠️ 跳过未知类别 '{name}' (在图片 {img_name} 中)。由于原 JSON 没有这个类，请检查拼写。")
                continue
                
            cat_id = category_map[name]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            w = xmax - xmin
            h = ymax - ymin
            area = w * h
            
            coco_data['annotations'].append({
                "id": next_anno_id,
                "image_id": next_img_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, w, h],
                "area": area,
                "segmentation": [],
                "iscrowd": 0
            })
            next_anno_id += 1
            
        next_img_id += 1
        added_img_count += 1
        
    # 5. 安全写回 JSON
    with open(target_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 成功向 {os.path.basename(target_json_path)} 追加了 {added_img_count} 张图片的数据！\n")

def main():
    print("🚀 开始安全追加数据集...")
    
    # 确保目标文件夹存在
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    
    valid_exts = ['.jpg', '.jpeg', '.png']
    img_files = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if len(img_files) == 0:
        print("⚠️ single 文件夹下没有找到图片。")
        return
        
    # 随机打乱并 8:2 划分
    random.shuffle(img_files)
    split_idx = int(len(img_files) * 0.8)
    train_imgs = img_files[:split_idx]
    val_imgs = img_files[split_idx:]
    
    print(f"📊 待追加的新数据按 8:2 划分 -> 训练集: {len(train_imgs)} 张，验证集: {len(val_imgs)} 张\n")
    
    # 分别追加
    append_to_existing_coco(train_imgs, TRAIN_IMG_DIR, TRAIN_JSON)
    append_to_existing_coco(val_imgs, VAL_IMG_DIR, VAL_JSON)
    
    print("🎉 所有追加操作完美结束！你的原数据非常安全。")

if __name__ == '__main__':
    main()