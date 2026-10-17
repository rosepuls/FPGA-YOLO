import os
import json
import xml.etree.ElementTree as ET
import shutil

# 1. 定义路径 (根据你提供的目录结构)
base_dir = r"D:\desktop\FPGA\Project\PaddleDetection\dataset\coco"
xml_dir = os.path.join(base_dir, "bucongan")
train_json_path = os.path.join(base_dir, r"annotations\instances_train2017.json")
val_json_path = os.path.join(base_dir, r"annotations\instances_val2017.json")

def fix_coco_annotations(json_path, xml_dir):
    if not os.path.exists(json_path):
        print(f"找不到 JSON 文件: {json_path}")
        return

    print(f"========== 开始处理: {os.path.basename(json_path)} ==========")
    
    # 备份原文件以防万一
    backup_path = json_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy(json_path, backup_path)
        print(f"已创建备份: {backup_path}")

    # 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 建立类别映射字典: class_name -> category_id
    category_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
    
    # 建立图片映射字典: file_name -> image_id
    image_map = {img['file_name']: img['id'] for img in coco_data['images']}
    
    # 获取当前 JSON 中最大的 annotation id，避免 id 冲突
    max_ann_id = 0
    if coco_data['annotations']:
        max_ann_id = max([ann['id'] for ann in coco_data['annotations']])

    updated_images_count = 0
    new_boxes_count = 0

    # 2. 遍历 XML 文件夹
    for xml_filename in os.listdir(xml_dir):
        if not xml_filename.endswith('.xml'):
            continue
        
        # 仅处理包含特定关键字的文件
        if 'zang_wu' not in xml_filename and 'zhen_kong' not in xml_filename:
            continue

        # 推导对应的图片文件名 (假设是 .jpg，如果你的原图是 png，请修改这里)
        image_filename = xml_filename.replace('.xml', '.jpg')
        
        # 如果这张图片不在当前的 JSON (训练集或验证集) 中，直接跳过
        if image_filename not in image_map:
            continue
        
        image_id = image_map[image_filename]
        
        # 3. 清理旧标注：剔除该 image_id 对应的所有错误/合并的标注
        original_ann_count = len(coco_data['annotations'])
        coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['image_id'] != image_id]
        deleted_count = original_ann_count - len(coco_data['annotations'])

        # 4. 解析 XML，添加真实的多目标标注
        xml_path = os.path.join(xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes_added_for_this_img = 0
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in category_map:
                print(f"警告: 类别 '{class_name}' 不在 COCO categories 中，跳过该框。")
                continue
            
            cat_id = category_map[class_name]
            bndbox = obj.find('bndbox')
            
            # 提取坐标并计算 COCO 需要的格式 [x, y, width, height]
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            width = xmax - xmin
            height = ymax - ymin
            
            max_ann_id += 1
            new_ann = {
                "id": max_ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            }
            coco_data['annotations'].append(new_ann)
            boxes_added_for_this_img += 1
            new_boxes_count += 1
        
        updated_images_count += 1
        print(f"修复: {image_filename} | 删除了 {deleted_count} 个旧大框，提取了 {boxes_added_for_this_img} 个独立标注")

    # 5. 保存修复后的 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4) # indent=4 保证格式依然易读
    
    print(f"处理完成！该集合共修复了 {updated_images_count} 张图片的标注，新增了 {new_boxes_count} 个独立框。\n")

# 执行修复
fix_coco_annotations(train_json_path, xml_dir)
fix_coco_annotations(val_json_path, xml_dir)