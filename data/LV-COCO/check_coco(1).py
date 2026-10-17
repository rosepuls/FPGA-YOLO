import json
import os
import glob

# --- 配置路径 ---
BASE_DIR = os.getcwd()
JSON_DIR = os.path.join(BASE_DIR, "annotations")

def check_coco_json(json_path):
    print(f"\n" + "="*50)
    print(f"🔍 正在校验: {os.path.basename(json_path)}")
    print("="*50)

    if not os.path.exists(json_path):
        print(f"❌ 错误: 找不到文件 {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ 错误: JSON 文件解析失败，可能已损坏！报错信息: {e}")
        return

    # 1. 检查基础关键字
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in dataset:
            print(f"❌ 致命错误: 缺失必须的顶层字段 '{key}'")
            return
    print("✅ 基础结构检查通过 (包含 images, annotations, categories)")

    # 2. 检查 Images 列表
    images = dataset.get('images', [])
    image_ids = set()
    duplicate_images = 0
    for img in images:
        if 'id' not in img or 'file_name' not in img or 'width' not in img or 'height' not in img:
            print(f"❌ 错误: 图片信息缺失必须字段 (id, file_name, width, height)。当前图片: {img.get('file_name', '未知')}")
            return
        if img['id'] in image_ids:
            duplicate_images += 1
        image_ids.add(img['id'])
    
    if duplicate_images > 0:
        print(f"❌ 错误: 发现 {duplicate_images} 个重复的图片 ID！")
    else:
        print(f"✅ 图片列表检查通过，共 {len(images)} 张图片，无重复 ID。")

    # 3. 检查 Annotations 列表
    annotations = dataset.get('annotations', [])
    annotated_image_ids = set()
    invalid_annos = 0
    for ann in annotations:
        if 'id' not in ann or 'image_id' not in ann or 'category_id' not in ann or 'bbox' not in ann:
            print(f"❌ 错误: 标注信息缺失必须字段 (id, image_id, category_id, bbox)。标注 ID: {ann.get('id', '未知')}")
            return
        
        img_id = ann['image_id']
        if img_id not in image_ids:
            invalid_annos += 1
        else:
            annotated_image_ids.add(img_id)

    if invalid_annos > 0:
        print(f"❌ 错误: 发现 {invalid_annos} 个标注框找不到对应的图片！(幽灵标注)")
    else:
        print(f"✅ 标注列表检查通过，共 {len(annotations)} 个标注框。")

    # 4. 负样本（纯背景图）统计验证
    background_images_count = len(image_ids) - len(annotated_image_ids)
    print("\n📊 --- 数据集统计摘要 ---")
    print(f"  总类别数 : {len(dataset.get('categories', []))}")
    print(f"  总图片数 : {len(images)}")
    print(f"  总标注数 : {len(annotations)}")
    print(f"  包含目标的图片数 : {len(annotated_image_ids)}")
    print(f"  纯背景图(负样本)数 : {background_images_count}  <--- (请确认这是否符合你刚添加的负样本数量)")
    
    if background_images_count > 0:
        print("  💡 提示: 存在纯背景图，这非常有助于缓解模型的误检(Overkill)问题！")

def main():
    # 查找 annotations 文件夹下所有的 json 文件
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    
    if not json_files:
        print(f"⚠️ 在 {JSON_DIR} 目录下没有找到任何 JSON 文件。")
        return

    for json_file in json_files:
        # 跳过我们刚才生成的备份文件
        if not json_file.endswith('.bak'):
            check_coco_json(json_file)

if __name__ == "__main__":
    main()