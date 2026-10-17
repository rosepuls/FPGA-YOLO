import os
import json
import xml.etree.ElementTree as ET

# 1. 定义路径
base_dir = r"D:\desktop\FPGA\Project\PaddleDetection\dataset\coco"
xml_dir = os.path.join(base_dir, "bucongan")
train_json_path = os.path.join(base_dir, r"annotations\instances_train2017.json")
val_json_path = os.path.join(base_dir, r"annotations\instances_val2017.json")

def verify_coco_annotations(json_path, xml_dir):
    if not os.path.exists(json_path):
        print(f"找不到文件，跳过: {json_path}")
        return

    print(f"\n========== 开始检测: {os.path.basename(json_path)} ==========")
    
    # 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    category_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
    image_map = {img['file_name']: img['id'] for img in coco_data['images']}
    
    # 预先统计 JSON 中每个 image_id 对应的标注数量，提高查询速度
    ann_counts = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        ann_counts[img_id] = ann_counts.get(img_id, 0) + 1

    total_checked = 0
    total_passed = 0
    total_failed = 0

    # 2. 遍历 XML 进行比对
    for xml_filename in os.listdir(xml_dir):
        if not xml_filename.endswith('.xml'):
            continue
        
        if 'zang_wu' not in xml_filename and 'zhen_kong' not in xml_filename:
            continue

        image_filename = xml_filename.replace('.xml', '.jpg')
        
        # 确定该图片是否在当前被检测的 JSON 集合中
        if image_filename not in image_map:
            continue 

        total_checked += 1
        image_id = image_map[image_filename]
        
        # 解析 XML 统计框的数量
        xml_path = os.path.join(xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        xml_box_count = 0
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in category_map:
                xml_box_count += 1
        
        # 获取 JSON 中实际拥有的框数量
        json_box_count = ann_counts.get(image_id, 0)
        
        # 3. 判断是否匹配
        if xml_box_count == json_box_count:
            total_passed += 1
        else:
            total_failed += 1
            print(f"[异常] {image_filename} 不匹配! XML有 {xml_box_count} 个框，但 JSON 却有 {json_box_count} 个。")

    # 4. 输出统计结果
    print(f"检测完成！该集合共核对 {total_checked} 张目标图片。")
    print(f"✅ 完全匹配: {total_passed} 张")
    if total_failed > 0:
        print(f"❌ 存在异常: {total_failed} 张 (请检查上方打印的异常文件)")
    elif total_checked > 0:
        print("🎉 恭喜！所有目标图片的标注数量均与 XML 完美对应！")
    else:
        print("⚠️ 没有找到需要核对的图片。")

# 运行验证
verify_coco_annotations(train_json_path, xml_dir)
verify_coco_annotations(val_json_path, xml_dir)