import os
import json

# --- 基础路径配置 (保持与之前一致) ---
BASE_DIR = r"D:\desktop\FPGA\Project\PaddleDetection\dataset\coco"
IMG_DIR = os.path.join(BASE_DIR, "single") # 这里放你要删除的图片

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train2017")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val2017")
ANNO_DIR = os.path.join(BASE_DIR, "annotations")

TRAIN_JSON = os.path.join(ANNO_DIR, "instances_train2017.json")
VAL_JSON = os.path.join(ANNO_DIR, "instances_val2017.json")

def remove_data_from_split(filenames_to_delete, img_dir, json_path):
    """从指定的图片文件夹和 JSON 标注中安全删除对应数据"""
    if not os.path.exists(json_path):
        print(f"⚠️ 跳过: 找不到 JSON 文件 {json_path}")
        return

    print(f"\n📂 正在检索: {os.path.basename(json_path)}")

    # 1. 读取原 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    original_img_count = len(coco_data.get('images', []))
    original_anno_count = len(coco_data.get('annotations', []))

    # 2. 找出需要删除的 image_id
    images_to_keep = []
    deleted_image_ids = set()
    deleted_filenames = set()

    for img in coco_data.get('images', []):
        if img['file_name'] in filenames_to_delete:
            deleted_image_ids.add(img['id'])
            deleted_filenames.add(img['file_name'])
        else:
            images_to_keep.append(img)

    # 如果这个 JSON 里没有包含我们要删的图片，直接跳过
    if not deleted_image_ids:
        print("   -> 🟢 没在此数据集中找到目标图片，无需操作。")
        return

    # 3. 过滤 annotations：剔除 image_id 在 deleted_image_ids 中的所有标注
    annotations_to_keep = [
        anno for anno in coco_data.get('annotations', [])
        if anno['image_id'] not in deleted_image_ids
    ]

    # 4. 更新 JSON 数据并写回
    coco_data['images'] = images_to_keep
    coco_data['annotations'] = annotations_to_keep

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    # 5. 物理删除对应的图片文件
    physical_delete_count = 0
    for filename in deleted_filenames:
        file_path = os.path.join(img_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            physical_delete_count += 1

    print(f"   ✅ JSON 更新完毕:")
    print(f"      - 图片记录: {original_img_count} -> {len(images_to_keep)} (清理了 {len(deleted_image_ids)} 条)")
    print(f"      - 标注记录: {original_anno_count} -> {len(annotations_to_keep)} (清理了 {original_anno_count - len(annotations_to_keep)} 个 BBox)")
    print(f"   🗑️ 物理删除图片: 删除了 {physical_delete_count} 张 (位于 {os.path.basename(img_dir)})")


def main():
    print("🚨 开始执行 COCO 数据集安全删除脚本...")
    
    if not os.path.exists(IMG_DIR):
        print(f"❌ 错误: 找不到存放待删除图片的文件夹 {IMG_DIR}")
        return

    # 获取 single 文件夹下所有的文件名（只取文件名，忽略文件夹）
    valid_exts = ['.jpg', '.jpeg', '.png']
    filenames_to_delete = {
        f for f in os.listdir(IMG_DIR) 
        if os.path.splitext(f)[1].lower() in valid_exts
    }
    
    if not filenames_to_delete:
        print("⚠️ single 文件夹是空的，或没有有效图片格式。")
        return

    print(f"🔍 在 single 文件夹中锁定了 {len(filenames_to_delete)} 个目标文件。")

    # 分别从 train 和 val 中尝试查找并删除
    remove_data_from_split(filenames_to_delete, TRAIN_IMG_DIR, TRAIN_JSON)
    remove_data_from_split(filenames_to_delete, VAL_IMG_DIR, VAL_JSON)

    print("\n🎉 所有删除操作圆满结束！你的数据集现在是干净的了。")

if __name__ == '__main__':
    main()