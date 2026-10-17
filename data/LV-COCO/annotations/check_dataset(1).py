import json
import os

def analyze_coco(json_path):
    """
    解析单个 COCO 格式的 JSON 文件，返回相关统计信息。
    """
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"⚠️ 警告: 找不到文件 {json_path}")
        return 0, 0, 0, {'划痕 (ca_shang)': 0, '脏污 (zang_wu)': 0, '褶皱 (zhe_zhou)': 0, '针孔 (zhen_kong)': 0}
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 统计图片总数
    total_images = len(data.get('images', []))
    
    # 2. 统计缺陷样本数（有标注的图片的集合）
    # 提取所有 annotation 中的 image_id 并去重
    annotated_image_ids = set([ann['image_id'] for ann in data.get('annotations', [])])
    defect_samples = len(annotated_image_ids)
    
    # 3. 统计纯净负样本（没有任何标注框的图片）
    pure_negative_samples = total_images - defect_samples
    
    # 4. 统计缺陷分布 (通过统计各种类别的 annotation 数量)
    defect_distribution = {
        '划痕 (ca_shang)': 0,
        '脏污 (zang_wu)': 0,
        '褶皱 (zhe_zhou)': 0,
        '针孔 (zhen_kong)': 0
    }
    
    category_map = {
        1: '划痕 (ca_shang)',
        2: '脏污 (zang_wu)',
        3: '褶皱 (zhe_zhou)',
        4: '针孔 (zhen_kong)'
    }
    
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        if cat_id in category_map:
            defect_distribution[category_map[cat_id]] += 1
            
    return total_images, defect_samples, pure_negative_samples, defect_distribution

def main():
    # 因为你当前已经在 annotations 目录下，直接用文件名即可
    train_json = "instances_train2017.json"
    test_json = "instances_val2017.json"
    
    print("正在解析数据集，请稍候...")
    
    # 分别解析训练集和验证/测试集
    train_total, train_defect, train_pure, train_dist = analyze_coco(train_json)
    test_total, test_defect, test_pure, test_dist = analyze_coco(test_json)
    
    # 计算全局总和
    total_samples = train_total + test_total
    total_defect = train_defect + test_defect
    total_pure = train_pure + test_pure
    
    # 打印全局统计信息
    print("\n" + "="*45)
    print("🌟 总体样本情况")
    print("="*45)
    print(f"全部样本总数:      {total_samples} 张")
    print(f"包含缺陷的样本总数: {total_defect} 张")
    print(f"纯净负样本总数:    {total_pure} 张")
    
    # 打印分类统计信息
    print("\n" + "="*45)
    print("📊 缺陷分布 (按标注框的数量)")
    print("="*45)
    all_dist = {k: train_dist[k] + test_dist[k] for k in train_dist}
    for k, v in all_dist.items():
        print(f" - {k.ljust(15)}: {v} 个")
        
    # 打印数据集划分比例
    print("\n" + "="*45)
    print("📁 数据集划分比例")
    print("="*45)
    if total_samples > 0:
        train_ratio = (train_total / total_samples) * 100
        test_ratio = (test_total / total_samples) * 100
    else:
        train_ratio = test_ratio = 0
        
    print(f"训练集 (Train) 数量: {train_total} 张 ({train_ratio:.2f}%)")
    print(f"验证集 (Test) 数量:  {test_total} 张 ({test_ratio:.2f}%)")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()