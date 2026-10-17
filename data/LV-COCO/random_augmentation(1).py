import os
import cv2
import random
import numpy as np

# --- 配置路径 ---
IMG_DIR = r"D:\desktop\FPGA\Project\PaddleDetection\dataset\coco\single"
AUGMENT_COUNT = 3  # 每张图生成 3 张增强图

def random_augmentation(img):
    """中度图像增强：每次随机抽取 1-3 种效果进行叠加"""
    h, w, c = img.shape
    
    # 随机抽取 1 到 3 种效果
    aug_types = random.sample([1, 2, 3, 4, 5], random.randint(1, 3))
    aug_img = img.copy()
    
    for aug in aug_types:
        if aug == 1:
            # 1. 中度高斯模糊
            ksize = random.choice([3, 5])
            aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
        elif aug == 2:
            # 2. 中度提高亮度
            beta = random.randint(20, 50)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=1.0, beta=beta)
        elif aug == 3:
            # 3. 中度降低亮度
            beta = random.randint(-50, -20)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=1.0, beta=beta)
        elif aug == 4:
            # 4. 中度改变对比度
            alpha = random.uniform(0.8, 1.3)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=0)
        elif aug == 5:
            # 5. 中度高斯噪声 (比极轻微版明显，但不会盖住物体)
            std_dev = random.uniform(8, 12) 
            noise = np.random.normal(0, std_dev, img.shape).astype(np.float32)
            aug_img = cv2.add(aug_img.astype(np.float32), noise)
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
            
    return aug_img

def process_dataset():
    # 检查路径是否存在
    if not os.path.exists(IMG_DIR):
        print(f"❌ 错误: 找不到图片文件夹 {IMG_DIR}")
        return
        
    valid_exts = ['.jpg', '.jpeg', '.png']
    img_files = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if len(img_files) == 0:
        print("⚠️ 警告：文件夹里没有找到图片，请检查路径是否正确！")
        return

    # 统计真正的原图数量（排除之前生成的 aug 图）
    original_imgs = [f for f in img_files if "_aug_" not in f]
    print(f"🔍 找到 {len(original_imgs)} 张原始图片，准备为每张图片生成 {AUGMENT_COUNT} 张增强图...")

    success_count = 0
    for img_name in original_imgs:
        base_name, ext = os.path.splitext(img_name)
        img_path = os.path.join(IMG_DIR, img_name)
        
        # 读取原图
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 警告: 无法读取图片 {img_name}，跳过。")
            continue
            
        # 生成 3 张增强图
        for i in range(1, AUGMENT_COUNT + 1):
            new_base_name = f"{base_name}_aug_{i}"
            new_img_name = new_base_name + ext
            new_img_path = os.path.join(IMG_DIR, new_img_name)
            
            # 图像增强并保存
            aug_img = random_augmentation(img)
            cv2.imwrite(new_img_path, aug_img)
                
        success_count += 1
        print(f"✅ [{success_count}/{len(original_imgs)}] {img_name} 扩充完成 (生成了3张)")
            
    print(f"\n🎉 所有数据增强完成！成功处理了 {success_count} 张原图，共生成 {success_count * AUGMENT_COUNT} 张新图。")

# --- 脚本入口 ---
if __name__ == "__main__":
    print("🚀 纯图片增强脚本开始运行...")
    process_dataset()