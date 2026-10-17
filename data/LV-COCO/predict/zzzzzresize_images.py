import os
from PIL import Image

def resize_images_in_directory():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置输出文件夹（为了安全，不直接覆盖原图）
    output_dir = os.path.join(current_dir, 'resized')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义目标尺寸 (宽, 高) - 注意：PIL的尺寸元组是 (width, height)
    target_size = (640, 480)
    
    # 支持的常见图片后缀名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
    
    # 统计处理数量
    success_count = 0

    print(f"正在扫描文件夹: {current_dir}")
    print(f"目标尺寸: 宽 640 x 高 480\n" + "-"*30)

    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        # 忽略大小写检查后缀名
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(current_dir, filename)
            
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 强制缩放，使用 LANCZOS 算法保证缩放后的图片质量
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 拼装输出路径并保存
                    output_path = os.path.join(output_dir, filename)
                    resized_img.save(output_path)
                    
                    print(f"✅ 成功缩放: {filename}")
                    success_count += 1
            except Exception as e:
                print(f"❌ 处理 {filename} 时出错: {e}")

    print("-" * 30)
    print(f"处理完毕！共成功缩放 {success_count} 张图片。")
    print(f"图片已保存至: {output_dir}")

if __name__ == "__main__":
    resize_images_in_directory()